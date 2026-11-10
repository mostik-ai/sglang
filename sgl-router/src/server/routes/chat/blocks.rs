// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

mod end;
mod output;

use super::forward::{RequestDurationGuard, SelectedWorkers};
use super::openai::sse_data;
use super::preparation::{model_openai_settings, PreparedRequest, GENERATE_PATH};
use super::{serves_franky, ModelRouting};
use crate::discovery::ModelId;
use crate::server::app_context::AppContext;
use crate::server::error::ApiError;
use axum::body::Body;
use axum::http::{header, HeaderMap, Response, StatusCode};
use bytes::Bytes;
use end::ChainEnd;
use futures::StreamExt;
use output::ChainOutput;
use serde_json::{json, Map, Value};
use sglang_processor::openai::Responder;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;

/// Sampling fields whose decoding state a new block would restart.
const CONSTRAINED_DECODING: [&str; 4] = ["json_schema", "regex", "ebnf", "structural_tag"];
/// Request fields whose outputs the chain cannot join across blocks.
const UNJOINED_FIELDS: [&str; 4] = [
    "return_hidden_states",
    "return_routed_experts",
    "return_flat_raw_top_logprobs",
    "session_params",
];
const DONE_EVENT: &[u8] = b"data: [DONE]\n\n";
const EVENT_BUFFER: usize = 64;

type Events = mpsc::Sender<Result<Bytes, axum::Error>>;

/// How the frames of one block ended.
enum Pumped {
    NextBlock,
    Ended,
    Failed,
    Gone,
}

/// A Franky request served as a chain of blocks (`--franky-block-size`). Each
/// block sends the prompt and every token generated so far to the chain's one
/// Sandy and one Ricky, which generates at most a block of tokens; its tokens
/// then read their own taps in the next block. The engines keep the chain's
/// cached prefix between blocks; the client gets one response.
pub(super) struct BlockChain {
    id: String,
    model: ModelId,
    block_size: u64,
    total_limit: Option<u64>,
    /// The request's `/generate` body without its prompt and rid.
    template: Map<String, Value>,
    prompt_ids: Vec<Value>,
    streaming: bool,
    responder: Option<Responder>,
    output: ChainOutput,
    workers: Option<SelectedWorkers>,
    end: ChainEnd,
    blocks_sent: u32,
}

impl BlockChain {
    /// The chain that serves `request` when blocks are on and Franky workers
    /// serve its model; `None` otherwise.
    ///
    /// Errors: `ApiError::BadRequest` for a request blocks cannot serve.
    pub(super) fn plan(
        ctx: &AppContext,
        request: &mut PreparedRequest,
        headers: &HeaderMap,
    ) -> Result<Option<Self>, ApiError> {
        let Some(block_size) = ctx.config.proxy.franky_block_size else {
            return Ok(None);
        };
        if !serves_franky(ctx, &request.model) {
            return Ok(None);
        }
        if request.path != GENERATE_PATH {
            return Err(refused(
                "a request the router does not serve through /generate",
            ));
        }
        if request.fans_out {
            return Err(refused("batches or n above 1"));
        }
        let Ok(Value::Object(mut template)) = serde_json::from_slice(&request.body) else {
            return Err(refused("a body that is not a JSON object"));
        };
        check_blocks_can_serve(&template)?;
        let prompt_ids = match template.remove("input_ids") {
            Some(Value::Array(ids)) if ids.iter().all(Value::is_u64) => ids,
            _ => {
                return Err(refused(
                    "a prompt without token ids; send input_ids or text the router tokenizes",
                ))
            }
        };
        template.remove("text");
        let client_rid = template.remove("rid");
        let total_limit = template
            .get("sampling_params")
            .and_then(|params| params.get("max_new_tokens"))
            .and_then(Value::as_u64);
        let id = uuid::Uuid::new_v4().simple().to_string();
        let reported_id = client_rid
            .as_ref()
            .and_then(Value::as_str)
            .map_or_else(|| id.clone(), str::to_owned);
        let incremental = model_openai_settings(ctx, &request.model)
            .is_some_and(|settings| settings.incremental_streaming_output);
        Ok(Some(Self {
            output: ChainOutput::new(reported_id, incremental),
            end: ChainEnd::new(ctx.proxy.admin_client().clone(), id.clone(), headers),
            id,
            model: request.model.clone(),
            block_size: u64::from(block_size.get()),
            total_limit,
            template,
            prompt_ids,
            streaming: request.streaming,
            responder: request.responder.take(),
            workers: None,
            blocks_sent: 0,
        }))
    }

    /// Serve the chain block by block, every block on the workers that answered
    /// the first, as one response.
    pub(super) async fn serve(
        mut self,
        ctx: &Arc<AppContext>,
        routing: &ModelRouting<'_>,
        headers: HeaderMap,
        start: Instant,
    ) -> Result<Response<Body>, ApiError> {
        let mut responder = self.responder.take();
        let streaming = self.streaming;
        let mut duration = None;
        let first = self
            .send_block(ctx, routing, &headers, start, &mut duration)
            .await?;
        let response = if first.status() != StatusCode::OK {
            first
        } else if streaming {
            self.stream(Arc::clone(ctx), first, headers, start, duration)
        } else {
            self.collect(ctx, routing, first, &headers, start, &mut duration)
                .await?
        };
        Ok(match responder {
            Some(_) => super::openai::respond(&mut responder, response, streaming).await,
            None => response,
        })
    }

    async fn send_block(
        &mut self,
        ctx: &AppContext,
        routing: &ModelRouting<'_>,
        headers: &HeaderMap,
        start: Instant,
        duration: &mut Option<Arc<RequestDurationGuard>>,
    ) -> Result<Response<Body>, ApiError> {
        let generated = self.output.generated_ids().len() as u64;
        let max_new_tokens = self.total_limit.map_or(self.block_size, |limit| {
            self.block_size.min(limit.saturating_sub(generated))
        });
        let last_block = self
            .total_limit
            .is_some_and(|limit| generated + max_new_tokens >= limit);
        self.blocks_sent += 1;
        let body = self.block_body(max_new_tokens, last_block);
        let mut request = PreparedRequest::generate_value(ctx, self.model.clone(), body, None)?;
        self.output.start_block(last_block);
        let (response, answered) = routing
            .forward_with_retries(
                ctx,
                &mut request,
                headers,
                start,
                duration,
                self.workers.clone(),
            )
            .await?;
        if let Some(workers) = answered {
            self.end.reach(&workers);
            self.workers = Some(workers);
        }
        Ok(response)
    }

    fn block_body(&self, max_new_tokens: u64, last_block: bool) -> Value {
        let mut body = self.template.clone();
        let context = self.prompt_ids.iter().chain(self.output.generated_ids());
        body.insert("input_ids".into(), Value::Array(context.cloned().collect()));
        body.insert(
            "rid".into(),
            format!("{}-{}", self.id, self.blocks_sent).into(),
        );
        body.insert(
            "franky_chain".into(),
            json!({"id": self.id, "last_block": last_block}),
        );
        let sampling = body.entry("sampling_params").or_insert_with(|| json!({}));
        sampling["max_new_tokens"] = max_new_tokens.into();
        if self.blocks_sent > 1 && body.contains_key("logprob_start_len") {
            body.insert("logprob_start_len".into(), (-1).into());
        }
        Value::Object(body)
    }

    async fn collect(
        &mut self,
        ctx: &AppContext,
        routing: &ModelRouting<'_>,
        mut response: Response<Body>,
        headers: &HeaderMap,
        start: Instant,
        duration: &mut Option<Arc<RequestDurationGuard>>,
    ) -> Result<Response<Body>, ApiError> {
        loop {
            if response.status() != StatusCode::OK {
                return Ok(response);
            }
            let (mut parts, body) = response.into_parts();
            let bytes = axum::body::to_bytes(body, usize::MAX)
                .await
                .map_err(|error| ApiError::Internal(error.into()))?;
            let frame =
                serde_json::from_slice(&bytes).map_err(|error| ApiError::Internal(error.into()))?;
            let chain_frame = self.output.chain_frame(frame, true);
            if !self.output.block_cut() {
                self.ended();
                parts.headers.remove(header::CONTENT_LENGTH);
                return Ok(Response::from_parts(
                    parts,
                    Body::from(chain_frame.to_string()),
                ));
            }
            self.output.finish_block();
            response = self
                .send_block(ctx, routing, headers, start, duration)
                .await?;
        }
    }

    fn stream(
        self,
        ctx: Arc<AppContext>,
        first: Response<Body>,
        headers: HeaderMap,
        start: Instant,
        duration: Option<Arc<RequestDurationGuard>>,
    ) -> Response<Body> {
        let (mut parts, body) = first.into_parts();
        parts.headers.remove(header::CONTENT_LENGTH);
        let (events, receiver) = mpsc::channel(EVENT_BUFFER);
        tokio::spawn(self.pump(ctx, body, events, headers, start, duration));
        let stream = futures::stream::unfold(receiver, |mut receiver| async move {
            let event = receiver.recv().await?;
            Some((event, receiver))
        });
        Response::from_parts(parts, Body::from_stream(stream))
    }

    async fn pump(
        mut self,
        ctx: Arc<AppContext>,
        mut body: Body,
        events: Events,
        headers: HeaderMap,
        start: Instant,
        mut duration: Option<Arc<RequestDurationGuard>>,
    ) {
        loop {
            match self.forward_frames(body, &events).await {
                Pumped::NextBlock => {}
                Pumped::Ended => {
                    self.ended();
                    let _ = events.send(Ok(Bytes::from_static(DONE_EVENT))).await;
                    return;
                }
                Pumped::Failed => {
                    let _ = events.send(Ok(Bytes::from_static(DONE_EVENT))).await;
                    return;
                }
                Pumped::Gone => return,
            }
            self.output.finish_block();
            let next = match ModelRouting::lookup(&ctx, &self.model) {
                Ok(routing) => {
                    self.send_block(&ctx, &routing, &headers, start, &mut duration)
                        .await
                }
                Err(error) => Err(error),
            };
            body = match next {
                Ok(response) if response.status() == StatusCode::OK => response.into_body(),
                Ok(response) => {
                    let _ = events.send(Ok(error_event_of(response).await)).await;
                    let _ = events.send(Ok(Bytes::from_static(DONE_EVENT))).await;
                    return;
                }
                Err(error) => {
                    let _ = events.send(Ok(error_event(&error.to_string()))).await;
                    let _ = events.send(Ok(Bytes::from_static(DONE_EVENT))).await;
                    return;
                }
            };
        }
    }

    async fn forward_frames(&mut self, body: Body, events: &Events) -> Pumped {
        let mut frames = std::pin::pin!(sse_data(body));
        while let Some(data) = frames.next().await {
            let data = match data {
                Ok(data) => data,
                Err(error) => {
                    let _ = events.send(Err(error)).await;
                    return Pumped::Gone;
                }
            };
            if data.as_ref() == b"[DONE]" {
                continue;
            }
            let frame = match serde_json::from_slice::<Value>(&data) {
                Ok(frame) if frame.get("error").is_none() => frame,
                _ => {
                    let _ = events.send(Ok(data_event(&data))).await;
                    return Pumped::Failed;
                }
            };
            let event = data_event(self.output.chain_frame(frame, false).to_string().as_bytes());
            if events.send(Ok(event)).await.is_err() {
                return Pumped::Gone;
            }
        }
        if self.output.block_cut() {
            Pumped::NextBlock
        } else {
            Pumped::Ended
        }
    }

    fn ended(&mut self) {
        if self.output.on_last_block() {
            self.end.ended_with_last_block();
        }
    }
}

fn check_blocks_can_serve(body: &Map<String, Value>) -> Result<(), ApiError> {
    let sampling = body.get("sampling_params");
    let constrained = CONSTRAINED_DECODING.iter().any(|field| {
        sampling
            .and_then(|params| params.get(field))
            .is_some_and(is_set)
    });
    if constrained {
        return Err(refused(
            "constrained decoding, whose state each block would restart",
        ));
    }
    match UNJOINED_FIELDS
        .iter()
        .find(|field| body.get(**field).is_some_and(is_set))
    {
        Some(field) => Err(refused(&format!("`{field}`"))),
        None => Ok(()),
    }
}

fn is_set(value: &Value) -> bool {
    !matches!(value, Value::Null | Value::Bool(false))
}

fn refused(what: &str) -> ApiError {
    ApiError::BadRequest(format!(
        "Franky blocks (--franky-block-size) cannot serve {what}"
    ))
}

fn data_event(data: &[u8]) -> Bytes {
    [b"data: ".as_slice(), data, b"\n\n"].concat().into()
}

fn error_event(message: &str) -> Bytes {
    data_event(
        json!({"error": {"message": message}})
            .to_string()
            .as_bytes(),
    )
}

/// A later block's error response, as an in-stream error frame.
async fn error_event_of(response: Response<Body>) -> Bytes {
    let status = response.status();
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap_or_default();
    match serde_json::from_slice::<Value>(&body) {
        Ok(error) if error.get("error").is_some() => data_event(&body),
        _ => error_event(&format!(
            "a Franky block failed with {status}: {}",
            String::from_utf8_lossy(&body)
        )),
    }
}
