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
use crate::tokenizer::adapter::decode_complete;
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

const CONSTRAINED_DECODING_FIELDS: [&str; 4] = ["json_schema", "regex", "ebnf", "structural_tag"];
const FIELDS_BLOCKS_CANNOT_JOIN: [&str; 4] = [
    "return_hidden_states",
    "return_routed_experts",
    "return_flat_raw_top_logprobs",
    "session_params",
];
const DONE_EVENT: &[u8] = b"data: [DONE]\n\n";
const EVENT_BUFFER: usize = 64;

type Events = mpsc::Sender<Result<Bytes, axum::Error>>;

enum FramesEnd {
    BlockCut,
    ChainEnded,
    Failed,
    StreamClosed,
}

/// A Franky request served as a chain of blocks (`--franky-block-size`), each sending
/// the prompt and every token generated so far to the same Sandy and Ricky.
pub(super) struct BlockChain {
    id: String,
    model: ModelId,
    block_size: u64,
    total_limit: Option<u64>,
    body_without_prompt: Map<String, Value>,
    prompt_ids: Vec<Value>,
    /// The client's own `extra_key` for Sandy's leg; `body_without_prompt`
    /// carries Ricky's, keyed by her view of the prompt.
    sandy_extra_key: Option<Value>,
    /// Sandy's own view of the prompt, when the client sent one; each block
    /// gives her that view and every token generated so far.
    sandy_prompt_ids: Option<Vec<u32>>,
    /// The token that ends an answer at a block's end (`--franky-boundary-token-id`),
    /// unless the client stops on it.
    boundary: Option<u32>,
    /// The request's answers before the current one, as `[start, end)` runs of
    /// the block's `input_ids`; each later block lists them as
    /// `franky_generated_spans` after the client's own.
    earlier_answers: Vec<[usize; 2]>,
    /// Where the current answer starts among the generated tokens.
    answer_start: usize,
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
            return Err(refused(&format!(
                "{} requests it does not lower to /generate; it lowers chat only \
                 for chat templates it renders as SGLang does, so send /generate \
                 or /v1/completions",
                request.path
            )));
        }
        if request.fans_out {
            return Err(refused("batches or n above 1"));
        }
        let Ok(Value::Object(mut body)) = serde_json::from_slice(&request.body) else {
            return Err(refused("a body that is not a JSON object"));
        };
        check_blocks_can_serve(&body)?;
        let prompt_ids = match body.remove("input_ids") {
            Some(Value::Array(ids)) if ids.iter().all(Value::is_u64) => ids,
            _ => {
                return Err(refused(
                    "a prompt without token ids; send input_ids or text the router tokenizes",
                ))
            }
        };
        body.remove("text");
        let sandy_prompt_ids = request.sandy_input_ids.take();
        let boundary = boundary_of(ctx, &body)?;
        let client_rid = body.remove("rid");
        let total_limit = body
            .get("sampling_params")
            .and_then(|params| params.get("max_new_tokens"))
            .and_then(Value::as_u64);
        let id = uuid::Uuid::new_v4().simple().to_string();
        let reported_id = client_rid
            .as_ref()
            .and_then(Value::as_str)
            .map_or_else(|| id.clone(), str::to_owned);
        let incremental_streaming_output = model_openai_settings(ctx, &request.model)
            .is_some_and(|settings| settings.incremental_streaming_output);
        let boundary_text = match boundary {
            Some(boundary) => boundary_text(ctx, &request.model, &body, boundary)?,
            None => None,
        };
        Ok(Some(Self {
            output: ChainOutput::new(
                reported_id,
                incremental_streaming_output,
                boundary,
                boundary_text,
            ),
            end: ChainEnd::new(ctx.proxy.admin_client().clone(), id.clone(), headers),
            id,
            model: request.model.clone(),
            block_size: u64::from(block_size.get()),
            total_limit,
            body_without_prompt: body,
            prompt_ids,
            sandy_extra_key: request.sandy_extra_key.take(),
            sandy_prompt_ids,
            boundary,
            earlier_answers: Vec::new(),
            answer_start: 0,
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
        if let Some(view) = &self.sandy_prompt_ids {
            let generated = self.output.generated_ids().iter().filter_map(Value::as_u64);
            let view = view.iter().copied().chain(generated.map(|id| id as u32));
            request.sandy_input_ids = Some(view.collect());
            request.sandy_extra_key = self.sandy_extra_key.clone();
        }
        self.output.start_block(max_new_tokens, last_block);
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
            self.end.remember_engines(&workers);
            self.workers = Some(workers);
        }
        Ok(response)
    }

    fn block_body(&self, max_new_tokens: u64, last_block: bool) -> Value {
        let mut body = self.body_without_prompt.clone();
        let context = self.prompt_ids.iter().chain(self.output.generated_ids());
        body.insert("input_ids".into(), Value::Array(context.cloned().collect()));
        body.insert(
            "rid".into(),
            format!("{}-{}", self.id, self.blocks_sent).into(),
        );
        body.insert(
            "franky_chain".into(),
            json!({
                "id": self.id,
                "block": self.blocks_sent,
                "block_size": self.block_size,
                "last_block": last_block,
            }),
        );
        let sampling = body.entry("sampling_params").or_insert_with(|| json!({}));
        sampling["max_new_tokens"] = max_new_tokens.into();
        if let Some(boundary) = self.boundary {
            let stops = &mut sampling["stop_token_ids"];
            if stops.is_null() {
                *stops = json!([]);
            }
            if let Value::Array(stops) = stops {
                stops.push(boundary.into());
            }
        }
        if !self.earlier_answers.is_empty() {
            let mut spans = match body.remove("franky_generated_spans") {
                Some(Value::Array(spans)) => spans,
                _ => Vec::new(),
            };
            spans.extend(self.earlier_answers.iter().map(|span| json!(span)));
            body.insert("franky_generated_spans".into(), Value::Array(spans));
        }
        if !self.output.generated_ids().is_empty() && body.contains_key("logprob_start_len") {
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
            let chain_frame = self.output.whole_block_frame(frame);
            if !self.output.block_cut() {
                self.ended();
                parts.headers.remove(header::CONTENT_LENGTH);
                return Ok(Response::from_parts(
                    parts,
                    Body::from(chain_frame.to_string()),
                ));
            }
            self.log_block();
            self.output.finish_block();
            self.start_answer_after_boundary();
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
                FramesEnd::BlockCut => {}
                FramesEnd::ChainEnded => {
                    self.ended();
                    let _ = events.send(Ok(Bytes::from_static(DONE_EVENT))).await;
                    return;
                }
                FramesEnd::Failed => {
                    let _ = events.send(Ok(Bytes::from_static(DONE_EVENT))).await;
                    return;
                }
                FramesEnd::StreamClosed => return,
            }
            self.log_block();
            self.output.finish_block();
            self.start_answer_after_boundary();
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
                    let _ = events
                        .send(Ok(error_event_from_response(response).await))
                        .await;
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

    async fn forward_frames(&mut self, body: Body, events: &Events) -> FramesEnd {
        let mut frames = std::pin::pin!(sse_data(body));
        while let Some(data) = frames.next().await {
            let data = match data {
                Ok(data) => data,
                Err(error) => {
                    let _ = events.send(Err(error)).await;
                    return FramesEnd::StreamClosed;
                }
            };
            if data.as_ref() == b"[DONE]" {
                continue;
            }
            let frame = match serde_json::from_slice::<Value>(&data) {
                Ok(frame) if frame.get("error").is_none() => frame,
                _ => {
                    let _ = events.send(Ok(data_event(&data))).await;
                    return FramesEnd::Failed;
                }
            };
            let event = data_event(self.output.streamed_frame(frame).to_string().as_bytes());
            if events.send(Ok(event)).await.is_err() {
                return FramesEnd::StreamClosed;
            }
        }
        if self.output.block_cut() {
            FramesEnd::BlockCut
        } else {
            FramesEnd::ChainEnded
        }
    }

    /// After a block that stopped on the boundary token, the next block starts
    /// a new answer. Its engines' chain counts blocks of one size from the
    /// answer's start, so the chain ends on them and goes on under a new id
    /// whose blocks list every earlier answer.
    fn start_answer_after_boundary(&mut self) {
        if !self.output.ended_on_boundary() {
            return;
        }
        let generated = self.output.generated_ids().len();
        let prompt = self.prompt_ids.len();
        self.earlier_answers
            .push([prompt + self.answer_start, prompt + generated]);
        self.answer_start = generated;
        self.id = uuid::Uuid::new_v4().simple().to_string();
        self.end.restart(self.id.clone());
        self.blocks_sent = 0;
    }

    fn ended(&mut self) {
        self.log_block();
        if self.output.on_last_block() {
            self.end.ended_with_last_block();
        }
    }

    fn log_block(&self) {
        let report = self.output.block_report();
        tracing::debug!(
            chain = %self.id,
            block = self.blocks_sent,
            prompt_tokens = %report.get("prompt_tokens").unwrap_or(&serde_json::Value::Null),
            cached_tokens = %report.get("cached_tokens").unwrap_or(&serde_json::Value::Null),
            report = %serde_json::Value::Object(report.clone()),
            "franky block ended"
        );
    }
}

/// The text of the boundary token that SGLang leaves out of a block stopped on
/// it, and the client must still get: the token decoded alone, as the client's
/// `skip_special_tokens` asks. `None` when the client keeps matched stops in
/// its text (`no_stop_trim`), as the engines then do.
///
/// Errors: `ApiError::BadRequest` without the router's tokenizer, which
/// decodes it.
fn boundary_text(
    ctx: &AppContext,
    model: &ModelId,
    body: &Map<String, Value>,
    boundary: u32,
) -> Result<Option<String>, ApiError> {
    let flag = |name: &str| body.get("sampling_params")?.get(name)?.as_bool();
    if flag("no_stop_trim") == Some(true) {
        return Ok(None);
    }
    let Some(tokenizer) = ctx.tokenizers.get(&model.0) else {
        return Err(refused(
            "--franky-boundary-token-id without the router's tokenizer, which gives the \
             boundary token's text back, unless the request sets no_stop_trim",
        ));
    };
    let skip_special_tokens = flag("skip_special_tokens").unwrap_or(true);
    decode_complete(&tokenizer, &[boundary], skip_special_tokens)
        .map(Some)
        .map_err(ApiError::Internal)
}

/// The router's boundary token for a request, unless the client stops on it.
///
/// Errors: `ApiError::BadRequest` for `ignore_eos`, which turns off the stop
/// that ends an answer at the boundary.
fn boundary_of(ctx: &AppContext, body: &Map<String, Value>) -> Result<Option<u32>, ApiError> {
    let Some(boundary) = ctx.config.proxy.franky_boundary_token_id else {
        return Ok(None);
    };
    let sampling = body.get("sampling_params");
    let client_stops = sampling
        .and_then(|params| params.get("stop_token_ids"))
        .and_then(Value::as_array);
    if client_stops.is_some_and(|stops| {
        stops
            .iter()
            .any(|id| id.as_u64() == Some(u64::from(boundary)))
    }) {
        return Ok(None);
    }
    if sampling
        .and_then(|params| params.get("ignore_eos"))
        .is_some_and(is_set)
    {
        return Err(refused(
            "ignore_eos, which turns off the stop that ends an answer at --franky-boundary-token-id",
        ));
    }
    Ok(Some(boundary))
}

fn check_blocks_can_serve(body: &Map<String, Value>) -> Result<(), ApiError> {
    let sampling = body.get("sampling_params");
    let constrained = CONSTRAINED_DECODING_FIELDS.iter().any(|field| {
        sampling
            .and_then(|params| params.get(field))
            .is_some_and(is_set)
    });
    if constrained {
        return Err(refused(
            "constrained decoding, whose state each block would restart",
        ));
    }
    match FIELDS_BLOCKS_CANNOT_JOIN
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

async fn error_event_from_response(response: Response<Body>) -> Bytes {
    let status = response.status();
    let body = match axum::body::to_bytes(response.into_body(), usize::MAX).await {
        Ok(body) => body,
        Err(error) => {
            return error_event(&format!(
                "a Franky block failed with {status}, and reading its body failed: {error}"
            ))
        }
    };
    match serde_json::from_slice::<Value>(&body) {
        Ok(error) if error.get("error").is_some() => data_event(&body),
        _ => error_event(&format!(
            "a Franky block failed with {status}: {}",
            String::from_utf8_lossy(&body)
        )),
    }
}
