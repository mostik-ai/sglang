// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

mod answer_limits;
mod end;
mod output;
mod ricky_leg;
mod sandy_call;
mod trace;

use super::forward::{forward_request, RequestDurationGuard, SelectedWorkers};
use super::openai::sse_data;
use super::preparation::{
    generate_room_id, model_openai_settings, FrankyRoom, PreparedRequest, TapSource, GENERATE_PATH,
};
use super::tap_runs::wire;
use super::{back_off, pick_sandy, serves_franky, ModelRouting};
use crate::config::FrankyChainConfig;
use crate::discovery::{ModelId, WorkerId};
use crate::policies_reorg::cache_aware::PrefixMemo;
use crate::proxy::AbortOnDrop;
use crate::server::app_context::AppContext;
use crate::server::error::ApiError;
use crate::workers::Worker;
use answer_limits::AnswerLimits;
use axum::body::Body;
use axum::http::response::Parts;
use axum::http::{header, HeaderMap, HeaderValue, Response, StatusCode};
use bytes::Bytes;
use end::ChainEnd;
use futures::StreamExt;
use output::ChainOutput;
use ricky_leg::{ricky_leg_body, RickyLegFields};
use sandy_call::{read_sandy_text, sandy_call_body, SandyText};
use serde_json::{json, Map, Value};
use sglang_processor::openai::Responder;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;
use trace::ChainTrace;

const CONSTRAINED_DECODING_FIELDS: [&str; 4] = ["json_schema", "regex", "ebnf", "structural_tag"];
const FIELDS_CHAINS_CANNOT_JOIN: [&str; 4] = [
    "return_hidden_states",
    "return_routed_experts",
    "return_flat_raw_top_logprobs",
    "session_params",
];
/// Per-token output, which Sandy's text has none of in Ricky's terms.
const PER_TOKEN_OUTPUT_FIELDS: [&str; 3] =
    ["return_logprob", "top_logprobs_num", "token_ids_logprob"];
/// The fields the chain sets on each of its requests.
const CHAIN_FIELDS: [&str; 6] = [
    "input_ids",
    "text",
    "rid",
    "franky_tap_runs",
    "franky_sandy_input_ids",
    "franky_chain",
];
const DONE_EVENT: &[u8] = b"data: [DONE]\n\n";
const EVENT_BUFFER: usize = 64;

type Events = mpsc::Sender<Result<Bytes, axum::Error>>;
type Duration = Option<Arc<RequestDurationGuard>>;

/// What one request of a chain answered.
enum Answer {
    /// A Sandy call wrote `text`; with taps, the next leg reads them in `room`.
    SandyText {
        head: Parts,
        text: SandyText,
        room: Option<Box<SandyRoom>>,
    },
    /// A Ricky leg answered with `body`, unary or streamed.
    Leg { head: Parts, body: Body },
    /// The request failed; the client gets its response.
    Failed(Response<Body>),
}

/// The room a Sandy call sent its taps to. Sandy holds the call until a Ricky
/// leg reads them, so unless one answers for it, dropping it aborts the call.
struct SandyRoom {
    room: FrankyRoom,
    unread: AbortOnDrop,
}

/// How a streamed Ricky leg ended.
enum FramesEnd {
    CallsSandy,
    AnswerEnded,
    Failed,
    StreamClosed,
}

/// A Franky request served as a chain (`--franky-chain`): Sandy calls that
/// write part of the answer, and Ricky legs that write blocks until Ricky's
/// engine ends a leg for the next Sandy call. Each request carries the whole
/// trace so far and goes to the chain's Sandy or Ricky, whose caches keep it.
pub(super) struct FrankyChain {
    id: String,
    model: ModelId,
    config: FrankyChainConfig,
    limits: AnswerLimits,
    body_without_prompt: Map<String, Value>,
    client_stop_ids: Vec<u64>,
    trace: ChainTrace,
    streaming: bool,
    responder: Option<Responder>,
    output: ChainOutput,
    ricky: Option<SelectedWorkers>,
    sandy: Option<Arc<Worker>>,
    end: Option<ChainEnd>,
    block: u64,
    requests_sent: u32,
}

impl FrankyChain {
    /// The chain that serves `request` when `--franky-chain` runs chains and
    /// Franky workers serve its model; `None` otherwise.
    ///
    /// Errors: `ApiError::BadRequest` for a request chains cannot serve.
    pub(super) fn plan(
        ctx: &AppContext,
        request: &mut PreparedRequest,
        headers: &HeaderMap,
    ) -> Result<Option<Self>, ApiError> {
        let config = &ctx.config.proxy.franky_chain;
        if !config.runs_chains() || !serves_franky(ctx, &request.model) {
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
        check_chains_can_serve(&body, config.sandy_call_tokens > 0)?;
        let prompt_ids = match body.remove("input_ids") {
            Some(Value::Array(ids)) if ids.iter().all(Value::is_u64) => ids,
            _ => {
                return Err(refused(
                    "a prompt without token ids; send input_ids or text the router tokenizes",
                ))
            }
        };
        let client_rid = body.remove("rid");
        for field in CHAIN_FIELDS {
            body.remove(field);
        }
        let sampling = body.get("sampling_params").unwrap_or(&Value::Null);
        let sandy = match ctx.franky_sandy.clone() {
            None => None,
            Some(tokens) => {
                let Some(prompt_ids) = request.take_franky_sandy_ids() else {
                    return Err(refused("a prompt the router did not render for Sandy"));
                };
                Some((tokens, prompt_ids))
            }
        };
        let client_stop_ids = match (&sandy, &sampling["stop_token_ids"]) {
            (None, Value::Array(ids)) => ids.iter().filter_map(Value::as_u64).collect(),
            _ => Vec::new(),
        };
        let limits = AnswerLimits::of_sampling(sampling);
        let id = uuid::Uuid::new_v4().simple().to_string();
        let reported_id = client_rid
            .as_ref()
            .and_then(Value::as_str)
            .map_or_else(|| id.clone(), str::to_owned);
        let incremental_streaming_output = model_openai_settings(ctx, &request.model)
            .is_some_and(|settings| settings.incremental_streaming_output);
        Ok(Some(Self {
            output: ChainOutput::new(
                reported_id,
                incremental_streaming_output,
                prompt_ids.len() as u64,
            ),
            end: config
                .blocks
                .is_some()
                .then(|| ChainEnd::new(ctx.proxy.admin_client().clone(), id.clone(), headers)),
            trace: ChainTrace::of_prompt(prompt_ids, sandy, request.franky_earlier_runs().to_vec()),
            id,
            model: request.model.clone(),
            config: config.clone(),
            limits,
            body_without_prompt: body,
            client_stop_ids,
            streaming: request.streaming,
            responder: request.responder.take(),
            ricky: None,
            sandy: None,
            block: 1,
            requests_sent: 0,
        }))
    }

    /// Serve the chain, Sandy call and Ricky leg after leg, as one response.
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
            .next_answer(ctx, routing, &headers, start, &mut duration)
            .await?;
        let response = match first {
            Answer::Failed(response) => response,
            first if streaming => self.stream(Arc::clone(ctx), first, headers, start, duration),
            first => {
                self.collect(ctx, routing, first, &headers, start, &mut duration)
                    .await?
            }
        };
        Ok(match responder {
            Some(_) => super::openai::respond(&mut responder, response, streaming).await,
            None => response,
        })
    }

    /// The chain's next request: a Sandy call when Sandy writes, else a Ricky
    /// leg.
    async fn next_answer(
        &mut self,
        ctx: &AppContext,
        routing: &ModelRouting<'_>,
        headers: &HeaderMap,
        start: Instant,
        duration: &mut Duration,
    ) -> Result<Answer, ApiError> {
        if self.config.sandy_call_tokens > 0 {
            self.call_sandy(ctx, headers, start, duration).await
        } else {
            self.send_ricky_leg(ctx, routing, headers, start, duration, None)
                .await
        }
    }

    async fn collect(
        &mut self,
        ctx: &AppContext,
        routing: &ModelRouting<'_>,
        mut answer: Answer,
        headers: &HeaderMap,
        start: Instant,
        duration: &mut Duration,
    ) -> Result<Response<Body>, ApiError> {
        loop {
            let (head, body) = match answer {
                Answer::Failed(response) => return Ok(response),
                Answer::SandyText { head, text, room } => {
                    self.add_sandy_text(&text);
                    if text.ends_answer {
                        let whole = self.output.whole_answer(&text.finish_reason);
                        return Ok(unary_response(head, &whole));
                    }
                    answer = self
                        .send_ricky_leg(ctx, routing, headers, start, duration, room)
                        .await?;
                    continue;
                }
                Answer::Leg { head, body } => (head, body),
            };
            let bytes = axum::body::to_bytes(body, usize::MAX)
                .await
                .map_err(|error| ApiError::Internal(error.into()))?;
            let frame =
                serde_json::from_slice(&bytes).map_err(|error| ApiError::Internal(error.into()))?;
            let chain_frame = self.output.whole_leg_frame(frame);
            if !self.output.calls_sandy() {
                self.log_leg();
                return Ok(unary_response(head, &chain_frame));
            }
            self.finish_leg()?;
            answer = self
                .next_answer(ctx, routing, headers, start, duration)
                .await?;
        }
    }

    fn stream(
        self,
        ctx: Arc<AppContext>,
        mut first: Answer,
        headers: HeaderMap,
        start: Instant,
        duration: Duration,
    ) -> Response<Body> {
        let mut head = match &mut first {
            Answer::SandyText { head, .. } | Answer::Leg { head, .. } => {
                std::mem::replace(head, empty_head())
            }
            Answer::Failed(_) => empty_head(),
        };
        head.headers.remove(header::CONTENT_LENGTH);
        head.headers.insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("text/event-stream"),
        );
        let (events, receiver) = mpsc::channel(EVENT_BUFFER);
        tokio::spawn(self.pump(ctx, first, events, headers, start, duration));
        let stream = futures::stream::unfold(receiver, |mut receiver| async move {
            let event = receiver.recv().await?;
            Some((event, receiver))
        });
        Response::from_parts(head, Body::from_stream(stream))
    }

    async fn pump(
        mut self,
        ctx: Arc<AppContext>,
        mut answer: Answer,
        events: Events,
        headers: HeaderMap,
        start: Instant,
        mut duration: Duration,
    ) {
        loop {
            let body = match answer {
                Answer::Failed(response) => {
                    let _ = events
                        .send(Ok(error_event_from_response(response).await))
                        .await;
                    let _ = events.send(Ok(Bytes::from_static(DONE_EVENT))).await;
                    return;
                }
                Answer::SandyText { text, room, .. } => {
                    let frame = self.add_sandy_text(&text);
                    if events.send(Ok(json_event(&frame))).await.is_err() {
                        return;
                    }
                    if text.ends_answer {
                        let _ = events.send(Ok(Bytes::from_static(DONE_EVENT))).await;
                        return;
                    }
                    let next = match ModelRouting::lookup(&ctx, &self.model) {
                        Ok(routing) => {
                            self.send_ricky_leg(
                                &ctx,
                                &routing,
                                &headers,
                                start,
                                &mut duration,
                                room,
                            )
                            .await
                        }
                        Err(error) => Err(error),
                    };
                    answer = match next {
                        Ok(answer) => answer,
                        Err(error) => return fail_stream(&events, &error.to_string()).await,
                    };
                    continue;
                }
                Answer::Leg { body, .. } => body,
            };
            match self.forward_frames(body, &events).await {
                FramesEnd::CallsSandy => {}
                FramesEnd::AnswerEnded => {
                    self.log_leg();
                    let _ = events.send(Ok(Bytes::from_static(DONE_EVENT))).await;
                    return;
                }
                FramesEnd::Failed => {
                    let _ = events.send(Ok(Bytes::from_static(DONE_EVENT))).await;
                    return;
                }
                FramesEnd::StreamClosed => return,
            }
            if let Err(error) = self.finish_leg() {
                return fail_stream(&events, &error.to_string()).await;
            }
            let next = match ModelRouting::lookup(&ctx, &self.model) {
                Ok(routing) => {
                    self.next_answer(&ctx, &routing, &headers, start, &mut duration)
                        .await
                }
                Err(error) => Err(error),
            };
            answer = match next {
                Ok(answer) => answer,
                Err(error) => return fail_stream(&events, &error.to_string()).await,
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
            let event = json_event(&self.output.streamed_frame(frame));
            if events.send(Ok(event)).await.is_err() {
                return FramesEnd::StreamClosed;
            }
        }
        if self.output.calls_sandy() {
            FramesEnd::CallsSandy
        } else {
            FramesEnd::AnswerEnded
        }
    }

    /// One Sandy call: the chain's Sandy writes at most `sandy_call_tokens` of
    /// the answer and, with taps, sends the taps of the trace and its text to a
    /// new room at its engine. A call that fails before any of it reached the
    /// client tries another Sandy, up to `--retry-max-attempts`.
    async fn call_sandy(
        &mut self,
        ctx: &AppContext,
        headers: &HeaderMap,
        start: Instant,
        duration: &mut Duration,
    ) -> Result<Answer, ApiError> {
        let written = self.trace.num_written() as u64;
        let call_tokens = u64::from(self.config.sandy_call_tokens);
        let limits = self.limits.of_request(written, Some(call_tokens));
        let limit_is_the_answers = self
            .limits
            .left(written)
            .is_some_and(|left| left <= call_tokens);
        let sandy_ids = self.trace.sandy_ids();
        let own_tokenizer = sandy_ids.is_some();
        let input_ids = match sandy_ids {
            Some(ids) => json!(ids),
            None => Value::Array(self.trace.ricky_ids()),
        };
        let mut body = sandy_call_body(&self.body_without_prompt, input_ids, limits, own_tokenizer);
        let rid = self.next_rid("s");
        body.insert("rid".into(), rid.clone().into());
        if let Some(chain) = self.chain_field(false) {
            body.insert("franky_chain".into(), chain);
        }
        let mut request =
            PreparedRequest::generate_value(ctx, self.model.clone(), Value::Object(body), None)?;
        let duration = duration
            .get_or_insert_with(|| RequestDurationGuard::new(ctx, &self.model, start))
            .clone();
        let mut excluded: Vec<WorkerId> = Vec::new();
        let mut failed = None;
        for attempt in 0..ctx.config.proxy.max_attempts.get() {
            if attempt > 0 && !back_off(ctx, start, attempt).await {
                break;
            }
            let sandy = match self.sandy.clone() {
                Some(sandy) if !excluded.contains(&sandy.id) => sandy,
                _ => {
                    let prefix = PrefixMemo::default();
                    match pick_sandy(ctx, &request, headers, &excluded, &prefix).await {
                        Ok(sandy) => sandy,
                        Err(error) => return failed.map(Answer::Failed).ok_or(error),
                    }
                }
            };
            if failed.is_some() {
                ctx.metrics.record_retry(&request.model.0);
            }
            let room = self.config.taps.then(|| room_at(&sandy));
            request.tap_source = room.clone().map_or(TapSource::None, TapSource::Room);
            let workers = SelectedWorkers {
                prefill: Arc::clone(&sandy),
                decode: None,
                sandy: None,
                track_dispatch_timestamps: false,
            };
            let dispatched = forward_request(
                ctx,
                &mut request,
                workers.clone(),
                headers.clone(),
                start,
                &duration,
            )
            .await?;
            if !dispatched.retry_excluding.is_empty() {
                self.sandy = None;
                excluded.extend(dispatched.retry_excluding);
                failed = Some(dispatched.response);
                continue;
            }
            if dispatched.response.status() != StatusCode::OK {
                return Ok(Answer::Failed(dispatched.response));
            }
            if let Some(end) = &mut self.end {
                end.remember_engines(&workers);
            }
            let room = room.map(|room| {
                Box::new(SandyRoom {
                    room,
                    unread: ctx.proxy.abort_on_drop(
                        &sandy.url,
                        sandy.protocol(),
                        headers,
                        Some(&rid),
                    ),
                })
            });
            self.sandy = Some(sandy);
            let (head, body) = dispatched.response.into_parts();
            let text = self
                .read_sandy_call(ctx, body, limit_is_the_answers)
                .await?;
            return Ok(Answer::SandyText { head, text, room });
        }
        Ok(Answer::Failed(failed.expect("at least one attempt")))
    }

    async fn read_sandy_call(
        &self,
        ctx: &AppContext,
        body: Body,
        limit_is_the_answers: bool,
    ) -> Result<SandyText, ApiError> {
        let bytes = axum::body::to_bytes(body, usize::MAX)
            .await
            .map_err(|error| ApiError::Internal(error.into()))?;
        let frame: Value =
            serde_json::from_slice(&bytes).map_err(|error| ApiError::Internal(error.into()))?;
        let mut text = read_sandy_text(&frame, limit_is_the_answers, &self.client_stop_ids);
        if self.trace.sandy_ids().is_some() {
            let ricky_ids = ctx
                .tokenizers
                .encode_answer_text(&self.model.0, &text.text)
                .ok_or_else(|| {
                    ApiError::Internal(anyhow::anyhow!(
                        "Ricky's tokenizer of model {} is not loaded",
                        self.model.0
                    ))
                })?;
            text.ricky_ids = ricky_ids.into_iter().map(Value::from).collect();
        }
        Ok(text)
    }

    /// Sends the chain's next Ricky leg. With taps, it reads the room of the
    /// Sandy call before it, or with Sandy writing nothing, those a Sandy leg
    /// beside it sends of the trace.
    async fn send_ricky_leg(
        &mut self,
        ctx: &AppContext,
        routing: &ModelRouting<'_>,
        headers: &HeaderMap,
        start: Instant,
        duration: &mut Duration,
        mut room: Option<Box<SandyRoom>>,
    ) -> Result<Answer, ApiError> {
        let tap_source = match (self.config.taps, &room) {
            (false, _) => TapSource::None,
            (true, Some(room)) => TapSource::Room(room.room.clone()),
            (true, None) => {
                self.trace.add_sandy_text(Vec::new(), Vec::new());
                TapSource::SandyLeg
            }
        };
        let franky = RickyLegFields {
            chain: self.chain_field(true),
            tap_runs: self.config.taps.then(|| wire(&self.trace.tap_runs())),
            sandy_input_ids: self
                .trace
                .sandy_ids()
                .filter(|_| self.config.taps)
                .map(|ids| json!(ids)),
        };
        let written = self.trace.num_written() as u64;
        let mut body = ricky_leg_body(
            &self.body_without_prompt,
            Value::Array(self.trace.ricky_ids()),
            self.limits.of_request(written, None),
            self.ricky.is_none(),
            franky,
        );
        body.insert("rid".into(), self.next_rid("r").into());
        let mut request =
            PreparedRequest::generate_value(ctx, self.model.clone(), Value::Object(body), None)?;
        if let (TapSource::SandyLeg, Some(sandy_ids)) = (&tap_source, self.trace.sandy_ids()) {
            request.set_franky_sandy_ids(sandy_ids);
        }
        request.tap_source = tap_source;
        self.output.start_leg();
        let (response, answered) = routing
            .forward_with_retries(
                ctx,
                &mut request,
                headers,
                start,
                duration,
                self.ricky.clone(),
            )
            .await?;
        if let Some(workers) = answered {
            if let Some(end) = &mut self.end {
                end.remember_engines(&workers);
            }
            self.ricky = Some(workers);
        }
        if response.status() != StatusCode::OK {
            return Ok(Answer::Failed(response));
        }
        if let Some(room) = &mut room {
            room.unread.disarm();
        }
        let (head, body) = response.into_parts();
        Ok(Answer::Leg { head, body })
    }

    /// Sandy's call wrote `text`: it joins the trace and the answer.
    ///
    /// Returns: the chain's streamed frame of the text.
    fn add_sandy_text(&mut self, text: &SandyText) -> Value {
        tracing::debug!(
            chain = %self.id,
            block = self.block,
            tokens = text.ricky_ids.len(),
            finish_reason = %text.finish_reason,
            "franky sandy call ended"
        );
        self.trace
            .add_sandy_text(text.ricky_ids.clone(), text.sandy_ids.clone());
        let finish_reason = if text.ends_answer {
            &text.finish_reason
        } else {
            &Value::Null
        };
        self.output
            .add_sandy_text(&text.text, text.ricky_ids.clone(), finish_reason)
    }

    /// The current leg ended for a Sandy call: its text joins the trace and
    /// the chain moves to the block Ricky writes after the call.
    fn finish_leg(&mut self) -> Result<(), ApiError> {
        self.log_leg();
        let (ids, text) = self.output.current_leg();
        let (ids, text) = (ids.to_vec(), text.to_owned());
        self.trace
            .add_ricky_text(&ids, &text)
            .map_err(|error| ApiError::Internal(error.context("tokenize a Ricky leg for Sandy")))?;
        self.output.finish_leg();
        if let Some(next_block) = self.output.next_block() {
            self.block = next_block;
        }
        Ok(())
    }

    /// `franky_chain` of the chain's next request, whose engine holds the
    /// chain's cache for the request after; a Ricky leg's also names Ricky's
    /// blocks. `None` for a chain without blocks: no Sandy call follows a
    /// Ricky leg, so nothing comes back for a held cache.
    fn chain_field(&self, ricky_leg: bool) -> Option<Value> {
        let blocks = self.config.blocks.as_ref()?;
        let mut chain = json!({"id": self.id, "block": self.block});
        if ricky_leg {
            chain["blocks"] = Value::Object(blocks.clone());
        }
        Some(chain)
    }

    fn next_rid(&mut self, kind: &str) -> String {
        self.requests_sent += 1;
        format!("{}-{kind}{}", self.id, self.requests_sent)
    }

    fn log_leg(&self) {
        let report = self.output.leg_report();
        tracing::debug!(
            chain = %self.id,
            block = self.block,
            prompt_tokens = %report.get("prompt_tokens").unwrap_or(&serde_json::Value::Null),
            cached_tokens = %report.get("cached_tokens").unwrap_or(&serde_json::Value::Null),
            report = %serde_json::Value::Object(report.clone()),
            "franky leg ended"
        );
    }
}

/// A new room at `sandy`'s tap bootstrap server.
fn room_at(sandy: &Worker) -> FrankyRoom {
    FrankyRoom {
        host: sandy.bootstrap_host().to_string(),
        port: sandy.bootstrap_port(),
        room: generate_room_id(),
    }
}

fn check_chains_can_serve(body: &Map<String, Value>, sandy_writes: bool) -> Result<(), ApiError> {
    let sampling = body.get("sampling_params");
    let constrained = CONSTRAINED_DECODING_FIELDS.iter().any(|field| {
        sampling
            .and_then(|params| params.get(field))
            .is_some_and(is_set)
    });
    if constrained {
        return Err(refused(
            "constrained decoding, whose state each request of a chain would restart",
        ));
    }
    if let Some(field) = FIELDS_CHAINS_CANNOT_JOIN
        .iter()
        .find(|field| body.get(**field).is_some_and(is_set))
    {
        return Err(refused(&format!("`{field}`")));
    }
    match PER_TOKEN_OUTPUT_FIELDS
        .iter()
        .find(|field| sandy_writes && body.get(**field).is_some_and(is_set))
    {
        Some(field) => Err(refused(&format!("`{field}` while Sandy writes"))),
        None => Ok(()),
    }
}

fn is_set(value: &Value) -> bool {
    !matches!(value, Value::Null | Value::Bool(false))
}

fn refused(what: &str) -> ApiError {
    ApiError::BadRequest(format!(
        "Franky chains (--franky-chain) cannot serve {what}"
    ))
}

fn empty_head() -> Parts {
    Response::new(()).into_parts().0
}

fn unary_response(mut head: Parts, frame: &Value) -> Response<Body> {
    head.headers.remove(header::CONTENT_LENGTH);
    Response::from_parts(head, Body::from(frame.to_string()))
}

async fn fail_stream(events: &Events, message: &str) {
    let _ = events.send(Ok(error_event(message))).await;
    let _ = events.send(Ok(Bytes::from_static(DONE_EVENT))).await;
}

fn json_event(frame: &Value) -> Bytes {
    data_event(frame.to_string().as_bytes())
}

fn data_event(data: &[u8]) -> Bytes {
    [b"data: ".as_slice(), data, b"\n\n"].concat().into()
}

fn error_event(message: &str) -> Bytes {
    json_event(&json!({"error": {"message": message}}))
}

async fn error_event_from_response(response: Response<Body>) -> Bytes {
    let status = response.status();
    let body = match axum::body::to_bytes(response.into_body(), usize::MAX).await {
        Ok(body) => body,
        Err(error) => {
            return error_event(&format!(
            "a Franky chain's request failed with {status}, and reading its body failed: {error}"
        ))
        }
    };
    match serde_json::from_slice::<Value>(&body) {
        Ok(error) if error.get("error").is_some() => data_event(&body),
        _ => error_event(&format!(
            "a Franky chain's request failed with {status}: {}",
            String::from_utf8_lossy(&body)
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn object(value: Value) -> Map<String, Value> {
        match value {
            Value::Object(fields) => fields,
            _ => panic!("not an object"),
        }
    }

    #[test]
    fn chains_refuse_what_their_requests_cannot_carry() {
        for (body, sandy_writes) in [
            (json!({"sampling_params": {"regex": "a+"}}), false),
            (json!({"return_hidden_states": true}), false),
            (json!({"return_logprob": true}), true),
        ] {
            assert!(
                check_chains_can_serve(&object(body.clone()), sandy_writes).is_err(),
                "{body}"
            );
        }
        let logprobs = object(json!({"return_logprob": true, "stream": true}));
        assert!(check_chains_can_serve(&logprobs, false).is_ok());
    }
}
