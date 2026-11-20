// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

mod answer_limits;
mod checks;
mod end;
mod events;
mod frames;
mod output;
mod ricky_leg;
mod sandy_call;
mod trace;

use super::forward::{RequestDurationGuard, SelectedWorkers};
use super::franky_request::{FrankyRequest, FrankyRoom, TapSource};
use super::preparation::{
    model_openai_settings, sandy_incremental_streaming, PreparedRequest, GENERATE_PATH,
};
use super::{forward_with_retries, reads_taps, serves_franky, ModelRouting, SandyPicker};
use crate::config::FrankyChainConfig;
use crate::discovery::ModelId;
use crate::proxy::AbortOnDrop;
use crate::server::app_context::AppContext;
use crate::server::error::ApiError;
use answer_limits::AnswerLimits;
use axum::body::Body;
use axum::http::response::Parts;
use axum::http::{header, HeaderMap, HeaderValue, Response, StatusCode};
use bytes::Bytes;
use checks::{check_chains_can_serve, refused};
use end::ChainEnd;
use events::{data_event, done_event, error_event_from_response, fail_stream, json_event, Events};
use frames::{engine_frames, EngineFrame, EngineFrames};
use futures::StreamExt;
use output::ChainOutput;
use ricky_leg::{ricky_leg_body, Leg, LegEnd};
use sandy_call::{
    sandy_call_body, SandyDelta, SandyLimit, SandyRead, SandyStream, SandyText, SandyTokenizer,
};
use serde_json::{json, Map, Value};
use sglang_processor::openai::Responder;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;
use trace::ChainTrace;

/// The client's fields the chain sets anew on each of its requests.
const CHAIN_FIELDS: [&str; 6] = [
    "input_ids",
    "text",
    "rid",
    "franky_tap_runs",
    "franky_sandy_input_ids",
    "franky_chain",
];
const EVENT_BUFFER: usize = 64;

type DurationGuard = Option<Arc<RequestDurationGuard>>;

/// The room a Sandy call sent its taps to. Sandy holds the call until a Ricky
/// leg reads them, so unless one answers for it, dropping it aborts the call.
struct SandyRoom {
    room: FrankyRoom,
    unread: AbortOnDrop,
}

/// A Sandy call's response, read frame by frame.
struct SandyCall {
    head: Parts,
    frames: EngineFrames,
    /// With taps, the room the next Ricky leg reads.
    room: Option<Box<SandyRoom>>,
    text: SandyStream,
}

/// A Ricky leg's response, read frame by frame.
struct RickyLeg {
    head: Parts,
    frames: EngineFrames,
    /// How the leg ends, once a frame says it.
    end: Option<LegEnd>,
}

/// Where a chain is between two of its steps.
enum Next {
    SandyCall(SandyCall),
    /// The Ricky leg after a Sandy call is to be sent; it reads the call's
    /// room.
    SendRickyLeg(Option<Box<SandyRoom>>),
    RickyLeg(RickyLeg),
    /// A request failed; the client gets its response.
    Failed(Response<Body>),
    /// The answer ended with the last frame, in the response of `head`.
    Ended(Parts),
}

/// What one step of a chain brings.
enum Step {
    /// The client gets this frame, and the chain goes on from `Next`.
    Frame(Value, Next),
    /// The chain goes on from `Next`.
    Goes(Next),
    /// The chain ends with this event.
    Ends(ChainEvent),
}

/// What a chain gives its client next.
enum ChainEvent {
    /// A frame of the answer: the answer so far, or with an incremental
    /// stream, what it adds.
    Frame(Value),
    /// The answer ended with the last frame, in the response of `head`.
    Ended(Parts),
    /// A request failed; the client gets its response.
    Failed(Response<Body>),
    /// A streamed request sent this error frame, which ends the answer.
    EngineError(Bytes),
    /// A request's response broke off.
    Broken(axum::Error),
}

/// A Franky request served as a chain (`--franky-chain`): Sandy calls that
/// write part of the answer, and Ricky legs that write blocks until Ricky's
/// engine ends a leg for the next Sandy call. Each request carries the whole
/// trace so far and goes to the chain's Sandy or Ricky, whose caches keep it.
pub(super) struct FrankyChain {
    id: String,
    model: ModelId,
    config: FrankyChainConfig,
    /// Whether the chain's requests read taps.
    taps: bool,
    /// Whether the chain's Sandy streams only what each frame adds; `None`
    /// when its workers do not all say the same, so its calls go unary.
    sandy_increments: Option<bool>,
    limits: AnswerLimits,
    body_without_prompt: Map<String, Value>,
    client_stop_ids: Vec<u64>,
    trace: ChainTrace,
    streaming: bool,
    responder: Option<Responder>,
    output: ChainOutput,
    ricky: Option<SelectedWorkers>,
    sandy: Option<SelectedWorkers>,
    end: Option<ChainEnd>,
    next: Option<Next>,
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
        check_chains_can_serve(&body, config)?;
        let Some(prompt_ids) = body.remove("input_ids").as_ref().and_then(token_ids) else {
            return Err(refused(
                "a prompt without token ids; send input_ids or text the router tokenizes",
            ));
        };
        let client_rid = body.remove("rid");
        for field in CHAIN_FIELDS {
            body.remove(field);
        }
        let sampling = body.get("sampling_params").unwrap_or(&Value::Null);
        let sandy = match ctx.franky_sandy.clone() {
            None => None,
            Some(tokens) => {
                let Some(prompt_ids) = request.franky.take_sandy_ids() else {
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
        let incremental = request.streaming
            && model_openai_settings(ctx, &request.model)
                .is_some_and(|settings| settings.incremental_streaming_output);
        let earlier_runs = request.franky.earlier_runs().to_vec();
        Ok(Some(Self {
            output: ChainOutput::new(reported_id, incremental, prompt_ids.len() as u64),
            end: config
                .blocks
                .is_some()
                .then(|| ChainEnd::new(ctx.proxy.admin_client().clone(), id.clone(), headers)),
            trace: ChainTrace::of_prompt(prompt_ids, sandy, earlier_runs),
            id,
            model: request.model.clone(),
            config: config.clone(),
            taps: reads_taps(ctx, &request.model),
            sandy_increments: sandy_incremental_streaming(ctx, &request.model),
            limits,
            body_without_prompt: body,
            client_stop_ids,
            streaming: request.streaming,
            responder: request.responder.take(),
            ricky: None,
            sandy: None,
            next: None,
            block: 1,
            requests_sent: 0,
        }))
    }

    /// Serve the chain, Sandy call and Ricky leg after leg, as one response.
    pub(super) async fn serve(
        mut self,
        ctx: &Arc<AppContext>,
        headers: HeaderMap,
        start: Instant,
    ) -> Result<Response<Body>, ApiError> {
        let mut responder = self.responder.take();
        let streaming = self.streaming;
        let mut duration = None;
        let first = self
            .next_request(ctx, &headers, start, &mut duration)
            .await?;
        let response = match first {
            Next::Failed(response) => response,
            first if streaming => self.stream(Arc::clone(ctx), first, headers, start, duration),
            first => {
                self.next = Some(first);
                self.collect(ctx, &headers, start, &mut duration).await?
            }
        };
        Ok(match responder {
            Some(_) => super::openai::respond(&mut responder, response, streaming).await,
            None => response,
        })
    }

    /// The unary response: the last frame, which holds the whole answer.
    async fn collect(
        &mut self,
        ctx: &AppContext,
        headers: &HeaderMap,
        start: Instant,
        duration: &mut DurationGuard,
    ) -> Result<Response<Body>, ApiError> {
        let mut last = Value::Null;
        loop {
            match self.next_event(ctx, headers, start, duration).await? {
                ChainEvent::Frame(frame) => last = frame,
                ChainEvent::Ended(mut head) => {
                    head.headers.remove(header::CONTENT_LENGTH);
                    return Ok(Response::from_parts(head, Body::from(last.to_string())));
                }
                ChainEvent::Failed(response) => return Ok(response),
                ChainEvent::EngineError(data) => {
                    return Err(ApiError::Internal(anyhow::anyhow!(
                        "a unary Franky request sent an error frame: {}",
                        String::from_utf8_lossy(&data)
                    )))
                }
                ChainEvent::Broken(error) => return Err(ApiError::Internal(error.into())),
            }
        }
    }

    /// The streamed response, whose head is that of the chain's `first`
    /// request.
    fn stream(
        mut self,
        ctx: Arc<AppContext>,
        mut first: Next,
        headers: HeaderMap,
        start: Instant,
        duration: DurationGuard,
    ) -> Response<Body> {
        let mut head = first.take_head();
        head.headers.remove(header::CONTENT_LENGTH);
        head.headers.insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("text/event-stream"),
        );
        self.next = Some(first);
        let (events, receiver) = mpsc::channel(EVENT_BUFFER);
        tokio::spawn(self.pump(ctx, events, headers, start, duration));
        let stream = futures::stream::unfold(receiver, |mut receiver| async move {
            let event = receiver.recv().await?;
            Some((event, receiver))
        });
        Response::from_parts(head, Body::from_stream(stream))
    }

    async fn pump(
        mut self,
        ctx: Arc<AppContext>,
        events: Events,
        headers: HeaderMap,
        start: Instant,
        mut duration: DurationGuard,
    ) {
        loop {
            let event = self.next_event(&ctx, &headers, start, &mut duration).await;
            let sent = match event {
                Ok(ChainEvent::Frame(frame)) => events.send(Ok(json_event(&frame))).await,
                Ok(ChainEvent::Ended(_)) => {
                    let _ = events.send(Ok(done_event())).await;
                    return;
                }
                Ok(ChainEvent::Failed(response)) => {
                    let event = error_event_from_response(response).await;
                    let _ = events.send(Ok(event)).await;
                    let _ = events.send(Ok(done_event())).await;
                    return;
                }
                Ok(ChainEvent::EngineError(data)) => {
                    let _ = events.send(Ok(data_event(&data))).await;
                    let _ = events.send(Ok(done_event())).await;
                    return;
                }
                Ok(ChainEvent::Broken(error)) => {
                    let _ = events.send(Err(error)).await;
                    return;
                }
                Err(error) => return fail_stream(&events, &error.to_string()).await,
            };
            if sent.is_err() {
                return;
            }
        }
    }

    /// Runs the chain up to what its client gets next.
    async fn next_event(
        &mut self,
        ctx: &AppContext,
        headers: &HeaderMap,
        start: Instant,
        duration: &mut DurationGuard,
    ) -> Result<ChainEvent, ApiError> {
        loop {
            let next = self.next.take().expect("a chain runs until it ends");
            match self.step(next, ctx, headers, start, duration).await? {
                Step::Frame(frame, next) => {
                    self.next = Some(next);
                    return Ok(ChainEvent::Frame(frame));
                }
                Step::Goes(next) => self.next = Some(next),
                Step::Ends(event) => return Ok(event),
            }
        }
    }

    async fn step(
        &mut self,
        next: Next,
        ctx: &AppContext,
        headers: &HeaderMap,
        start: Instant,
        duration: &mut DurationGuard,
    ) -> Result<Step, ApiError> {
        Ok(match next {
            Next::SandyCall(mut call) => match call.frames.next().await {
                Some(Ok(EngineFrame::Frame(frame))) => self.read_sandy_frame(ctx, &frame, call)?,
                Some(Ok(EngineFrame::Error(data))) => Step::Ends(ChainEvent::EngineError(data)),
                Some(Err(error)) => Step::Ends(ChainEvent::Broken(error)),
                None => Step::Ends(ChainEvent::Broken(axum::Error::new(anyhow::anyhow!(
                    "a Sandy call's stream ended unfinished"
                )))),
            },
            Next::SendRickyLeg(room) => Step::Goes(
                self.send_ricky_leg(ctx, headers, start, duration, room)
                    .await?,
            ),
            Next::RickyLeg(mut leg) => match leg.frames.next().await {
                Some(Ok(EngineFrame::Frame(frame))) => self.read_leg_frame(frame, leg),
                Some(Ok(EngineFrame::Error(data))) => Step::Ends(ChainEvent::EngineError(data)),
                Some(Err(error)) => Step::Ends(ChainEvent::Broken(error)),
                None => Step::Goes(self.after_leg(ctx, headers, start, duration, leg).await?),
            },
            Next::Failed(response) => Step::Ends(ChainEvent::Failed(response)),
            Next::Ended(head) => Step::Ends(ChainEvent::Ended(head)),
        })
    }

    fn read_sandy_frame(
        &mut self,
        ctx: &AppContext,
        frame: &Value,
        mut call: SandyCall,
    ) -> Result<Step, ApiError> {
        let model = &self.model.0;
        let read = (call.text)
            .read(frame, &self.client_stop_ids, |text| {
                ctx.tokenizers.encode_answer_text(model, text)
            })
            .map_err(|error| ApiError::Internal(error.context("tokenize Sandy's text")))?;
        Ok(match read {
            SandyRead::Adds(adds) => {
                let frame = self
                    .output
                    .add_sandy_text(&adds.text, &adds.ids, &Value::Null);
                Step::Frame(frame, Next::SandyCall(call))
            }
            SandyRead::Ends { text, adds } => {
                let frame = self.end_sandy_call(&text, &adds);
                let next = if text.ends_answer {
                    Next::Ended(call.head)
                } else {
                    Next::SendRickyLeg(call.room)
                };
                Step::Frame(frame, next)
            }
        })
    }

    fn read_leg_frame(&mut self, frame: Value, mut leg: RickyLeg) -> Step {
        leg.end = LegEnd::of_frame(&frame).or(leg.end);
        let frame = self.output.leg_frame(frame, leg.end.as_ref());
        Step::Frame(frame, Next::RickyLeg(leg))
    }

    async fn after_leg(
        &mut self,
        ctx: &AppContext,
        headers: &HeaderMap,
        start: Instant,
        duration: &mut DurationGuard,
        leg: RickyLeg,
    ) -> Result<Next, ApiError> {
        self.log_leg();
        match leg.end {
            Some(LegEnd::CallsSandy { next_block }) => {
                self.finish_leg(next_block)?;
                self.next_request(ctx, headers, start, duration).await
            }
            Some(LegEnd::AnswerEnds) | None => Ok(Next::Ended(leg.head)),
        }
    }

    /// The chain's next request: a Sandy call when Sandy writes, else a Ricky
    /// leg.
    async fn next_request(
        &mut self,
        ctx: &AppContext,
        headers: &HeaderMap,
        start: Instant,
        duration: &mut DurationGuard,
    ) -> Result<Next, ApiError> {
        if self.config.sandy_call_tokens > 0 {
            self.call_sandy(ctx, headers, start, duration).await
        } else {
            self.send_ricky_leg(ctx, headers, start, duration, None)
                .await
        }
    }

    /// One Sandy call: the chain's Sandy writes at most `sandy_call_tokens` of
    /// the answer and, with taps, sends the taps of the trace and its text to a
    /// room at its engine. It streams to a streaming client when Ricky reads
    /// Sandy's ids as they are and Sandy says how it streams.
    async fn call_sandy(
        &mut self,
        ctx: &AppContext,
        headers: &HeaderMap,
        start: Instant,
        duration: &mut DurationGuard,
    ) -> Result<Next, ApiError> {
        let written = self.trace.num_written();
        let call_tokens = u64::from(self.config.sandy_call_tokens);
        let limit = match self.limits.left(written) {
            Some(left) if left <= call_tokens => SandyLimit::Answer,
            _ => SandyLimit::Call,
        };
        let (input_ids, tokenizer) = match self.trace.sandy_ids() {
            Some(ids) => (json!(ids), SandyTokenizer::Own),
            None => (json!(self.trace.ricky_ids()), SandyTokenizer::Ricky),
        };
        let streamed_increments = (self.sandy_increments)
            .filter(|_| self.streaming && tokenizer == SandyTokenizer::Ricky);
        let limits = self.limits.of_request(written, Some(call_tokens));
        let mut body = sandy_call_body(
            &self.body_without_prompt,
            input_ids,
            limits,
            tokenizer,
            streamed_increments.is_some(),
        );
        let rid = self.next_rid("s");
        body.insert("rid".into(), rid.clone().into());
        if let Some(chain) = self.sandy_call_chain() {
            body.insert("franky_chain".into(), chain);
        }
        let mut request =
            PreparedRequest::generate_value(ctx, self.model.clone(), Value::Object(body), None)?;
        request.franky = FrankyRequest::sandy_call(self.taps);
        let (response, answered) = forward_with_retries(
            ctx,
            &SandyPicker,
            &mut request,
            headers,
            start,
            duration,
            self.sandy.clone(),
        )
        .await?;
        let Some(workers) = answered.filter(|_| response.status() == StatusCode::OK) else {
            return Ok(Next::Failed(response));
        };
        if let Some(end) = &mut self.end {
            end.remember_engines(&workers);
        }
        let sandy = workers.answering();
        let room = request.franky.take_room().map(|room| {
            Box::new(SandyRoom {
                room,
                unread: ctx
                    .proxy
                    .abort_on_drop(&sandy.url, sandy.protocol(), headers, Some(&rid)),
            })
        });
        self.sandy = Some(workers);
        let (head, body) = response.into_parts();
        Ok(Next::SandyCall(SandyCall {
            head,
            frames: engine_frames(body, streamed_increments.is_some()),
            room,
            text: SandyStream::new(tokenizer, limit, streamed_increments.unwrap_or(false)),
        }))
    }

    /// Sends the chain's next Ricky leg. With taps, it reads the room of the
    /// Sandy call before it, or with Sandy writing nothing, those a Sandy leg
    /// beside it sends of the trace.
    async fn send_ricky_leg(
        &mut self,
        ctx: &AppContext,
        headers: &HeaderMap,
        start: Instant,
        duration: &mut DurationGuard,
        mut room: Option<Box<SandyRoom>>,
    ) -> Result<Next, ApiError> {
        let tap_source = match (self.taps, &room) {
            (false, _) => TapSource::None,
            (true, Some(room)) => TapSource::Room(room.room.clone()),
            (true, None) => {
                self.trace.start_tap_leg();
                TapSource::SandyLeg
            }
        };
        let leg = if self.ricky.is_none() {
            Leg::First
        } else {
            Leg::Later
        };
        let limits = self.limits.of_request(self.trace.num_written(), None);
        let input_ids = json!(self.trace.ricky_ids());
        let chain = self.ricky_leg_chain();
        let mut body = ricky_leg_body(&self.body_without_prompt, input_ids, limits, leg, chain);
        body.insert("rid".into(), self.next_rid("r").into());
        let mut request =
            PreparedRequest::generate_value(ctx, self.model.clone(), Value::Object(body), None)?;
        request.franky =
            FrankyRequest::ricky_leg(tap_source, self.trace.tap_runs(), self.trace.sandy_ids());
        let routing = ModelRouting::lookup(ctx, &self.model)?;
        let (response, answered) = forward_with_retries(
            ctx,
            &routing,
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
            return Ok(Next::Failed(response));
        }
        if let Some(room) = &mut room {
            room.unread.disarm();
        }
        let (head, body) = response.into_parts();
        Ok(Next::RickyLeg(RickyLeg {
            head,
            frames: engine_frames(body, self.streaming),
            end: None,
        }))
    }

    /// Sandy's call ended with `text`, of which the answer gets `adds` now:
    /// all of it joins the trace.
    ///
    /// Returns: the chain's frame of `adds`, which ends the answer when the
    /// call did.
    fn end_sandy_call(&mut self, text: &SandyText, adds: &SandyDelta) -> Value {
        tracing::debug!(
            chain = %self.id,
            block = self.block,
            tokens = text.ricky_ids.len(),
            finish_reason = %text.finish_reason,
            "franky sandy call ended"
        );
        self.trace.add_sandy_text(&text.ricky_ids, &text.sandy_ids);
        let finish_reason = if text.ends_answer {
            &text.finish_reason
        } else {
            &Value::Null
        };
        self.output
            .add_sandy_text(&adds.text, &adds.ids, finish_reason)
    }

    /// The current leg ended for a Sandy call: its text joins the trace, and
    /// Ricky writes block `next_block` after the call.
    fn finish_leg(&mut self, next_block: u64) -> Result<(), ApiError> {
        let (ids, text) = self.output.finish_leg();
        self.trace
            .add_ricky_text(&ids, &text)
            .map_err(|error| ApiError::Internal(error.context("tokenize a Ricky leg for Sandy")))?;
        self.block = next_block;
        Ok(())
    }

    /// `franky_chain` of a Sandy call, whose engine holds the chain's cache
    /// for its next call. `None` for a chain without blocks, which makes no
    /// Sandy call after a Ricky leg.
    fn sandy_call_chain(&self) -> Option<Value> {
        self.config.blocks.as_ref()?;
        Some(json!({"id": self.id, "block": self.block}))
    }

    /// `franky_chain` of a Ricky leg, which also names Ricky's blocks; `None`
    /// for a chain without blocks.
    fn ricky_leg_chain(&self) -> Option<Value> {
        let blocks = self.config.blocks.as_ref()?;
        Some(json!({"id": self.id, "block": self.block, "blocks": blocks}))
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

impl Next {
    /// The head of the response a streamed chain answers with: its first
    /// request's.
    fn take_head(&mut self) -> Parts {
        let empty = Response::new(()).into_parts().0;
        match self {
            Self::SandyCall(SandyCall { head, .. }) | Self::RickyLeg(RickyLeg { head, .. }) => {
                std::mem::replace(head, empty)
            }
            Self::SendRickyLeg(_) | Self::Failed(_) | Self::Ended(_) => empty,
        }
    }
}

fn token_ids(ids: &Value) -> Option<Vec<u32>> {
    (ids.as_array()?.iter())
        .map(|id| id.as_u64().and_then(|id| u32::try_from(id).ok()))
        .collect()
}
