// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

mod answer_limits;
mod checks;
mod end;
mod events;
mod output;
mod ricky_leg;
mod sandy_call;
mod trace;

use super::forward::{RequestDurationGuard, SelectedWorkers};
use super::preparation::{
    model_openai_settings, FrankyRoom, PreparedRequest, TapSource, GENERATE_PATH,
};
use super::tap_runs::wire;
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
use futures::stream::BoxStream;
use futures::StreamExt;
use output::ChainOutput;
use ricky_leg::{leg_frames, ricky_leg_body, Leg, LegEnd, LegFrame, RickyLegFields};
use sandy_call::{read_sandy_text, sandy_call_body, SandyLimit, SandyText, SandyTokenizer};
use serde_json::{json, Map, Value};
use sglang_processor::openai::Responder;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;
use trace::ChainTrace;

/// The fields the chain sets on each of its requests.
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

/// Where a chain is between two of its events.
enum Next {
    /// The answer of the chain's last request.
    Answer(Answer),
    /// The Ricky leg after a Sandy call, which reads its room.
    RickyLeg(Option<Box<SandyRoom>>),
    /// A Ricky leg's frames, and how the leg ends once a frame says it.
    Frames {
        head: Parts,
        frames: BoxStream<'static, Result<LegFrame, axum::Error>>,
        end: Option<LegEnd>,
    },
    /// The answer ended with the last frame, in the response of `head`.
    Ended(Parts),
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
    /// A streamed leg sent this error frame, which ends the answer.
    LegError(Bytes),
    /// A leg's response broke off.
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
        let incremental = request.streaming
            && model_openai_settings(ctx, &request.model)
                .is_some_and(|settings| settings.incremental_streaming_output);
        Ok(Some(Self {
            output: ChainOutput::new(reported_id, incremental, prompt_ids.len() as u64),
            end: config
                .blocks
                .is_some()
                .then(|| ChainEnd::new(ctx.proxy.admin_client().clone(), id.clone(), headers)),
            trace: ChainTrace::of_prompt(prompt_ids, sandy, request.franky_earlier_runs().to_vec()),
            id,
            model: request.model.clone(),
            config: config.clone(),
            taps: reads_taps(ctx, &request.model),
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
            .next_answer(ctx, &headers, start, &mut duration)
            .await?;
        let response = match first {
            Answer::Failed(response) => response,
            first if streaming => self.stream(Arc::clone(ctx), first, headers, start, duration),
            first => {
                self.next = Some(Next::Answer(first));
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
                ChainEvent::LegError(data) => {
                    return Err(ApiError::Internal(anyhow::anyhow!(
                        "a unary Franky leg sent an error frame: {}",
                        String::from_utf8_lossy(&data)
                    )))
                }
                ChainEvent::Broken(error) => return Err(ApiError::Internal(error.into())),
            }
        }
    }

    /// The streamed response, whose head is that of the chain's `first` answer.
    fn stream(
        mut self,
        ctx: Arc<AppContext>,
        mut first: Answer,
        headers: HeaderMap,
        start: Instant,
        duration: DurationGuard,
    ) -> Response<Body> {
        let mut head = match &mut first {
            Answer::SandyText { head, .. } | Answer::Leg { head, .. } => {
                std::mem::replace(head, Response::new(()).into_parts().0)
            }
            Answer::Failed(_) => Response::new(()).into_parts().0,
        };
        head.headers.remove(header::CONTENT_LENGTH);
        head.headers.insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("text/event-stream"),
        );
        self.next = Some(Next::Answer(first));
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
                Ok(ChainEvent::LegError(data)) => {
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
            self.next = match next {
                Next::Answer(Answer::Failed(response)) => return Ok(ChainEvent::Failed(response)),
                Next::Answer(Answer::SandyText { head, text, room }) => {
                    let frame = self.add_sandy_text(&text);
                    self.next = Some(if text.ends_answer {
                        Next::Ended(head)
                    } else {
                        Next::RickyLeg(room)
                    });
                    return Ok(ChainEvent::Frame(frame));
                }
                Next::Answer(Answer::Leg { head, body }) => Some(Next::Frames {
                    head,
                    frames: leg_frames(body, self.streaming),
                    end: None,
                }),
                Next::RickyLeg(room) => {
                    let answer = self
                        .send_ricky_leg(ctx, headers, start, duration, room)
                        .await?;
                    Some(Next::Answer(answer))
                }
                Next::Frames {
                    head,
                    mut frames,
                    mut end,
                } => match frames.next().await {
                    Some(Ok(LegFrame::Frame(frame))) => {
                        end = LegEnd::of_frame(&frame).or(end);
                        let frame = self.output.leg_frame(frame, end.as_ref());
                        self.next = Some(Next::Frames { head, frames, end });
                        return Ok(ChainEvent::Frame(frame));
                    }
                    Some(Ok(LegFrame::Error(data))) => return Ok(ChainEvent::LegError(data)),
                    Some(Err(error)) => return Ok(ChainEvent::Broken(error)),
                    None => {
                        self.log_leg();
                        match end {
                            Some(LegEnd::CallsSandy { next_block }) => {
                                self.finish_leg(next_block)?;
                                let answer =
                                    self.next_answer(ctx, headers, start, duration).await?;
                                Some(Next::Answer(answer))
                            }
                            Some(LegEnd::AnswerEnds) | None => Some(Next::Ended(head)),
                        }
                    }
                },
                Next::Ended(head) => return Ok(ChainEvent::Ended(head)),
            };
        }
    }

    /// The chain's next request: a Sandy call when Sandy writes, else a Ricky
    /// leg.
    async fn next_answer(
        &mut self,
        ctx: &AppContext,
        headers: &HeaderMap,
        start: Instant,
        duration: &mut DurationGuard,
    ) -> Result<Answer, ApiError> {
        if self.config.sandy_call_tokens > 0 {
            self.call_sandy(ctx, headers, start, duration).await
        } else {
            self.send_ricky_leg(ctx, headers, start, duration, None)
                .await
        }
    }

    /// One Sandy call: the chain's Sandy writes at most `sandy_call_tokens` of
    /// the answer and, with taps, sends the taps of the trace and its text to a
    /// room at its engine.
    async fn call_sandy(
        &mut self,
        ctx: &AppContext,
        headers: &HeaderMap,
        start: Instant,
        duration: &mut DurationGuard,
    ) -> Result<Answer, ApiError> {
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
        let limits = self.limits.of_request(written, Some(call_tokens));
        let mut body = sandy_call_body(&self.body_without_prompt, input_ids, limits, tokenizer);
        let rid = self.next_rid("s");
        body.insert("rid".into(), rid.clone().into());
        if let Some(chain) = self.sandy_call_chain() {
            body.insert("franky_chain".into(), chain);
        }
        let mut request =
            PreparedRequest::generate_value(ctx, self.model.clone(), Value::Object(body), None)?;
        if self.taps {
            request.franky.tap_source = TapSource::SandyCall;
        }
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
        let Some(answered) = answered.filter(|_| response.status() == StatusCode::OK) else {
            return Ok(Answer::Failed(response));
        };
        if let Some(end) = &mut self.end {
            end.remember_engines(&answered.workers);
        }
        let sandy = &answered.workers.prefill;
        let room = answered.franky_room.map(|room| {
            Box::new(SandyRoom {
                room,
                unread: ctx
                    .proxy
                    .abort_on_drop(&sandy.url, sandy.protocol(), headers, Some(&rid)),
            })
        });
        self.sandy = Some(answered.workers);
        let (head, body) = response.into_parts();
        let bytes = axum::body::to_bytes(body, usize::MAX)
            .await
            .map_err(|error| ApiError::Internal(error.into()))?;
        let text = self.read_sandy_call(ctx, &bytes, limit, tokenizer)?;
        Ok(Answer::SandyText { head, text, room })
    }

    fn read_sandy_call(
        &self,
        ctx: &AppContext,
        bytes: &[u8],
        limit: SandyLimit,
        tokenizer: SandyTokenizer,
    ) -> Result<SandyText, ApiError> {
        let frame: Value =
            serde_json::from_slice(bytes).map_err(|error| ApiError::Internal(error.into()))?;
        let mut text = read_sandy_text(&frame, limit, tokenizer, &self.client_stop_ids);
        if tokenizer == SandyTokenizer::Own {
            text.ricky_ids = ctx
                .tokenizers
                .encode_answer_text(&self.model.0, &text.text)
                .map_err(|error| ApiError::Internal(error.context("tokenize Sandy's text")))?;
        }
        Ok(text)
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
    ) -> Result<Answer, ApiError> {
        let tap_source = match (self.taps, &room) {
            (false, _) => TapSource::None,
            (true, Some(room)) => TapSource::Room(room.room.clone()),
            (true, None) => {
                self.trace.start_tap_leg();
                TapSource::SandyLeg
            }
        };
        let franky = RickyLegFields {
            chain: self.ricky_leg_chain(),
            tap_runs: self.taps.then(|| wire(&self.trace.tap_runs())),
            sandy_input_ids: self
                .trace
                .sandy_ids()
                .filter(|_| self.taps)
                .map(|ids| json!(ids)),
        };
        let leg = if self.ricky.is_none() {
            Leg::First
        } else {
            Leg::Later
        };
        let limits = self.limits.of_request(self.trace.num_written(), None);
        let input_ids = json!(self.trace.ricky_ids());
        let mut body = ricky_leg_body(&self.body_without_prompt, input_ids, limits, leg, franky);
        body.insert("rid".into(), self.next_rid("r").into());
        let mut request =
            PreparedRequest::generate_value(ctx, self.model.clone(), Value::Object(body), None)?;
        if let (TapSource::SandyLeg, Some(sandy_ids)) = (&tap_source, self.trace.sandy_ids()) {
            request.set_franky_sandy_ids(sandy_ids);
        }
        request.franky.tap_source = tap_source;
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
        if let Some(answered) = answered {
            if let Some(end) = &mut self.end {
                end.remember_engines(&answered.workers);
            }
            self.ricky = Some(answered.workers);
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
    /// Returns: the chain's frame of the text.
    fn add_sandy_text(&mut self, text: &SandyText) -> Value {
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
            .add_sandy_text(&text.text, &text.ricky_ids, finish_reason)
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

/// `ids` as token ids, when every one is one.
fn token_ids(ids: &Value) -> Option<Vec<u32>> {
    (ids.as_array()?.iter())
        .map(|id| id.as_u64().and_then(|id| u32::try_from(id).ok()))
        .collect()
}
