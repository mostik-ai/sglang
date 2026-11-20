// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use super::super::openai::sse_data;
use super::answer_limits::WriteLimits;
use axum::body::Body;
use bytes::Bytes;
use futures::stream::{self, BoxStream};
use futures::{Stream, StreamExt};
use serde_json::{json, Map, Value};

/// The finish reason of a Ricky leg that ends where Sandy writes next.
const SANDY_CALL_FINISH: &str = "franky_sandy_call";

/// Which of a chain's legs a body is for.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Leg {
    First,
    Later,
}

/// What a Ricky leg carries for Franky beside the client's fields.
#[derive(Debug, Default)]
pub(super) struct RickyLegFields {
    /// `franky_chain`: the chain, the leg's first block and Ricky's blocks;
    /// `None` for a chain without blocks, whose engines hold nothing.
    pub(super) chain: Option<Value>,
    /// `franky_tap_runs`; `None` for a leg that reads no taps.
    pub(super) tap_runs: Option<Value>,
    /// `franky_sandy_input_ids`, when Sandy tokenizes on its own.
    pub(super) sandy_input_ids: Option<Value>,
}

/// How a Ricky leg ended, as the finish reason of its last frame says.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum LegEnd {
    /// Sandy writes next; Ricky's block after the call is `next_block`.
    CallsSandy { next_block: u64 },
    /// EOS, a stop rule or the limit ended the answer.
    AnswerEnds,
}

impl LegEnd {
    /// How the leg ends by `frame`; `None` while it goes on.
    pub(super) fn of_frame(frame: &Value) -> Option<Self> {
        let finish_reason = &frame["meta_info"]["finish_reason"];
        if finish_reason.is_null() {
            return None;
        }
        if finish_reason["type"] != SANDY_CALL_FINISH {
            return Some(Self::AnswerEnds);
        }
        let next_block = finish_reason["next_block"].as_u64()?;
        Some(Self::CallsSandy { next_block })
    }
}

/// One frame of a Ricky leg, as its engine sent it.
pub(super) enum LegFrame {
    Frame(Value),
    /// An error frame of a streamed leg, which ends it.
    Error(Bytes),
}

/// The frames of a Ricky leg's response: its one frame when unary, its SSE
/// frames when streamed.
pub(super) fn leg_frames(
    body: Body,
    streamed: bool,
) -> BoxStream<'static, Result<LegFrame, axum::Error>> {
    if !streamed {
        return stream::once(async move {
            let bytes = axum::body::to_bytes(body, usize::MAX).await?;
            let frame = serde_json::from_slice(&bytes).map_err(axum::Error::new)?;
            Ok(LegFrame::Frame(frame))
        })
        .boxed();
    }
    streamed_frames(body).boxed()
}

fn streamed_frames(body: Body) -> impl Stream<Item = Result<LegFrame, axum::Error>> {
    let not_done = |data: &Result<Bytes, axum::Error>| {
        futures::future::ready(!matches!(data, Ok(data) if data.as_ref() == b"[DONE]"))
    };
    sse_data(body).filter(not_done).map(|data| {
        let data = data?;
        Ok(match serde_json::from_slice::<Value>(&data) {
            Ok(frame) if frame.get("error").is_none() => LegFrame::Frame(frame),
            _ => LegFrame::Error(data),
        })
    })
}

/// The `/generate` body of a Ricky leg that continues the trace `input_ids`
/// within `limits`, with the client's sampling and stop rules. A later leg
/// returns no prompt logprobs, which the first leg already did.
pub(super) fn ricky_leg_body(
    client_body: &Map<String, Value>,
    input_ids: Value,
    limits: WriteLimits,
    leg: Leg,
    franky: RickyLegFields,
) -> Map<String, Value> {
    let mut body = client_body.clone();
    body.insert("input_ids".into(), input_ids);
    let sampling = body.entry("sampling_params").or_insert_with(|| json!({}));
    limits.set_on(sampling);
    if leg == Leg::Later && body.contains_key("logprob_start_len") {
        body.insert("logprob_start_len".into(), (-1).into());
    }
    let fields = [
        ("franky_chain", franky.chain),
        ("franky_tap_runs", franky.tap_runs),
        ("franky_sandy_input_ids", franky.sandy_input_ids),
    ];
    for (name, value) in fields {
        if let Some(value) = value {
            body.insert(name.into(), value);
        }
    }
    body
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_leg_continues_the_trace_with_what_is_left_and_its_franky_fields() {
        let mut client = Map::new();
        client.insert("sampling_params".into(), json!({"temperature": 0.0}));
        client.insert("logprob_start_len".into(), json!(0));
        let franky = RickyLegFields {
            chain: Some(json!({"id": "c", "block": 3})),
            tap_runs: Some(json!([[4, 4]])),
            sandy_input_ids: None,
        };
        let limits = WriteLimits {
            max: Some(10),
            min: None,
        };

        let body = ricky_leg_body(&client, json!([1, 2, 3, 4]), limits, Leg::Later, franky);

        assert_eq!(body["input_ids"], json!([1, 2, 3, 4]));
        assert_eq!(
            body["sampling_params"],
            json!({"temperature": 0.0, "max_new_tokens": 10})
        );
        assert_eq!(body["logprob_start_len"], json!(-1));
        assert_eq!(body["franky_chain"], json!({"id": "c", "block": 3}));
        assert_eq!(body["franky_tap_runs"], json!([[4, 4]]));
        assert!(!body.contains_key("franky_sandy_input_ids"));
    }

    #[test]
    fn without_a_limit_a_leg_writes_up_to_the_context() {
        let limits = WriteLimits {
            max: None,
            min: None,
        };
        let body = ricky_leg_body(
            &Map::new(),
            json!([1]),
            limits,
            Leg::First,
            RickyLegFields::default(),
        );

        assert_eq!(body["sampling_params"], json!({"max_new_tokens": null}));
    }

    #[test]
    fn the_last_frame_says_whether_sandy_writes_next() {
        let frame = |finish: Value| json!({"meta_info": {"finish_reason": finish}});

        assert_eq!(LegEnd::of_frame(&frame(Value::Null)), None);
        assert_eq!(
            LegEnd::of_frame(&frame(
                json!({"type": "franky_sandy_call", "next_block": 3})
            )),
            Some(LegEnd::CallsSandy { next_block: 3 })
        );
        assert_eq!(
            LegEnd::of_frame(&frame(json!({"type": "stop", "matched": 2}))),
            Some(LegEnd::AnswerEnds)
        );
    }
}
