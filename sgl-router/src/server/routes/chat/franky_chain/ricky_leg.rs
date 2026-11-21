// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use super::answer_limits::WriteLimits;
use serde_json::{json, Map, Value};

/// The finish reason of a Ricky leg that ends where Sandy writes next.
const SANDY_CALL_FINISH: &str = "franky_sandy_call";

/// Which of a chain's legs a body is for.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Leg {
    First,
    Later,
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

/// The `/generate` body of a Ricky leg that continues the trace `input_ids`
/// within `limits`, with the client's sampling and stop rules. A later leg
/// returns no prompt logprobs, which the first leg already did.
///
/// Args:
///     chain: `franky_chain`: the chain, the leg's first block and Ricky's
///         blocks; `None` for a chain without blocks, whose engines hold nothing.
pub(super) fn ricky_leg_body(
    client_body: &Map<String, Value>,
    input_ids: Value,
    limits: WriteLimits,
    leg: Leg,
    chain: Option<Value>,
) -> Map<String, Value> {
    let mut body = client_body.clone();
    body.insert("input_ids".into(), input_ids);
    let sampling = body.entry("sampling_params").or_insert_with(|| json!({}));
    limits.set_on(sampling);
    if leg == Leg::Later && body.contains_key("logprob_start_len") {
        body.insert("logprob_start_len".into(), (-1).into());
    }
    if let Some(chain) = chain {
        body.insert("franky_chain".into(), chain);
    }
    body
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_leg_continues_the_trace_with_what_is_left_and_its_chain() {
        let mut client = Map::new();
        client.insert("sampling_params".into(), json!({"temperature": 0.0}));
        client.insert("logprob_start_len".into(), json!(0));
        let chain = Some(json!({"id": "c", "block": 3}));
        let limits = WriteLimits {
            max: Some(10),
            min: None,
        };

        let body = ricky_leg_body(&client, json!([1, 2, 3, 4]), limits, Leg::Later, chain);

        assert_eq!(body["input_ids"], json!([1, 2, 3, 4]));
        assert_eq!(
            body["sampling_params"],
            json!({"temperature": 0.0, "max_new_tokens": 10})
        );
        assert_eq!(body["logprob_start_len"], json!(-1));
        assert_eq!(body["franky_chain"], json!({"id": "c", "block": 3}));
    }

    #[test]
    fn without_a_limit_a_leg_writes_up_to_the_context() {
        let limits = WriteLimits {
            max: None,
            min: None,
        };
        let body = ricky_leg_body(&Map::new(), json!([1]), limits, Leg::First, None);

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
