// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use super::answer_limits::WriteLimits;
use serde_json::{json, Map, Value};

/// The client's stop rules in Ricky's ids, which go only to a Sandy that
/// shares Ricky's tokenizer.
const TOKEN_STOP_FIELDS: [&str; 1] = ["stop_token_ids"];

/// What a Sandy call wrote, and whether it ended the answer.
#[derive(Debug, PartialEq)]
pub(super) struct SandyText {
    pub(super) text: String,
    /// The text in Ricky's ids, with a Sandy that reads them.
    pub(super) ricky_ids: Vec<Value>,
    /// The text in Sandy's ids.
    pub(super) sandy_ids: Vec<u32>,
    /// The call stopped at a client stop rule or the answer's token limit.
    pub(super) ends_answer: bool,
    pub(super) finish_reason: Value,
}

/// The unary `/generate` body of a Sandy call that writes the answer after
/// `input_ids` within `limits`, with the client's sampling and stop rules.
/// `own_tokenizer`: Sandy tokenizes on its own, so the client's stop rules in
/// Ricky's ids stay out.
pub(super) fn sandy_call_body(
    client_body: &Map<String, Value>,
    input_ids: Value,
    limits: WriteLimits,
    own_tokenizer: bool,
) -> Map<String, Value> {
    let mut body = client_body.clone();
    body.insert("input_ids".into(), input_ids);
    body.insert("stream".into(), false.into());
    let sampling = body.entry("sampling_params").or_insert_with(|| json!({}));
    limits.set_on(sampling);
    if own_tokenizer {
        if let Value::Object(sampling) = sampling {
            for field in TOKEN_STOP_FIELDS {
                sampling.remove(field);
            }
        }
    }
    body
}

/// Reads a Sandy call's unary `/generate` frame.
///
/// `limit_is_the_answers`: the call could write all that was left of the
/// answer, so its token limit ends the answer. `client_stop_ids` are the
/// client's stop token ids that went to Sandy. Sandy's own EOS ends only its
/// text and is dropped.
pub(super) fn read_sandy_text(
    frame: &Value,
    limit_is_the_answers: bool,
    client_stop_ids: &[u64],
) -> SandyText {
    let finish_reason = frame["meta_info"]["finish_reason"].clone();
    let mut ids: Vec<Value> = frame["output_ids"].as_array().cloned().unwrap_or_default();
    let matched = &finish_reason["matched"];
    let stopped_at_eos = finish_reason["type"] == "stop"
        && matched
            .as_u64()
            .is_some_and(|id| !client_stop_ids.contains(&id));
    if stopped_at_eos && ids.last() == Some(matched) {
        ids.pop();
    }
    let at_answer_limit = finish_reason["type"] == "length" && limit_is_the_answers;
    let at_client_stop = finish_reason["type"] == "stop" && !stopped_at_eos;
    SandyText {
        text: frame["text"].as_str().unwrap_or_default().to_owned(),
        sandy_ids: ids
            .iter()
            .filter_map(|id| id.as_u64())
            .map(|id| id as u32)
            .collect(),
        ricky_ids: ids,
        ends_answer: at_answer_limit || at_client_stop,
        finish_reason,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(text: &str, ids: &[u64], finish: Value) -> Value {
        json!({"text": text, "output_ids": ids, "meta_info": {"finish_reason": finish}})
    }

    #[test]
    fn a_call_writes_with_the_clients_sampling_and_stops_up_to_its_tokens() {
        let mut client = Map::new();
        client.insert(
            "sampling_params".into(),
            json!({"temperature": 0.5, "stop": ["\n\n"], "stop_token_ids": [7]}),
        );
        client.insert("stream".into(), true.into());

        let limits = WriteLimits {
            max: Some(64),
            min: None,
        };
        let shared = sandy_call_body(&client, json!([1, 2]), limits, false);
        let own = sandy_call_body(&client, json!([9]), limits, true);

        assert_eq!(shared["input_ids"], json!([1, 2]));
        assert_eq!(shared["stream"], json!(false));
        assert_eq!(
            shared["sampling_params"],
            json!({"temperature": 0.5, "stop": ["\n\n"], "stop_token_ids": [7], "max_new_tokens": 64})
        );
        assert_eq!(
            own["sampling_params"],
            json!({"temperature": 0.5, "stop": ["\n\n"], "max_new_tokens": 64})
        );
    }

    #[test]
    fn sandys_eos_ends_only_its_text_and_leaves_the_trace() {
        let eos = json!({"type": "stop", "matched": 2});

        let text = read_sandy_text(&frame("ab", &[5, 6, 2], eos.clone()), false, &[]);

        assert_eq!(text.ricky_ids, vec![json!(5), json!(6)]);
        assert_eq!(text.sandy_ids, vec![5, 6]);
        assert!(!text.ends_answer);
        assert_eq!(text.finish_reason, eos);
    }

    #[test]
    fn a_client_stop_rule_or_the_answers_limit_ends_the_answer() {
        for (finish, limit_is_the_answers, ends) in [
            (json!({"type": "stop", "matched": "\n\n"}), false, true),
            (json!({"type": "stop", "matched": 7}), false, true),
            (json!({"type": "length", "length": 64}), true, true),
            (json!({"type": "length", "length": 64}), false, false),
        ] {
            let text = read_sandy_text(
                &frame("ab", &[5, 7], finish.clone()),
                limit_is_the_answers,
                &[7],
            );

            assert_eq!(text.ends_answer, ends, "{finish}");
            assert_eq!(text.ricky_ids.len(), 2, "{finish}");
        }
    }
}
