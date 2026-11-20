// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use super::answer_limits::WriteLimits;
use serde_json::{json, Map, Value};

/// The client's stop rules in Ricky's ids, which go only to a Sandy that
/// shares Ricky's tokenizer.
const TOKEN_STOP_FIELDS: [&str; 1] = ["stop_token_ids"];

/// Which tokenizer a chain's Sandy reads with.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum SandyTokenizer {
    Ricky,
    Own,
}

/// Which limit a Sandy call's `max_new_tokens` was.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum SandyLimit {
    /// The call's own tokens, with more of the answer left after them.
    Call,
    /// All that was left of the answer, so reaching it ends the answer.
    Answer,
}

/// What a Sandy call wrote, and whether it ended the answer.
#[derive(Debug, PartialEq)]
pub(super) struct SandyText {
    pub(super) text: String,
    /// The text in Ricky's ids.
    pub(super) ricky_ids: Vec<u32>,
    /// The text in Sandy's ids.
    pub(super) sandy_ids: Vec<u32>,
    /// The call stopped at a client stop rule or the answer's token limit.
    pub(super) ends_answer: bool,
    pub(super) finish_reason: Value,
}

/// The `/generate` body of a Sandy call that writes the answer after
/// `input_ids` within `limits`, with the client's sampling and stop rules,
/// but the stop rules in Ricky's ids only for a Sandy that reads them.
pub(super) fn sandy_call_body(
    client_body: &Map<String, Value>,
    input_ids: Value,
    limits: WriteLimits,
    tokenizer: SandyTokenizer,
    streamed: bool,
) -> Map<String, Value> {
    let mut body = client_body.clone();
    body.insert("input_ids".into(), input_ids);
    body.insert("stream".into(), streamed.into());
    let sampling = body.entry("sampling_params").or_insert_with(|| json!({}));
    limits.set_on(sampling);
    if tokenizer == SandyTokenizer::Own {
        if let Value::Object(sampling) = sampling {
            for field in TOKEN_STOP_FIELDS {
                sampling.remove(field);
            }
        }
    }
    body
}

/// Reads a Sandy call's unary `/generate` frame, its text in Sandy's ids and,
/// for a Sandy that shares Ricky's tokenizer, in Ricky's too; a Sandy with its
/// own tokenizer leaves `ricky_ids` to the caller.
///
/// `client_stop_ids` are the client's stop token ids that went to Sandy.
/// Sandy's own EOS ends only its text and is dropped.
pub(super) fn read_sandy_text(
    frame: &Value,
    limit: SandyLimit,
    tokenizer: SandyTokenizer,
    client_stop_ids: &[u64],
) -> SandyText {
    let mut stream = SandyStream::default();
    stream.add(frame, client_stop_ids);
    let mut text = stream.finish(&frame["meta_info"]["finish_reason"], limit, client_stop_ids);
    if tokenizer == SandyTokenizer::Own {
        text.ricky_ids.clear();
    }
    text
}

/// A Sandy call's text as its frames bring it, for a Sandy that shares
/// Ricky's tokenizer: a frame holds either the text so far or only what it
/// adds, which its ids tell, since the text so far has all the call's
/// `completion_tokens`.
#[derive(Debug, Default)]
pub(super) struct SandyStream {
    text: String,
    ids: Vec<u32>,
}

/// What one frame of a Sandy call adds to its text.
#[derive(Debug, PartialEq)]
pub(super) struct SandyDelta {
    pub(super) text: String,
    pub(super) ids: Vec<u32>,
}

impl SandyStream {
    /// What `frame` adds to the call's text, without Sandy's own EOS.
    pub(super) fn add(&mut self, frame: &Value, client_stop_ids: &[u64]) -> SandyDelta {
        let mut ids: Vec<u32> = (frame["output_ids"].as_array())
            .map(|ids| {
                (ids.iter())
                    .filter_map(|id| id.as_u64().and_then(|id| u32::try_from(id).ok()))
                    .collect()
            })
            .unwrap_or_default();
        let text = frame["text"].as_str().unwrap_or_default();
        let whole = frame["meta_info"]["completion_tokens"].as_u64() == Some(ids.len() as u64);
        let (mut new_ids, new_text) = if whole {
            let seen = self.ids.len().min(ids.len());
            let new_text = text.strip_prefix(self.text.as_str()).unwrap_or_default();
            (ids.split_off(seen), new_text.to_owned())
        } else {
            (ids, text.to_owned())
        };
        if stopped_at_eos(&frame["meta_info"]["finish_reason"], client_stop_ids)
            .is_some_and(|eos| new_ids.last().map(|&id| u64::from(id)) == Some(eos))
        {
            new_ids.pop();
        }
        self.ids.extend_from_slice(&new_ids);
        self.text.push_str(&new_text);
        SandyDelta {
            text: new_text,
            ids: new_ids,
        }
    }

    /// The call's whole text, once a frame ended it with `finish_reason`.
    pub(super) fn finish(
        self,
        finish_reason: &Value,
        limit: SandyLimit,
        client_stop_ids: &[u64],
    ) -> SandyText {
        let at_eos = stopped_at_eos(finish_reason, client_stop_ids).is_some();
        let at_answer_limit = finish_reason["type"] == "length" && limit == SandyLimit::Answer;
        let at_client_stop = finish_reason["type"] == "stop" && !at_eos;
        SandyText {
            text: self.text,
            ricky_ids: self.ids.clone(),
            sandy_ids: self.ids,
            ends_answer: at_answer_limit || at_client_stop,
            finish_reason: finish_reason.clone(),
        }
    }
}

/// The id of Sandy's own EOS when it stopped the call: a stop token that is
/// none of the client's.
fn stopped_at_eos(finish_reason: &Value, client_stop_ids: &[u64]) -> Option<u64> {
    let matched = finish_reason["matched"].as_u64()?;
    (finish_reason["type"] == "stop" && !client_stop_ids.contains(&matched)).then_some(matched)
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

        let shared = sandy_call_body(&client, json!([1, 2]), limits, SandyTokenizer::Ricky, false);
        let own = sandy_call_body(&client, json!([9]), limits, SandyTokenizer::Own, false);

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

        let text = read_sandy_text(
            &frame("ab", &[5, 6, 2], eos.clone()),
            SandyLimit::Call,
            SandyTokenizer::Ricky,
            &[],
        );

        assert_eq!(text.ricky_ids, vec![5, 6]);
        assert_eq!(text.sandy_ids, vec![5, 6]);
        assert!(!text.ends_answer);
        assert_eq!(text.finish_reason, eos);
    }

    #[test]
    fn a_client_stop_rule_or_the_answers_limit_ends_the_answer() {
        for (finish, limit, ends) in [
            (
                json!({"type": "stop", "matched": "\n\n"}),
                SandyLimit::Call,
                true,
            ),
            (
                json!({"type": "stop", "matched": 7}),
                SandyLimit::Call,
                true,
            ),
            (
                json!({"type": "length", "length": 64}),
                SandyLimit::Answer,
                true,
            ),
            (
                json!({"type": "length", "length": 64}),
                SandyLimit::Call,
                false,
            ),
        ] {
            let text = read_sandy_text(
                &frame("ab", &[5, 7], finish.clone()),
                limit,
                SandyTokenizer::Ricky,
                &[7],
            );

            assert_eq!(text.ends_answer, ends, "{finish}");
            assert_eq!(text.ricky_ids.len(), 2, "{finish}");
        }
    }

    #[test]
    fn a_stream_adds_what_each_frame_adds_whole_or_not() {
        let eos = json!({"type": "stop", "matched": 2});
        let frame = |text: &str, ids: &[u64], completion: u64, finish: Value| {
            json!({"text": text, "output_ids": ids,
                   "meta_info": {"completion_tokens": completion, "finish_reason": finish}})
        };
        for frames in [
            vec![
                frame("a", &[5], 1, Value::Null),
                frame("ab", &[5, 6], 2, Value::Null),
                frame("ab", &[5, 6, 2], 3, eos.clone()),
            ],
            vec![
                frame("a", &[5], 1, Value::Null),
                frame("b", &[6], 2, Value::Null),
                frame("", &[2], 3, eos.clone()),
            ],
        ] {
            let mut stream = SandyStream::default();
            let deltas: Vec<_> = frames.iter().map(|frame| stream.add(frame, &[])).collect();
            let text = stream.finish(&eos, SandyLimit::Call, &[]);

            assert_eq!(
                deltas,
                [
                    SandyDelta {
                        text: "a".into(),
                        ids: vec![5]
                    },
                    SandyDelta {
                        text: "b".into(),
                        ids: vec![6]
                    },
                    SandyDelta {
                        text: "".into(),
                        ids: vec![]
                    },
                ]
            );
            assert_eq!((text.text.as_str(), text.sandy_ids), ("ab", vec![5, 6]));
            assert!(!text.ends_answer);
        }
    }

    #[test]
    fn a_sandy_with_its_own_tokenizer_leaves_rickys_ids_to_the_caller() {
        let text = read_sandy_text(
            &frame("ab", &[5, 6], json!({"type": "length", "length": 2})),
            SandyLimit::Call,
            SandyTokenizer::Own,
            &[],
        );

        assert_eq!(text.sandy_ids, vec![5, 6]);
        assert!(text.ricky_ids.is_empty());
    }
}
