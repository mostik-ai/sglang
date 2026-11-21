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
    /// The text in Ricky's ids.
    pub(super) ricky_ids: Vec<u32>,
    /// The text in Sandy's ids.
    pub(super) sandy_ids: Vec<u32>,
    /// The call stopped at a client stop rule or the answer's token limit.
    pub(super) ends_answer: bool,
    pub(super) finish_reason: Value,
}

/// Text of a Sandy call that the answer gets, and Ricky's ids of it.
#[derive(Debug, Default, PartialEq)]
pub(super) struct SandyDelta {
    pub(super) text: String,
    pub(super) ids: Vec<u32>,
}

/// What a frame of a Sandy call brings.
#[derive(Debug, PartialEq)]
pub(super) enum SandyRead {
    /// Text the answer gets now.
    Adds(SandyDelta),
    /// The call ended with `text`, of which the answer gets `adds` now.
    Ends { text: SandyText, adds: SandyDelta },
}

/// The `/generate` body of a Sandy call that writes the answer after
/// `input_ids` within `limits`, with the client's sampling and stop rules,
/// but the stop rules in Ricky's ids only for a Sandy that reads them. The
/// call also stops at `text_end`, the token where the template of Sandy's
/// input starts closing an answer before Sandy's EOS.
pub(super) fn sandy_call_body(
    client_body: &Map<String, Value>,
    input_ids: Value,
    limits: WriteLimits,
    text_end: Option<u32>,
    tokenizer: SandyTokenizer,
    streamed: bool,
) -> Map<String, Value> {
    let mut body = client_body.clone();
    body.insert("input_ids".into(), input_ids);
    body.insert("stream".into(), streamed.into());
    let sampling = body.entry("sampling_params").or_insert_with(|| json!({}));
    limits.set_on(sampling);
    if let Value::Object(sampling) = sampling {
        if tokenizer == SandyTokenizer::Own {
            for field in TOKEN_STOP_FIELDS {
                sampling.remove(field);
            }
        }
        if let Some(text_end) = text_end {
            let stops = sampling
                .entry("stop_token_ids")
                .or_insert_with(|| json!([]));
            if let Value::Array(stops) = stops {
                stops.push(text_end.into());
            }
        }
    }
    body
}

/// A Sandy call's text as the frames of its response bring it, unary or
/// streamed.
#[derive(Debug)]
pub(super) struct SandyStream {
    tokenizer: SandyTokenizer,
    limit: SandyLimit,
    increments: bool,
    text: String,
    ids: Vec<u32>,
}

impl SandyStream {
    /// Args:
    ///     increments: each frame holds only what it adds, not the text so
    ///         far, as in a streamed call to a Sandy with incremental output.
    pub(super) fn new(tokenizer: SandyTokenizer, limit: SandyLimit, increments: bool) -> Self {
        Self {
            tokenizer,
            limit,
            increments,
            text: String::new(),
            ids: Vec::new(),
        }
    }

    /// Reads the call's next `frame`. Sandy's own EOS ends only its text and
    /// is dropped. A Sandy that shares Ricky's tokenizer adds its text as it
    /// comes; one with its own adds it all once the call ends, in the ids
    /// `ricky_ids_of` gives.
    ///
    /// Args:
    ///     client_stop_ids: the client's stop token ids that went to Sandy.
    /// Errors: those of `ricky_ids_of`.
    pub(super) fn read(
        &mut self,
        frame: &Value,
        client_stop_ids: &[u64],
        ricky_ids_of: impl FnOnce(&str) -> anyhow::Result<Vec<u32>>,
    ) -> anyhow::Result<SandyRead> {
        let finish_reason = &frame["meta_info"]["finish_reason"];
        let added = self.add(frame, finish_reason, client_stop_ids);
        if finish_reason.is_null() {
            return Ok(SandyRead::Adds(match self.tokenizer {
                SandyTokenizer::Ricky => added,
                SandyTokenizer::Own => SandyDelta::default(),
            }));
        }
        let adds = match self.tokenizer {
            SandyTokenizer::Ricky => added,
            SandyTokenizer::Own => SandyDelta {
                ids: ricky_ids_of(&self.text)?,
                text: std::mem::take(&mut self.text),
            },
        };
        let ricky_ids = match self.tokenizer {
            SandyTokenizer::Ricky => self.ids.clone(),
            SandyTokenizer::Own => adds.ids.clone(),
        };
        let text = SandyText {
            ricky_ids,
            sandy_ids: std::mem::take(&mut self.ids),
            ends_answer: self.ends_answer(finish_reason, client_stop_ids),
            finish_reason: finish_reason.clone(),
        };
        Ok(SandyRead::Ends { text, adds })
    }

    fn add(&mut self, frame: &Value, finish_reason: &Value, client_stop_ids: &[u64]) -> SandyDelta {
        let mut ids: Vec<u32> = (frame["output_ids"].as_array())
            .map(|ids| {
                (ids.iter())
                    .filter_map(|id| id.as_u64().and_then(|id| u32::try_from(id).ok()))
                    .collect()
            })
            .unwrap_or_default();
        let text = frame["text"].as_str().unwrap_or_default();
        let (mut new_ids, new_text) = if self.increments {
            (ids, text.to_owned())
        } else {
            let seen = self.ids.len().min(ids.len());
            let new_text = text.strip_prefix(self.text.as_str()).unwrap_or_default();
            (ids.split_off(seen), new_text.to_owned())
        };
        if stopped_at_eos(finish_reason, client_stop_ids)
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

    fn ends_answer(&self, finish_reason: &Value, client_stop_ids: &[u64]) -> bool {
        let at_eos = stopped_at_eos(finish_reason, client_stop_ids).is_some();
        let at_answer_limit = finish_reason["type"] == "length" && self.limit == SandyLimit::Answer;
        let at_client_stop = finish_reason["type"] == "stop" && !at_eos;
        at_answer_limit || at_client_stop
    }
}

/// The id of Sandy's own EOS, or of its template's text end, when it stopped
/// the call: a stop token that is none of the client's.
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

    fn no_ricky_ids(_: &str) -> anyhow::Result<Vec<u32>> {
        unreachable!("a Sandy that shares Ricky's tokenizer gives Ricky's ids")
    }

    fn read_unary(frame: &Value, limit: SandyLimit, client_stop_ids: &[u64]) -> SandyRead {
        let mut stream = SandyStream::new(SandyTokenizer::Ricky, limit, false);
        stream.read(frame, client_stop_ids, no_ricky_ids).unwrap()
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

        let body = |input_ids: Value, text_end, tokenizer| {
            sandy_call_body(&client, input_ids, limits, text_end, tokenizer, false)
        };
        let shared = body(json!([1, 2]), None, SandyTokenizer::Ricky);
        let own = body(json!([9]), None, SandyTokenizer::Own);
        let shared_closing = body(json!([1, 2]), Some(5), SandyTokenizer::Ricky);
        let own_closing = body(json!([9]), Some(5), SandyTokenizer::Own);

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
        assert_eq!(
            shared_closing["sampling_params"]["stop_token_ids"],
            json!([7, 5])
        );
        assert_eq!(
            own_closing["sampling_params"],
            json!({"temperature": 0.5, "stop": ["\n\n"], "stop_token_ids": [5], "max_new_tokens": 64})
        );
    }

    #[test]
    fn sandys_eos_ends_only_its_text_and_leaves_the_trace() {
        let eos = json!({"type": "stop", "matched": 2});

        let read = read_unary(&frame("ab", &[5, 6, 2], eos.clone()), SandyLimit::Call, &[]);

        let SandyRead::Ends { text, adds } = read else {
            panic!("a finished frame ends the call");
        };
        assert_eq!(text.ricky_ids, vec![5, 6]);
        assert_eq!(text.sandy_ids, vec![5, 6]);
        assert!(!text.ends_answer);
        assert_eq!(text.finish_reason, eos);
        assert_eq!(
            adds,
            SandyDelta {
                text: "ab".into(),
                ids: vec![5, 6]
            }
        );
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
            let read = read_unary(&frame("ab", &[5, 7], finish.clone()), limit, &[7]);

            let SandyRead::Ends { text, .. } = read else {
                panic!("a finished frame ends the call");
            };
            assert_eq!(text.ends_answer, ends, "{finish}");
            assert_eq!(text.ricky_ids.len(), 2, "{finish}");
        }
    }

    #[test]
    fn a_stream_adds_what_each_frame_adds_whole_or_incremental() {
        let eos = json!({"type": "stop", "matched": 2});
        for (increments, frames) in [
            (
                false,
                [
                    frame("a", &[5], Value::Null),
                    frame("ab", &[5, 6], Value::Null),
                    frame("ab", &[5, 6, 2], eos.clone()),
                ],
            ),
            (
                true,
                [
                    frame("a", &[5], Value::Null),
                    frame("b", &[6], Value::Null),
                    frame("", &[2], eos.clone()),
                ],
            ),
        ] {
            let mut stream = SandyStream::new(SandyTokenizer::Ricky, SandyLimit::Call, increments);

            let reads: Vec<_> = (frames.iter())
                .map(|frame| stream.read(frame, &[], no_ricky_ids).unwrap())
                .collect();

            let delta = |text: &str, ids: Vec<u32>| SandyDelta {
                text: text.into(),
                ids,
            };
            assert_eq!(reads[0], SandyRead::Adds(delta("a", vec![5])));
            assert_eq!(reads[1], SandyRead::Adds(delta("b", vec![6])));
            let SandyRead::Ends { text, adds } = &reads[2] else {
                panic!("the last frame ends the call");
            };
            assert_eq!(adds, &delta("", vec![]));
            assert_eq!(text.sandy_ids, vec![5, 6]);
            assert!(!text.ends_answer);
        }
    }

    #[test]
    fn a_sandy_with_its_own_tokenizer_adds_its_text_once_the_call_ends() {
        let mut stream = SandyStream::new(SandyTokenizer::Own, SandyLimit::Call, true);
        let ricky_ids_of = |text: &str| {
            assert_eq!(text, "ab");
            Ok(vec![40, 41, 42])
        };

        let first = stream.read(&frame("a", &[5], Value::Null), &[], no_ricky_ids);
        let last = stream.read(
            &frame("b", &[6], json!({"type": "length", "length": 2})),
            &[],
            ricky_ids_of,
        );

        assert_eq!(first.unwrap(), SandyRead::Adds(SandyDelta::default()));
        let SandyRead::Ends { text, adds } = last.unwrap() else {
            panic!("the last frame ends the call");
        };
        assert_eq!(text.sandy_ids, vec![5, 6]);
        assert_eq!(text.ricky_ids, vec![40, 41, 42]);
        assert_eq!(
            adds,
            SandyDelta {
                text: "ab".into(),
                ids: vec![40, 41, 42]
            }
        );
    }
}
