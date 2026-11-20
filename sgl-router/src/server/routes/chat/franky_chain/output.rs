// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use super::ricky_leg::LegEnd;
use serde_json::{json, Map, Value};

const PER_TOKEN_FIELDS: [&str; 5] = [
    "output_token_logprobs",
    "output_top_logprobs",
    "output_token_ids_logprobs",
    "output_token_sampling_mask",
    "output_token_sampling_logprobs",
];
const PROMPT_FIELDS: [&str; 6] = [
    "prompt_tokens",
    "cached_tokens",
    "cached_tokens_details",
    "input_token_logprobs",
    "input_top_logprobs",
    "input_token_ids_logprobs",
];
const SUMMED_COUNTS: [&str; 2] = ["completion_tokens", "reasoning_tokens"];
const LEG_REPORT_FIELDS: [&str; 7] = [
    "prompt_tokens",
    "cached_tokens",
    "request_received_ts",
    "queue_time",
    "forward_entry_time",
    "first_token_latency",
    "request_finished_ts",
];

#[derive(Default)]
struct Generated {
    text: String,
    output_ids: Vec<u32>,
    token_fields: Map<String, Value>,
    counts: [u64; SUMMED_COUNTS.len()],
}

impl Generated {
    fn of_frame(frame: &Value) -> Self {
        let mut generated = Self::default();
        generated.extend_by_frame(frame);
        generated.counts =
            SUMMED_COUNTS.map(|count| frame["meta_info"][count].as_u64().unwrap_or(0));
        generated
    }

    fn of_sandy_text(text: &str, ids: &[u32]) -> Self {
        Self {
            text: text.to_owned(),
            output_ids: ids.to_vec(),
            token_fields: Map::new(),
            counts: [ids.len() as u64, 0],
        }
    }

    fn extend_by_frame(&mut self, frame: &Value) {
        if let Some(text) = frame["text"].as_str() {
            self.text.push_str(text);
        }
        if let Some(ids) = frame["output_ids"].as_array() {
            let ids = ids.iter().filter_map(Value::as_u64);
            self.output_ids
                .extend(ids.filter_map(|id| u32::try_from(id).ok()));
        }
        for field in PER_TOKEN_FIELDS {
            if let Some(values) = frame["meta_info"][field].as_array() {
                append(&mut self.token_fields, field, values);
            }
        }
    }

    fn extend(&mut self, other: Generated) {
        self.text.push_str(&other.text);
        self.output_ids.extend(other.output_ids);
        for (field, values) in other.token_fields {
            if let Value::Array(values) = values {
                append(&mut self.token_fields, &field, &values);
            }
        }
        for (count, other) in self.counts.iter_mut().zip(other.counts) {
            *count += other;
        }
    }

    fn prepend_to(&self, frame: &mut Map<String, Value>, meta: &mut Map<String, Value>) {
        if let Some(Value::String(text)) = frame.get_mut("text") {
            text.insert_str(0, &self.text);
        }
        if let Some(Value::Array(ids)) = frame.get_mut("output_ids") {
            ids.splice(0..0, self.output_ids.iter().map(|&id| Value::from(id)));
        }
        for (field, values) in &self.token_fields {
            if let (Some(Value::Array(frame_values)), Value::Array(values)) =
                (meta.get_mut(field), values)
            {
                frame_values.splice(0..0, values.iter().cloned());
            }
        }
    }
}

fn fields_of(meta: &Value, names: &[&str]) -> Map<String, Value> {
    names
        .iter()
        .filter_map(|name| Some(((*name).to_string(), meta.get(name)?.clone())))
        .collect()
}

fn append(fields: &mut Map<String, Value>, field: &str, values: &[Value]) {
    if let Value::Array(existing) = fields
        .entry(field)
        .or_insert_with(|| Value::Array(Vec::new()))
    {
        existing.extend(values.iter().cloned());
    }
}

/// A chain's answer so far, and its frames: Sandy's text and each Ricky leg's
/// `/generate` frames become frames of one request that names `id`, with the
/// client's prompt and one finish reason. Frames are whole, the answer so far,
/// unless the client streams incrementally. The prompt fields are the first
/// leg's, with its cache hits counted within the client's prompt, which a Sandy
/// call's text may follow in that leg's input.
pub(super) struct ChainOutput {
    id: String,
    incremental: bool,
    prompt_tokens: u64,
    finished: Generated,
    current: Generated,
    prompt_meta: Option<Map<String, Value>>,
    leg_report: Map<String, Value>,
}

impl ChainOutput {
    pub(super) fn new(id: String, incremental: bool, prompt_tokens: u64) -> Self {
        Self {
            id,
            incremental,
            prompt_tokens,
            finished: Generated::default(),
            current: Generated::default(),
            prompt_meta: None,
            leg_report: Map::new(),
        }
    }

    /// The chain's frame of a Ricky leg's `frame`, which `end` ends, if any;
    /// a leg that calls Sandy shows no finish reason.
    pub(super) fn leg_frame(&mut self, mut frame: Value, end: Option<&LegEnd>) -> Value {
        self.record(&frame);
        let Value::Object(fields) = &mut frame else {
            return frame;
        };
        let mut meta = match fields.remove("meta_info") {
            Some(Value::Object(meta)) => meta,
            _ => Map::new(),
        };
        if !self.incremental {
            self.finished.prepend_to(fields, &mut meta);
        }
        for (count, finished) in SUMMED_COUNTS.iter().zip(self.finished.counts) {
            if let Some(value) = meta.get(*count).and_then(Value::as_u64) {
                meta.insert((*count).into(), (value + finished).into());
            }
        }
        self.add_prompt_fields(&mut meta);
        meta.insert("id".into(), self.id.clone().into());
        if matches!(end, Some(LegEnd::CallsSandy { .. })) {
            meta.insert("finish_reason".into(), Value::Null);
        }
        fields.insert("meta_info".into(), Value::Object(meta));
        frame
    }

    /// The current leg ended for a Sandy call: it joins the answer.
    ///
    /// Returns: the leg's token ids and text.
    pub(super) fn finish_leg(&mut self) -> (Vec<u32>, String) {
        let current = std::mem::take(&mut self.current);
        let leg = (current.output_ids.clone(), current.text.clone());
        self.finished.extend(current);
        leg
    }

    /// A Sandy call wrote `text`, `ids` in Ricky's ids, which joins the answer;
    /// `finish_reason` is the answer's when the call ended it, else null.
    ///
    /// Returns: the chain's frame of the text.
    pub(super) fn add_sandy_text(
        &mut self,
        text: &str,
        ids: &[u32],
        finish_reason: &Value,
    ) -> Value {
        let written = Generated::of_sandy_text(text, ids);
        let mut frame = if self.incremental {
            json!({"text": written.text, "output_ids": written.output_ids})
        } else {
            Value::Null
        };
        self.finished.extend(written);
        if frame.is_null() {
            frame = json!({"text": self.finished.text, "output_ids": self.finished.output_ids});
        }
        frame["meta_info"] = self.answer_meta(finish_reason);
        frame
    }

    /// How the current leg went on its engine, as its last frame said; timings
    /// need the engine's `--enable-metrics`.
    pub(super) fn leg_report(&self) -> &Map<String, Value> {
        &self.leg_report
    }

    fn answer_meta(&self, finish_reason: &Value) -> Value {
        let mut meta = Map::new();
        self.add_prompt_fields(&mut meta);
        meta.insert("id".into(), self.id.clone().into());
        meta.insert("finish_reason".into(), finish_reason.clone());
        for (count, finished) in SUMMED_COUNTS.iter().zip(self.finished.counts) {
            meta.insert((*count).into(), finished.into());
        }
        Value::Object(meta)
    }

    fn add_prompt_fields(&self, meta: &mut Map<String, Value>) {
        if let Some(prompt) = &self.prompt_meta {
            for field in PROMPT_FIELDS {
                match prompt.get(field) {
                    Some(value) => meta.insert(field.into(), value.clone()),
                    None => meta.remove(field),
                };
            }
        }
        meta.insert("prompt_tokens".into(), self.prompt_tokens.into());
        if let Some(cached) = meta.get("cached_tokens").and_then(Value::as_u64) {
            meta.insert(
                "cached_tokens".into(),
                cached.min(self.prompt_tokens).into(),
            );
        }
    }

    fn record(&mut self, frame: &Value) {
        let meta = &frame["meta_info"];
        self.leg_report = fields_of(meta, &LEG_REPORT_FIELDS);
        if self.prompt_meta.is_none() {
            self.prompt_meta = Some(fields_of(meta, &PROMPT_FIELDS));
        }
        if self.incremental {
            self.current.extend_by_frame(frame);
            self.current.counts = SUMMED_COUNTS.map(|count| meta[count].as_u64().unwrap_or(0));
        } else {
            self.current = Generated::of_frame(frame);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(text: &str, ids: &[u32], completion: u64, finish: Value) -> Value {
        json!({
            "text": text,
            "output_ids": ids,
            "meta_info": {
                "id": "chain-1",
                "prompt_tokens": 9,
                "completion_tokens": completion,
                "output_token_logprobs": ids.iter().map(|id| json!([-0.5, id, null])).collect::<Vec<_>>(),
                "finish_reason": finish,
            },
        })
    }

    const CALLS_SANDY: LegEnd = LegEnd::CallsSandy { next_block: 2 };

    fn sandy_call() -> Value {
        json!({"type": "franky_sandy_call", "next_block": 2})
    }

    #[test]
    fn a_later_whole_frame_continues_the_finished_legs() {
        let mut output = ChainOutput::new("chain".into(), false, 9);
        let first = output.leg_frame(frame("ab", &[1, 2], 2, sandy_call()), Some(&CALLS_SANDY));
        assert_eq!(first["meta_info"]["finish_reason"], Value::Null);
        assert_eq!(output.finish_leg(), (vec![1, 2], "ab".to_string()));

        let mut later = frame("c", &[3], 1, Value::Null);
        later["meta_info"]["prompt_tokens"] = json!(11);
        later["meta_info"]["cached_tokens"] = json!(10);
        let later = output.leg_frame(later, None);

        assert_eq!(later["text"], "abc");
        assert_eq!(later["output_ids"], json!([1, 2, 3]));
        assert_eq!(later["meta_info"]["completion_tokens"], 3);
        assert_eq!(later["meta_info"]["prompt_tokens"], 9);
        assert_eq!(later["meta_info"]["cached_tokens"], Value::Null);
        assert_eq!(later["meta_info"]["id"], "chain");
        assert_eq!(
            later["meta_info"]["output_token_logprobs"]
                .as_array()
                .unwrap()
                .len(),
            3
        );
    }

    #[test]
    fn the_first_legs_cache_hits_count_within_the_clients_prompt() {
        let mut output = ChainOutput::new("chain".into(), false, 9);
        output.add_sandy_text("XYZ", &[7, 8, 9], &Value::Null);
        let mut first = frame("c", &[3], 1, Value::Null);
        first["meta_info"]["prompt_tokens"] = json!(12);
        first["meta_info"]["cached_tokens"] = json!(11);

        let first = output.leg_frame(first, None);

        assert_eq!(first["meta_info"]["prompt_tokens"], 9);
        assert_eq!(first["meta_info"]["cached_tokens"], 9);
        assert_eq!(first["text"], "XYZc");
    }

    #[test]
    fn incremental_frames_stay_deltas_and_their_counts_add_up() {
        let mut output = ChainOutput::new("chain".into(), true, 9);
        output.leg_frame(frame("a", &[1], 1, Value::Null), None);
        output.leg_frame(frame("b", &[2], 2, sandy_call()), Some(&CALLS_SANDY));
        output.finish_leg();

        let delta = output.leg_frame(frame("c", &[3], 1, Value::Null), None);

        assert_eq!(delta["text"], "c");
        assert_eq!(delta["output_ids"], json!([3]));
        assert_eq!(delta["meta_info"]["completion_tokens"], 3);
    }

    #[test]
    fn a_leg_that_ends_the_answer_keeps_its_finish_reason() {
        let stop = json!({"type": "stop", "matched": 7});
        let mut output = ChainOutput::new("chain".into(), false, 9);

        let stopped =
            output.leg_frame(frame("a", &[7], 1, stop.clone()), Some(&LegEnd::AnswerEnds));

        assert_eq!(stopped["meta_info"]["finish_reason"], stop);
    }

    #[test]
    fn sandys_text_joins_the_answer_between_ricky_legs() {
        let mut output = ChainOutput::new("chain".into(), false, 9);
        output.leg_frame(frame("ab", &[1, 2], 2, sandy_call()), Some(&CALLS_SANDY));
        output.finish_leg();

        let sandy = output.add_sandy_text("XY", &[8, 9], &Value::Null);
        let whole = output.leg_frame(
            frame("c", &[3], 1, json!({"type": "length"})),
            Some(&LegEnd::AnswerEnds),
        );

        assert_eq!(sandy["text"], "abXY");
        assert_eq!(sandy["output_ids"], json!([1, 2, 8, 9]));
        assert_eq!(sandy["meta_info"]["completion_tokens"], 4);
        assert_eq!(sandy["meta_info"]["prompt_tokens"], 9);
        assert_eq!(sandy["meta_info"]["finish_reason"], Value::Null);
        assert_eq!(whole["text"], "abXYc");
        assert_eq!(whole["output_ids"], json!([1, 2, 8, 9, 3]));
        assert_eq!(whole["meta_info"]["completion_tokens"], 5);
    }

    #[test]
    fn an_incremental_stream_shows_only_sandys_new_text() {
        let mut output = ChainOutput::new("chain".into(), true, 9);
        output.add_sandy_text("ab", &[1, 2], &Value::Null);

        let sandy = output.add_sandy_text("XY", &[8, 9], &Value::Null);

        assert_eq!(sandy["text"], "XY");
        assert_eq!(sandy["output_ids"], json!([8, 9]));
        assert_eq!(sandy["meta_info"]["completion_tokens"], 4);
    }

    #[test]
    fn a_sandy_call_that_ends_the_answer_gives_the_whole_answer() {
        let limit = json!({"type": "length", "length": 2});
        let mut output = ChainOutput::new("chain".into(), false, 9);

        let whole = output.add_sandy_text("XY", &[8, 9], &limit);

        assert_eq!(whole["text"], "XY");
        assert_eq!(whole["meta_info"]["finish_reason"], limit);
        assert_eq!(whole["meta_info"]["completion_tokens"], 2);
        assert_eq!(whole["meta_info"]["prompt_tokens"], 9);
    }
}
