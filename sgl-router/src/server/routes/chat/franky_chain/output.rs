// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

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
/// The finish reason of a Ricky leg that ends where Sandy writes next.
const SANDY_CALL_FINISH: &str = "franky_sandy_call";

#[derive(Default)]
struct Generated {
    text: String,
    output_ids: Vec<Value>,
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

    fn of_sandy_text(text: &str, ids: Vec<Value>) -> Self {
        let num_tokens = ids.len() as u64;
        Self {
            text: text.to_owned(),
            output_ids: ids,
            token_fields: Map::new(),
            counts: [num_tokens, 0],
        }
    }

    fn extend_by_frame(&mut self, frame: &Value) {
        if let Some(text) = frame["text"].as_str() {
            self.text.push_str(text);
        }
        if let Some(ids) = frame["output_ids"].as_array() {
            self.output_ids.extend(ids.iter().cloned());
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
            ids.splice(0..0, self.output_ids.iter().cloned());
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

/// A chain's output so far: Sandy's text and each Ricky leg's `/generate`
/// frames become frames of one request that names `id`, with the client's
/// prompt and one finish reason. The prompt fields are the first leg's, with
/// its cache hits counted within the client's prompt, which a Sandy call's
/// text may follow in that leg's input.
pub(super) struct ChainOutput {
    id: String,
    incremental_streaming_output: bool,
    prompt_tokens: u64,
    finished: Generated,
    current: Generated,
    prompt_meta: Option<Map<String, Value>>,
    leg_report: Map<String, Value>,
    finish_reason: Value,
}

impl ChainOutput {
    pub(super) fn new(id: String, incremental_streaming_output: bool, prompt_tokens: u64) -> Self {
        Self {
            id,
            incremental_streaming_output,
            prompt_tokens,
            finished: Generated::default(),
            current: Generated::default(),
            prompt_meta: None,
            leg_report: Map::new(),
            finish_reason: Value::Null,
        }
    }

    pub(super) fn start_leg(&mut self) {
        self.finish_reason = Value::Null;
    }

    /// The chain's frame for the unary response of the current leg.
    pub(super) fn whole_leg_frame(&mut self, frame: Value) -> Value {
        self.chain_frame(frame, false)
    }

    /// The chain's frame for one streamed frame of the current leg.
    pub(super) fn streamed_frame(&mut self, frame: Value) -> Value {
        self.chain_frame(frame, self.incremental_streaming_output)
    }

    /// A Sandy call wrote `text`, `ids` in Ricky's ids, which joins the answer
    /// as a finished part; `finish_reason` is the answer's when the call ended
    /// it, else null.
    ///
    /// Returns: the chain's streamed frame of the text.
    pub(super) fn add_sandy_text(
        &mut self,
        text: &str,
        ids: Vec<Value>,
        finish_reason: &Value,
    ) -> Value {
        let written = Generated::of_sandy_text(text, ids);
        if !self.incremental_streaming_output {
            self.finished.extend(written);
            return self.whole_answer(finish_reason);
        }
        let mut frame = json!({"text": written.text, "output_ids": written.output_ids});
        self.finished.extend(written);
        frame["meta_info"] = self.answer_meta(finish_reason);
        frame
    }

    /// The whole answer so far as one frame that ends with `finish_reason`.
    pub(super) fn whole_answer(&self, finish_reason: &Value) -> Value {
        json!({
            "text": self.finished.text,
            "output_ids": self.finished.output_ids,
            "meta_info": self.answer_meta(finish_reason),
        })
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

    /// Whether the current leg ended where Sandy writes next.
    pub(super) fn calls_sandy(&self) -> bool {
        self.finish_reason["type"] == SANDY_CALL_FINISH
    }

    /// The block Ricky writes after the Sandy call the current leg ended for.
    pub(super) fn next_block(&self) -> Option<u64> {
        self.finish_reason["next_block"].as_u64()
    }

    /// The current leg's token ids and text.
    pub(super) fn current_leg(&self) -> (&[Value], &str) {
        (&self.current.output_ids, &self.current.text)
    }

    pub(super) fn finish_leg(&mut self) {
        let current = std::mem::take(&mut self.current);
        self.finished.extend(current);
    }

    /// How the current leg went on its engine, as its last frame said; timings
    /// need the engine's `--enable-metrics`.
    pub(super) fn leg_report(&self) -> &Map<String, Value> {
        &self.leg_report
    }

    fn chain_frame(&mut self, mut frame: Value, is_delta: bool) -> Value {
        self.record(&frame, is_delta);
        let Value::Object(fields) = &mut frame else {
            return frame;
        };
        let mut meta = match fields.remove("meta_info") {
            Some(Value::Object(meta)) => meta,
            _ => Map::new(),
        };
        if !is_delta {
            self.finished.prepend_to(fields, &mut meta);
        }
        for (count, finished) in SUMMED_COUNTS.iter().zip(self.finished.counts) {
            if let Some(value) = meta.get(*count).and_then(Value::as_u64) {
                meta.insert((*count).into(), (value + finished).into());
            }
        }
        self.add_prompt_fields(&mut meta);
        meta.insert("id".into(), self.id.clone().into());
        if self.calls_sandy() {
            meta.insert("finish_reason".into(), Value::Null);
        }
        fields.insert("meta_info".into(), Value::Object(meta));
        frame
    }

    fn record(&mut self, frame: &Value, is_delta: bool) {
        let meta = &frame["meta_info"];
        self.leg_report = fields_of(meta, &LEG_REPORT_FIELDS);
        if self.prompt_meta.is_none() {
            self.prompt_meta = Some(fields_of(meta, &PROMPT_FIELDS));
        }
        if !meta["finish_reason"].is_null() {
            self.finish_reason = meta["finish_reason"].clone();
        }
        if is_delta {
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

    fn sandy_call(next_block: u64) -> Value {
        json!({"type": "franky_sandy_call", "next_block": next_block})
    }

    #[test]
    fn a_later_cumulative_frame_continues_the_finished_legs() {
        let mut output = ChainOutput::new("chain".into(), false, 9);
        output.start_leg();
        let first = output.streamed_frame(frame("ab", &[1, 2], 2, sandy_call(2)));
        assert!(output.calls_sandy());
        assert_eq!(output.next_block(), Some(2));
        assert_eq!(first["meta_info"]["finish_reason"], Value::Null);
        output.finish_leg();

        output.start_leg();
        let mut later = frame("c", &[3], 1, Value::Null);
        later["meta_info"]["prompt_tokens"] = json!(11);
        later["meta_info"]["cached_tokens"] = json!(10);
        let later = output.streamed_frame(later);

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
        output.add_sandy_text("XYZ", vec![json!(7), json!(8), json!(9)], &Value::Null);
        let mut first = frame("c", &[3], 1, Value::Null);
        first["meta_info"]["prompt_tokens"] = json!(12);
        first["meta_info"]["cached_tokens"] = json!(11);

        let first = output.streamed_frame(first);

        assert_eq!(first["meta_info"]["prompt_tokens"], 9);
        assert_eq!(first["meta_info"]["cached_tokens"], 9);
        assert_eq!(first["text"], "XYZc");
    }

    #[test]
    fn incremental_frames_stay_deltas_and_their_counts_add_up() {
        let mut output = ChainOutput::new("chain".into(), true, 9);
        output.start_leg();
        output.streamed_frame(frame("a", &[1], 1, Value::Null));
        output.streamed_frame(frame("b", &[2], 2, sandy_call(2)));
        output.finish_leg();

        output.start_leg();
        let delta = output.streamed_frame(frame("c", &[3], 1, Value::Null));

        assert_eq!(delta["text"], "c");
        assert_eq!(delta["output_ids"], json!([3]));
        assert_eq!(delta["meta_info"]["completion_tokens"], 3);
    }

    #[test]
    fn a_stop_or_the_limit_keeps_its_finish_reason() {
        let stop = json!({"type": "stop", "matched": 7});
        let mut output = ChainOutput::new("chain".into(), false, 9);
        output.start_leg();
        let stopped = output.streamed_frame(frame("a", &[7], 1, stop.clone()));
        assert!(!output.calls_sandy());
        assert_eq!(stopped["meta_info"]["finish_reason"], stop);

        output.start_leg();
        let limit = json!({"type": "length", "length": 1});
        let last = output.streamed_frame(frame("b", &[2], 1, limit.clone()));
        assert!(!output.calls_sandy());
        assert_eq!(last["meta_info"]["finish_reason"], limit);
    }

    #[test]
    fn sandys_text_joins_the_answer_between_ricky_legs() {
        let mut output = ChainOutput::new("chain".into(), false, 9);
        output.start_leg();
        output.whole_leg_frame(frame("ab", &[1, 2], 2, sandy_call(2)));
        output.finish_leg();

        let sandy = output.add_sandy_text("XY", vec![json!(8), json!(9)], &Value::Null);
        output.start_leg();
        let whole = output.whole_leg_frame(frame("c", &[3], 1, json!({"type": "length"})));

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
        output.add_sandy_text("ab", vec![json!(1), json!(2)], &Value::Null);

        let sandy = output.add_sandy_text("XY", vec![json!(8), json!(9)], &Value::Null);

        assert_eq!(sandy["text"], "XY");
        assert_eq!(sandy["output_ids"], json!([8, 9]));
        assert_eq!(sandy["meta_info"]["completion_tokens"], 4);
    }

    #[test]
    fn a_sandy_call_that_ends_the_answer_makes_the_whole_answer() {
        let limit = json!({"type": "length", "length": 2});
        let mut output = ChainOutput::new("chain".into(), true, 9);

        output.add_sandy_text("XY", vec![json!(8), json!(9)], &limit);

        let whole = output.whole_answer(&limit);
        assert_eq!(whole["text"], "XY");
        assert_eq!(whole["meta_info"]["finish_reason"], limit);
        assert_eq!(whole["meta_info"]["completion_tokens"], 2);
        assert_eq!(whole["meta_info"]["prompt_tokens"], 9);
    }
}
