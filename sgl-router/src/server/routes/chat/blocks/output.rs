// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::server::error::ApiError;
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use serde_json::{Map, Value};

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
/// Per-token rows, base64 of fixed-size records, that each block returns for
/// its own tokens only and the chain joins in order.
const JOINED_ROWS: [&str; 2] = ["routed_experts", "franky_sandy_routed_experts"];
const BLOCK_REPORT_FIELDS: [&str; 7] = [
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
    output_ids: Vec<Value>,
    token_fields: Map<String, Value>,
    counts: [u64; SUMMED_COUNTS.len()],
    /// Per field of `JOINED_ROWS`, each block's rows, base64.
    rows: [Vec<String>; JOINED_ROWS.len()],
}

impl Generated {
    fn of_frame(frame: &Value) -> Self {
        let mut generated = Self::default();
        generated.extend_by_frame(frame);
        generated.counts =
            SUMMED_COUNTS.map(|count| frame["meta_info"][count].as_u64().unwrap_or(0));
        generated
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
        for (field, rows) in JOINED_ROWS.iter().zip(self.rows.iter_mut()) {
            if let Some(block_rows) = frame["meta_info"][field].as_str() {
                rows.push(block_rows.to_owned());
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
        for (rows, other) in self.rows.iter_mut().zip(other.rows) {
            rows.extend(other);
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

fn token_ids(ids: &[Value]) -> Vec<u32> {
    ids.iter()
        .filter_map(|id| id.as_u64().and_then(|id| u32::try_from(id).ok()))
        .collect()
}

/// The rows of every block, in order, as one base64 string, as one request
/// would have returned them.
///
/// Errors: `ApiError::Internal` for a block's rows that are not base64.
fn joined_rows<'a>(blocks: impl Iterator<Item = &'a String>) -> Result<String, ApiError> {
    let mut bytes = Vec::new();
    for block in blocks {
        let rows = STANDARD.decode(block).map_err(|error| {
            ApiError::Internal(anyhow::Error::new(error).context("decode a Franky block's rows"))
        })?;
        bytes.extend(rows);
    }
    Ok(STANDARD.encode(bytes))
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

/// A chain's output so far: each block's `/generate` frames become frames of one
/// request that names `id`, with the first block's prompt and one finish reason.
pub(super) struct ChainOutput {
    id: String,
    incremental_streaming_output: bool,
    finished: Generated,
    current: Generated,
    prompt_meta: Option<Map<String, Value>>,
    block_report: Map<String, Value>,
    last_block: bool,
    finish_reason: Value,
    /// The token that ends an answer at a block's end, when the client does
    /// not stop on it.
    boundary: Option<u32>,
}

impl ChainOutput {
    pub(super) fn new(
        id: String,
        incremental_streaming_output: bool,
        boundary: Option<u32>,
    ) -> Self {
        Self {
            id,
            incremental_streaming_output,
            boundary,
            finished: Generated::default(),
            current: Generated::default(),
            prompt_meta: None,
            block_report: Map::new(),
            last_block: false,
            finish_reason: Value::Null,
        }
    }

    pub(super) fn start_block(&mut self, last_block: bool) {
        self.last_block = last_block;
        self.finish_reason = Value::Null;
    }

    /// The chain's frame for the unary response of the current block.
    ///
    /// Errors: `ApiError::Internal` for a block's rows that are not base64.
    pub(super) fn whole_block_frame(&mut self, frame: Value) -> Result<Value, ApiError> {
        self.chain_frame(frame, false)
    }

    /// The chain's frame for one streamed frame of the current block.
    ///
    /// Errors: `ApiError::Internal` for a block's rows that are not base64.
    pub(super) fn streamed_frame(&mut self, frame: Value) -> Result<Value, ApiError> {
        self.chain_frame(frame, self.incremental_streaming_output)
    }

    /// Whether the current block ended at its own token limit or at the
    /// boundary token, so the chain goes on with another block.
    pub(super) fn block_cut(&self) -> bool {
        !self.last_block && (self.finish_reason["type"] == "length" || self.ended_on_boundary())
    }

    /// Whether the current block stopped on the boundary token, so the next
    /// block starts a new answer.
    pub(super) fn ended_on_boundary(&self) -> bool {
        self.finish_reason["type"] == "stop"
            && self.boundary.is_some_and(|boundary| {
                self.finish_reason["matched"].as_u64() == Some(u64::from(boundary))
            })
    }

    /// Whether the current block is the chain's last one: no engine holds the
    /// chain's prefix after it.
    pub(super) fn on_last_block(&self) -> bool {
        self.last_block
    }

    /// Whether the last frame ended the chain: it carries a finish reason that
    /// cuts no block.
    pub(super) fn ends_chain(&self) -> bool {
        !self.finish_reason.is_null() && !self.block_cut()
    }

    pub(super) fn finish_block(&mut self) {
        let current = std::mem::take(&mut self.current);
        self.finished.extend(current);
    }

    /// How the current block went on its engine, as its last frame said; timings
    /// need the engine's `--enable-metrics`.
    pub(super) fn block_report(&self) -> &Map<String, Value> {
        &self.block_report
    }

    /// The token ids every finished block generated, in order.
    pub(super) fn generated_ids(&self) -> &[Value] {
        &self.finished.output_ids
    }

    /// The token ids every finished block generated, in order.
    pub(super) fn generated_token_ids(&self) -> Vec<u32> {
        token_ids(&self.finished.output_ids)
    }

    /// The token ids every block generated, the current one's so far too.
    pub(super) fn all_generated_token_ids(&self) -> Vec<u32> {
        let mut ids = self.generated_token_ids();
        ids.extend(token_ids(&self.current.output_ids));
        ids
    }

    fn chain_frame(&mut self, mut frame: Value, is_delta: bool) -> Result<Value, ApiError> {
        self.record(&frame, is_delta);
        let Value::Object(fields) = &mut frame else {
            return Ok(frame);
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
        if let Some(prompt) = &self.prompt_meta {
            for field in PROMPT_FIELDS {
                match prompt.get(field) {
                    Some(value) => meta.insert(field.into(), value.clone()),
                    None => meta.remove(field),
                };
            }
        }
        meta.insert("id".into(), self.id.clone().into());
        if self.block_cut() {
            meta.insert("finish_reason".into(), Value::Null);
        }
        for (index, field) in JOINED_ROWS.iter().enumerate() {
            if !meta.contains_key(*field) {
                continue;
            }
            // A block's frame carries its own rows; the chain's last carries all.
            if self.block_cut() {
                meta.remove(*field);
                continue;
            }
            let blocks = self.finished.rows[index].iter();
            let rows = joined_rows(blocks.chain(&self.current.rows[index]))?;
            meta.insert((*field).into(), rows.into());
        }
        fields.insert("meta_info".into(), Value::Object(meta));
        Ok(frame)
    }

    fn record(&mut self, frame: &Value, is_delta: bool) {
        let meta = &frame["meta_info"];
        self.block_report = fields_of(meta, &BLOCK_REPORT_FIELDS);
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
    use serde_json::json;

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

    fn length(n: u64) -> Value {
        json!({"type": "length", "length": n})
    }

    #[test]
    fn a_later_cumulative_frame_continues_the_finished_blocks() {
        let mut output = ChainOutput::new("chain".into(), false, None);
        output.start_block(false);
        let first = output
            .streamed_frame(frame("ab", &[1, 2], 2, length(2)))
            .unwrap();
        assert!(output.block_cut());
        assert_eq!(first["meta_info"]["finish_reason"], Value::Null);
        output.finish_block();

        output.start_block(true);
        let mut later = frame("c", &[3], 1, Value::Null);
        later["meta_info"]["prompt_tokens"] = json!(11);
        let later = output.streamed_frame(later).unwrap();

        assert_eq!(later["text"], "abc");
        assert_eq!(later["output_ids"], json!([1, 2, 3]));
        assert_eq!(later["meta_info"]["completion_tokens"], 3);
        assert_eq!(later["meta_info"]["prompt_tokens"], 9);
        assert_eq!(later["meta_info"]["id"], "chain");
        assert_eq!(
            later["meta_info"]["output_token_logprobs"]
                .as_array()
                .unwrap()
                .len(),
            3
        );
        assert_eq!(output.generated_ids(), &[json!(1), json!(2)]);
    }

    #[test]
    fn incremental_frames_stay_deltas_and_their_counts_add_up() {
        let mut output = ChainOutput::new("chain".into(), true, None);
        output.start_block(false);
        output
            .streamed_frame(frame("a", &[1], 1, Value::Null))
            .unwrap();
        output
            .streamed_frame(frame("b", &[2], 2, length(2)))
            .unwrap();
        output.finish_block();

        output.start_block(false);
        let delta = output
            .streamed_frame(frame("c", &[3], 1, Value::Null))
            .unwrap();

        assert_eq!(delta["text"], "c");
        assert_eq!(delta["output_ids"], json!([3]));
        assert_eq!(delta["meta_info"]["completion_tokens"], 3);
        assert_eq!(output.generated_ids(), &[json!(1), json!(2)]);
    }

    #[test]
    fn the_last_block_and_a_stop_keep_their_finish_reason() {
        let stop = json!({"type": "stop", "matched": 7});
        let mut output = ChainOutput::new("chain".into(), false, None);
        output.start_block(false);
        let stopped = output
            .streamed_frame(frame("a", &[7], 1, stop.clone()))
            .unwrap();
        assert!(!output.block_cut());
        assert_eq!(stopped["meta_info"]["finish_reason"], stop);

        output.start_block(true);
        let last = output
            .streamed_frame(frame("b", &[2], 1, length(1)))
            .unwrap();
        assert!(!output.block_cut());
        assert_eq!(last["meta_info"]["finish_reason"], length(1));
    }

    #[test]
    fn a_unary_block_is_one_whole_frame_even_with_incremental_streams() {
        let mut output = ChainOutput::new("chain".into(), true, None);
        output.start_block(false);
        output
            .whole_block_frame(frame("ab", &[1, 2], 2, length(2)))
            .unwrap();
        output.finish_block();
        output.start_block(true);

        let whole = output
            .whole_block_frame(frame("c", &[3], 1, length(1)))
            .unwrap();

        assert_eq!(whole["text"], "abc");
        assert_eq!(whole["output_ids"], json!([1, 2, 3]));
    }

    #[test]
    fn each_blocks_rows_join_in_the_chains_last_frame() {
        let rows = |bytes: &[u8]| STANDARD.encode(bytes);
        let mut output = ChainOutput::new("chain".into(), false, None);
        output.start_block(false);
        let mut first = frame("ab", &[1, 2], 2, length(2));
        first["meta_info"]["routed_experts"] = json!(rows(&[1, 2, 3, 4]));
        first["meta_info"]["franky_sandy_routed_experts"] = json!(rows(&[9]));
        let cut = output.whole_block_frame(first).unwrap();
        output.finish_block();
        output.start_block(true);
        let mut last = frame("c", &[3], 1, length(1));
        last["meta_info"]["routed_experts"] = json!(rows(&[5, 6, 7, 8]));
        last["meta_info"]["franky_sandy_routed_experts"] = json!(rows(&[10, 11]));

        let whole = output.whole_block_frame(last).unwrap();

        assert!(cut["meta_info"].get("routed_experts").is_none());
        assert!(cut["meta_info"]
            .get("franky_sandy_routed_experts")
            .is_none());
        assert_eq!(
            whole["meta_info"]["routed_experts"],
            json!(rows(&[1, 2, 3, 4, 5, 6, 7, 8]))
        );
        assert_eq!(
            whole["meta_info"]["franky_sandy_routed_experts"],
            json!(rows(&[9, 10, 11]))
        );
    }

    #[test]
    fn rows_that_are_not_base64_fail_the_chain() {
        let mut output = ChainOutput::new("chain".into(), false, None);
        output.start_block(true);
        let mut last = frame("c", &[3], 1, length(1));
        last["meta_info"]["routed_experts"] = json!("not base64!");

        assert!(output.whole_block_frame(last).is_err());
    }

    #[test]
    fn a_stop_on_the_boundary_token_cuts_the_block_and_clears_the_finish_reason() {
        let mut output = ChainOutput::new("chain".into(), false, Some(7));
        output.start_block(false);
        let boundary = json!({"type": "stop", "matched": 7});
        let cut = output
            .whole_block_frame(frame("<3>", &[3, 7], 2, boundary))
            .unwrap();

        assert!(output.block_cut());
        assert!(output.ended_on_boundary());
        assert_eq!(cut["meta_info"]["finish_reason"], Value::Null);

        output.start_block(true);
        let last = output
            .whole_block_frame(frame(
                "<8>",
                &[8, 7],
                2,
                json!({"type": "stop", "matched": 7}),
            ))
            .unwrap();
        assert!(!output.block_cut());
        assert_eq!(
            last["meta_info"]["finish_reason"],
            json!({"type": "stop", "matched": 7})
        );
    }
}
