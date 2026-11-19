// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::server::app_context::AppContext;
use crate::server::error::ApiError;
use crate::tokenizer::franky_sandy::{FrankySandyTokens, GeneratedText};
use axum::body::Body;
use axum::http::{header, Response};
use serde_json::{json, Map, Value};
use std::sync::Arc;

/// Sandy's ids of a request whose Sandy tokenizes on its own: her ids of its
/// prompt, then her tokens of the text Ricky generates, made block by block from
/// Ricky's ids: decoded with his tokenizer, encoded with hers. A block's Sandy
/// prompt extends the last one's, so Sandy's cache keeps it, and her tokens of
/// the whole answer extend them all, so a later turn's ids can extend those.
pub(in crate::server::routes::chat) struct SandyChain {
    tokens: Arc<FrankySandyTokens>,
    text: GeneratedText,
    ids: SandyIds,
}

impl SandyChain {
    /// The chain of a request to Franky workers whose Sandy tokenizes on its
    /// own, from Sandy's ids of the request's prompt; `None` for other workers.
    ///
    /// Errors: `ApiError::Internal` when the router has no tokenizer for the
    /// served model, which decodes Ricky's tokens.
    pub(in crate::server::routes::chat) fn of_prompt(
        ctx: &AppContext,
        model: &str,
        prompt_ids: Vec<u32>,
    ) -> Result<Option<Self>, ApiError> {
        let Some(tokens) = ctx.franky_sandy.clone() else {
            return Ok(None);
        };
        let ricky = ctx.tokenizers.get(model).ok_or_else(|| {
            ApiError::Internal(anyhow::anyhow!(
                "Franky workers whose Sandy tokenizes on its own need the router's \
                 tokenizer of model {model}, which decodes Ricky's tokens for Sandy"
            ))
        })?;
        Ok(Some(Self {
            tokens,
            text: GeneratedText::new(ricky),
            ids: SandyIds::of_prompt(prompt_ids),
        }))
    }

    /// The ids Sandy reads for the next block, after her tokens of the text
    /// that `generated_ids`, every token Ricky generated so far, add up to their
    /// last whole character.
    ///
    /// Errors: `ApiError::Internal` when a tokenizer fails.
    pub(super) fn next_block(&mut self, generated_ids: &[u32]) -> Result<SandyBlock, ApiError> {
        let new_text = self.new_text(generated_ids, false)?;
        let tokens = &self.tokens;
        self.ids
            .next_block(&new_text, |text| tokens.encode_answer_text(text))
            .map_err(sandy_tokens_failed)
    }

    /// Ends the current answer after Sandy's tokens of the text so far, at a
    /// boundary token; the next block starts a new answer.
    ///
    /// Errors: `ApiError::Internal` when a tokenizer fails.
    pub(super) fn end_answer(&mut self, generated_ids: &[u32]) -> Result<(), ApiError> {
        let new_text = self.new_text(generated_ids, false)?;
        let tokens = &self.tokens;
        self.ids
            .end_answer(&new_text, |text| tokens.encode_answer_text(text))
            .map_err(sandy_tokens_failed)
    }

    /// The answers of the request that ended at a boundary token, in order.
    pub(super) fn earlier_answers(&self) -> &[SandyRun] {
        &self.ids.earlier_answers
    }

    /// The response's fields for a sample to rebuild what Sandy read:
    /// `franky_sandy_output_ids`, her tokens of the whole answer, the last
    /// block's text too, which follow her prompt's ids; and
    /// `franky_sandy_block_starts`, her tokens before each block after the
    /// first, counted from her ids' start.
    ///
    /// Errors: `ApiError::Internal` when a tokenizer fails.
    pub(in crate::server::routes::chat) fn finish(
        &mut self,
        generated_ids: &[u32],
    ) -> Result<Map<String, Value>, ApiError> {
        let new_text = self.new_text(generated_ids, true)?;
        let tokens = &self.tokens;
        let (output_ids, block_starts) = self
            .ids
            .finish(&new_text, |text| tokens.encode_answer_text(text))
            .map_err(sandy_tokens_failed)?;
        let mut fields = Map::new();
        fields.insert("franky_sandy_output_ids".into(), json!(output_ids));
        fields.insert("franky_sandy_block_starts".into(), json!(block_starts));
        Ok(fields)
    }

    /// A unary `/generate` response of a request served in one block, with
    /// `franky_sandy_output_ids` and `franky_sandy_block_starts` in its
    /// `meta_info`, from the response's `output_ids`.
    ///
    /// Errors: `ApiError::Internal` for a body that is not a JSON object, or a
    /// tokenizer that fails.
    pub(in crate::server::routes::chat) async fn add_answer_to(
        mut self,
        response: Response<Body>,
    ) -> Result<Response<Body>, ApiError> {
        if !response.status().is_success() {
            return Ok(response);
        }
        let (mut parts, body) = response.into_parts();
        let bytes = axum::body::to_bytes(body, usize::MAX)
            .await
            .map_err(|error| ApiError::Internal(error.into()))?;
        let mut frame: Value = serde_json::from_slice(&bytes).map_err(|error| {
            ApiError::Internal(anyhow::Error::new(error).context("parse Ricky's response"))
        })?;
        let generated_ids: Vec<u32> = frame["output_ids"]
            .as_array()
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| id.as_u64().and_then(|id| u32::try_from(id).ok()))
                    .collect()
            })
            .unwrap_or_default();
        self.next_block(&[])?;
        let fields = self.finish(&generated_ids)?;
        let Some(meta) = frame.get_mut("meta_info").and_then(Value::as_object_mut) else {
            return Err(ApiError::Internal(anyhow::anyhow!(
                "Ricky's response has no meta_info for Sandy's ids"
            )));
        };
        meta.extend(fields);
        parts.headers.remove(header::CONTENT_LENGTH);
        Ok(Response::from_parts(parts, Body::from(frame.to_string())))
    }

    fn new_text(&mut self, generated_ids: &[u32], to_the_end: bool) -> Result<String, ApiError> {
        self.text
            .new_text(generated_ids, to_the_end)
            .map_err(sandy_tokens_failed)
    }
}

fn sandy_tokens_failed(error: anyhow::Error) -> ApiError {
    ApiError::Internal(error.context("make Sandy's tokens of Ricky's answer"))
}

/// Sandy's ids of a request: her prompt, then her tokens of every answer.
#[derive(Debug, Default, PartialEq)]
struct SandyIds {
    prompt_ids: Vec<u32>,
    answer_ids: Vec<u32>,
    /// Sandy's tokens before each block sent so far, of every answer.
    block_starts: Vec<usize>,
    /// The first of `block_starts` that belongs to the current answer.
    answer_first_block: usize,
    /// The answers that ended at a boundary token.
    earlier_answers: Vec<SandyRun>,
}

/// One answer in Sandy's ids.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct SandyRun {
    /// `[start, end)` of the answer in Sandy's ids.
    pub(super) span: [usize; 2],
    /// Sandy's tokens before each of its blocks after the first.
    pub(super) block_starts: Vec<usize>,
}

/// What one block sends a Sandy that tokenizes on its own.
#[derive(Debug, PartialEq)]
pub(super) struct SandyBlock {
    pub(super) ids: Vec<u32>,
    /// Sandy's tokens before each earlier block of the current answer.
    pub(super) earlier_starts: Vec<usize>,
}

impl SandyIds {
    fn of_prompt(prompt_ids: Vec<u32>) -> Self {
        Self {
            prompt_ids,
            ..Self::default()
        }
    }

    /// The next block's ids, after `encode`'s tokens of `new_text`, the text the
    /// finished blocks added since the last block.
    fn next_block(
        &mut self,
        new_text: &str,
        encode: impl FnOnce(&str) -> anyhow::Result<Vec<u32>>,
    ) -> anyhow::Result<SandyBlock> {
        self.add(new_text, encode)?;
        let earlier_starts = self.block_starts[self.answer_first_block..].to_vec();
        self.block_starts.push(self.len());
        Ok(SandyBlock {
            ids: self.ids(),
            earlier_starts,
        })
    }

    /// Ends the current answer after `encode`'s tokens of `new_text`.
    fn end_answer(
        &mut self,
        new_text: &str,
        encode: impl FnOnce(&str) -> anyhow::Result<Vec<u32>>,
    ) -> anyhow::Result<()> {
        self.add(new_text, encode)?;
        let Some((&start, later_starts)) =
            self.block_starts[self.answer_first_block..].split_first()
        else {
            anyhow::bail!("an answer ended before its first block");
        };
        self.earlier_answers.push(SandyRun {
            span: [start, self.len()],
            block_starts: later_starts.to_vec(),
        });
        self.answer_first_block = self.block_starts.len();
        Ok(())
    }

    /// Sandy's tokens of every answer after `encode`'s tokens of `new_text`, the
    /// rest of the last one, and her tokens before each block after the first.
    fn finish(
        &mut self,
        new_text: &str,
        encode: impl FnOnce(&str) -> anyhow::Result<Vec<u32>>,
    ) -> anyhow::Result<(Vec<u32>, Vec<usize>)> {
        self.add(new_text, encode)?;
        let later_starts = self.block_starts.get(1..).unwrap_or_default();
        Ok((self.answer_ids.clone(), later_starts.to_vec()))
    }

    fn add(
        &mut self,
        new_text: &str,
        encode: impl FnOnce(&str) -> anyhow::Result<Vec<u32>>,
    ) -> anyhow::Result<()> {
        if !new_text.is_empty() {
            self.answer_ids.extend(encode(new_text)?);
        }
        Ok(())
    }

    fn ids(&self) -> Vec<u32> {
        [self.prompt_ids.as_slice(), self.answer_ids.as_slice()].concat()
    }

    fn len(&self) -> usize {
        self.prompt_ids.len() + self.answer_ids.len()
    }
}

/// Gives Ricky's block body where each block starts in Sandy's tokens, and
/// Sandy's input ids up to this block's start, which key Ricky's taps.
pub(super) fn add_sandy_block(
    body: &mut Map<String, Value>,
    chain: &mut Value,
    sandy: &SandyBlock,
) {
    chain["sandy_block_starts"] = json!(sandy.earlier_starts);
    body.insert("franky_sandy_input_ids".into(), json!(sandy.ids));
}

/// Adds the answers a chain ended at a boundary token to the earlier answers a
/// block body lists in Sandy's tokens, after the client's own: their runs as
/// `franky_sandy_spans` and their block starts after the first as
/// `franky_sandy_block_starts`. A client's answer without block starts reads
/// before its whole run.
pub(super) fn add_sandy_earlier_answers(body: &mut Map<String, Value>, answers: &[SandyRun]) {
    if answers.is_empty() {
        return;
    }
    let client_spans = match body.remove("franky_sandy_spans") {
        Some(Value::Array(spans)) => spans,
        _ => Vec::new(),
    };
    let client_starts = match body.remove("franky_sandy_block_starts") {
        Some(Value::Array(starts)) => starts,
        _ => vec![json!([]); client_spans.len()],
    };
    let spans = client_spans
        .into_iter()
        .chain(answers.iter().map(|answer| json!(answer.span)));
    let starts = client_starts
        .into_iter()
        .chain(answers.iter().map(|answer| json!(answer.block_starts)));
    body.insert("franky_sandy_spans".into(), Value::Array(spans.collect()));
    body.insert(
        "franky_sandy_block_starts".into(),
        Value::Array(starts.collect()),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn letters(text: &str) -> anyhow::Result<Vec<u32>> {
        Ok(text.bytes().map(u32::from).collect())
    }

    #[test]
    fn each_block_extends_sandys_ids_with_its_own_tokens_of_the_new_text() {
        let mut ids = SandyIds::of_prompt(vec![1, 2, 3]);

        let first = ids.next_block("", letters).unwrap();
        let second = ids.next_block("ab", letters).unwrap();
        let third = ids.next_block("cde", letters).unwrap();

        assert_eq!(
            first,
            SandyBlock {
                ids: vec![1, 2, 3],
                earlier_starts: vec![]
            }
        );
        assert_eq!(second.ids, vec![1, 2, 3, 97, 98]);
        assert_eq!(second.earlier_starts, vec![3]);
        assert_eq!(third.ids, vec![1, 2, 3, 97, 98, 99, 100, 101]);
        assert_eq!(third.earlier_starts, vec![3, 5]);
    }

    #[test]
    fn a_block_without_new_text_starts_where_the_last_one_did() {
        let mut ids = SandyIds::of_prompt(vec![7]);
        ids.next_block("", letters).unwrap();
        ids.next_block("x", letters).unwrap();

        let next = ids.next_block("", letters).unwrap();

        assert_eq!(next.ids, vec![7, 120]);
        assert_eq!(next.earlier_starts, vec![1, 2]);
    }

    #[test]
    fn the_finished_answer_holds_its_last_blocks_text_and_the_later_blocks_starts() {
        let mut ids = SandyIds::of_prompt(vec![1, 2, 3]);
        ids.next_block("", letters).unwrap();
        ids.next_block("ab", letters).unwrap();

        let (answer, block_starts) = ids.finish("cd", letters).unwrap();

        assert_eq!(answer, vec![97, 98, 99, 100]);
        assert_eq!(block_starts, vec![5]);
    }

    #[test]
    fn an_answer_of_one_block_has_no_later_block_starts() {
        let mut ids = SandyIds::of_prompt(vec![1, 2, 3]);
        ids.next_block("", letters).unwrap();

        let (answer, block_starts) = ids.finish("ab", letters).unwrap();

        assert_eq!(answer, vec![97, 98]);
        assert_eq!(block_starts, Vec::<usize>::new());
    }

    #[test]
    fn an_answer_ended_at_a_boundary_keeps_its_run_and_the_next_starts_its_blocks_anew() {
        let mut ids = SandyIds::of_prompt(vec![1, 2]);
        ids.next_block("", letters).unwrap();
        ids.next_block("ab", letters).unwrap();
        ids.end_answer("c", letters).unwrap();

        let next = ids.next_block("", letters).unwrap();
        let after = ids.next_block("de", letters).unwrap();
        let (answer, block_starts) = ids.finish("", letters).unwrap();

        assert_eq!(
            ids.earlier_answers,
            vec![SandyRun {
                span: [2, 5],
                block_starts: vec![4],
            }]
        );
        assert_eq!(next.earlier_starts, Vec::<usize>::new());
        assert_eq!(next.ids, vec![1, 2, 97, 98, 99]);
        assert_eq!(after.earlier_starts, vec![5]);
        assert_eq!(answer, vec![97, 98, 99, 100, 101]);
        assert_eq!(block_starts, vec![4, 5, 7]);
    }

    #[test]
    fn ricky_reads_the_blocks_sandy_starts_and_sandys_input_ids() {
        let mut body = Map::new();
        let mut chain = json!({"id": "c", "block": 3});
        let block = SandyBlock {
            ids: vec![1, 2, 3, 4, 5, 6],
            earlier_starts: vec![3, 5],
        };

        add_sandy_block(&mut body, &mut chain, &block);

        assert_eq!(chain["sandy_block_starts"], json!([3, 5]));
        assert_eq!(body["franky_sandy_input_ids"], json!([1, 2, 3, 4, 5, 6]));
    }

    #[test]
    fn a_chains_earlier_answers_follow_the_clients_in_sandys_tokens() {
        let answers = [SandyRun {
            span: [20, 26],
            block_starts: vec![23],
        }];
        let mut listed = Map::new();
        listed.insert("franky_sandy_spans".into(), json!([[3, 9]]));
        listed.insert("franky_sandy_block_starts".into(), json!([[6]]));
        let mut unlisted = Map::new();
        unlisted.insert("franky_sandy_spans".into(), json!([[3, 9]]));
        let mut none = Map::new();

        add_sandy_earlier_answers(&mut listed, &answers);
        add_sandy_earlier_answers(&mut unlisted, &answers);
        add_sandy_earlier_answers(&mut none, &answers);

        assert_eq!(listed["franky_sandy_spans"], json!([[3, 9], [20, 26]]));
        assert_eq!(listed["franky_sandy_block_starts"], json!([[6], [23]]));
        assert_eq!(unlisted["franky_sandy_block_starts"], json!([[], [23]]));
        assert_eq!(none["franky_sandy_spans"], json!([[20, 26]]));
        assert_eq!(none["franky_sandy_block_starts"], json!([[23]]));
    }
}
