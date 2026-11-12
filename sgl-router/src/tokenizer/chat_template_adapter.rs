// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

mod qwen35;

pub use qwen35::Qwen35;

use crate::config::ChatTemplateAdapterKind;
use anyhow::Result;
use dynamo_tokenizers::Tokenizer;
use std::ops::Range;
use std::sync::Arc;

/// Reads a chat that the model's own template rendered, and finds the earlier answers in it.
pub trait ChatTemplateAdapter: Send + Sync {
    /// `[start, end)` runs of `input_ids` that the model generated.
    fn generated_spans(&self, input_ids: &[u32]) -> Vec<Range<usize>>;
}

/// The adapter of the chat template `kind`, with its marker tokens looked up in `tokenizer`.
///
/// Errors: `tokenizer` lacks a marker token of that template.
pub fn build(
    kind: ChatTemplateAdapterKind,
    tokenizer: &Tokenizer,
) -> Result<Arc<dyn ChatTemplateAdapter>> {
    match kind {
        ChatTemplateAdapterKind::Qwen35 => Ok(Arc::new(Qwen35::from_tokenizer(tokenizer)?)),
    }
}
