// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

mod marked_answers;

pub use marked_answers::MarkedAnswers;

use crate::config::ChatTemplateAdapterKind;
use anyhow::Result;
use dynamo_tokenizers::Tokenizer;
use std::ops::Range;
use std::sync::Arc;

/// Reads a chat that the model's own template rendered, and finds the earlier answers in it.
pub trait ChatTemplateAdapter: Send + Sync {
    /// `[start, end)` runs of `input_ids` that the model generated.
    fn generated_spans(&self, input_ids: &[u32]) -> Vec<Range<usize>>;

    /// The token a model writes where its answer's text ends and the template
    /// starts closing the answer before the model's EOS; `None` for a template
    /// whose answer's text ends at the EOS.
    fn text_end(&self) -> Option<u32> {
        None
    }
}

/// The adapter of the chat template `kind`, with its marker tokens looked up in `tokenizer`.
///
/// Errors: `tokenizer` lacks a marker token of that template.
pub fn build(
    kind: ChatTemplateAdapterKind,
    tokenizer: &Tokenizer,
) -> Result<Arc<dyn ChatTemplateAdapter>> {
    let adapter = match kind {
        ChatTemplateAdapterKind::Qwen35 => MarkedAnswers::qwen35(tokenizer)?,
        ChatTemplateAdapterKind::DeepseekV4 => MarkedAnswers::deepseek_v4(tokenizer)?,
        ChatTemplateAdapterKind::KimiK3 => MarkedAnswers::kimi_k3(tokenizer)?,
    };
    Ok(Arc::new(adapter))
}
