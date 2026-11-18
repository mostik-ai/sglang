// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use super::adapter;
use super::chat_formatter::ChatFormatter;
use super::chat_template_adapter::{self, ChatTemplateAdapter};
use crate::config::{FrankySandyConfig, TokenizerConfig};
use anyhow::{Context, Result};
use dynamo_tokenizers::{EncodeSegment, Tokenizer};
use serde_json::Value;
use std::ops::Range;
use std::sync::Arc;

/// A Franky Sandy that tokenizes on its own: its tokenizer, chat template and
/// the markers of its earlier answers, kept apart from the served model's.
pub struct FrankySandyTokens {
    tokenizer: Arc<Tokenizer>,
    formatter: ChatFormatter,
    adapter: Arc<dyn ChatTemplateAdapter>,
}

/// A chat as Sandy reads it.
pub struct SandyChat {
    pub input_ids: Vec<u32>,
    /// The runs of the chat's earlier answers in `input_ids`.
    pub generated_spans: Vec<Range<usize>>,
}

impl FrankySandyTokens {
    /// Errors: the tokenizer does not load, Sandy's model files have no chat
    /// template the router renders, or the tokenizer lacks the template's markers.
    pub fn load(config: &FrankySandyConfig, tokenizer_config: TokenizerConfig) -> Result<Self> {
        let path = &config.tokenizer_path;
        let (tokenizer, _) = adapter::load_with(path, tokenizer_config)
            .with_context(|| format!("load Sandy's tokenizer from {path}"))?;
        let formatter = ChatFormatter::load_from(path, &adapter::ModelFiles::open(path))?
            .with_context(|| {
                format!("Sandy's model files {path} have no chat template the router renders")
            })?;
        let adapter = chat_template_adapter::build(config.chat_adapter, &tokenizer)
            .context("build --franky-sandy-chat-adapter")?;
        Ok(Self {
            tokenizer,
            formatter,
            adapter,
        })
    }

    /// Sandy's tokens of `text` inside an answer it already holds the start of:
    /// plain text, which cannot name special tokens.
    ///
    /// Errors: the tokenizer fails.
    pub fn encode_answer_text(&self, text: &str) -> Result<Vec<u32>> {
        let segment = [EncodeSegment::new(text, false)];
        if self.formatter.is_kimi_k3() {
            return super::kimi::encode(&self.tokenizer, &segment);
        }
        Ok(self
            .tokenizer
            .encode_segments(&segment)?
            .token_ids()
            .to_vec())
    }

    /// Errors: Sandy's template does not render `chat`.
    pub fn render(&self, chat: &Value) -> Result<SandyChat> {
        let input_ids = self.formatter.encode(&self.tokenizer, chat)?;
        let generated_spans = self.adapter.generated_spans(&input_ids);
        Ok(SandyChat {
            input_ids,
            generated_spans,
        })
    }
}
