// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use super::adapter;
use super::chat_formatter::ChatFormatter;
use super::chat_template_adapter::{self, ChatTemplateAdapter};
use super::stats::EncodeBackend;
use crate::config::{FrankySandyConfig, TokenizerConfig};
use anyhow::{Context, Result};
use dynamo_tokenizers::Tokenizer;
use serde_json::Value;
use std::num::NonZeroUsize;
use std::ops::Range;
use std::sync::Arc;

/// A Franky Sandy that tokenizes on its own: its tokenizer, chat template and
/// the markers of its earlier answers, kept apart from the served model's.
pub struct FrankySandyTokens {
    tokenizer: Arc<Tokenizer>,
    backend: EncodeBackend,
    formatter: ChatFormatter,
    adapter: Arc<dyn ChatTemplateAdapter>,
    keep_every: NonZeroUsize,
}

/// A chat as Sandy reads it.
#[derive(Debug, PartialEq)]
pub struct SandyChat {
    pub input_ids: Vec<u32>,
    /// The runs of the chat's earlier answers in `input_ids`.
    pub generated_spans: Vec<Range<usize>>,
}

impl SandyChat {
    /// The chat with its earlier answers whole and, outside them, only every
    /// `keep_every`-th token, counted from the rendering's start so that a
    /// chat's prefix keeps a prefix of its tokens for Sandy's cache.
    fn keep_every(self, keep_every: NonZeroUsize) -> Self {
        let keep_every = keep_every.get();
        let input_ids = &self.input_ids;
        let kept_between = move |between: Range<usize>| {
            between
                .filter(move |index| index % keep_every == keep_every - 1)
                .map(move |index| input_ids[index])
        };
        let mut kept_ids = Vec::new();
        let mut kept_spans = Vec::with_capacity(self.generated_spans.len());
        let mut done = 0;
        for span in &self.generated_spans {
            kept_ids.extend(kept_between(done..span.start));
            let start = kept_ids.len();
            kept_ids.extend_from_slice(&input_ids[span.clone()]);
            kept_spans.push(start..kept_ids.len());
            done = span.end;
        }
        kept_ids.extend(kept_between(done..input_ids.len()));
        Self {
            input_ids: kept_ids,
            generated_spans: kept_spans,
        }
    }
}

impl FrankySandyTokens {
    /// Errors: the tokenizer does not load, Sandy's model files have no chat
    /// template the router renders, or the tokenizer lacks the template's markers.
    pub fn load(config: &FrankySandyConfig, tokenizer_config: TokenizerConfig) -> Result<Self> {
        let path = &config.tokenizer_path;
        let (tokenizer, stats) = adapter::load_with(path, tokenizer_config)
            .with_context(|| format!("load Sandy's tokenizer from {path}"))?;
        let formatter = ChatFormatter::load_from(path, &adapter::ModelFiles::open(path))?
            .with_context(|| {
                format!("Sandy's model files {path} have no chat template the router renders")
            })?;
        let adapter = chat_template_adapter::build(config.chat_adapter, &tokenizer)
            .context("build --franky-sandy-chat-adapter")?;
        Ok(Self {
            tokenizer,
            backend: stats.backend(),
            formatter,
            adapter,
            keep_every: config.keep_every,
        })
    }

    /// Sandy's tokens of `text` inside an answer it already holds the start of:
    /// plain text, which cannot name special tokens.
    ///
    /// Errors: the tokenizer fails.
    pub fn encode_answer_text(&self, text: &str) -> Result<Vec<u32>> {
        adapter::encode_answer_text(&self.tokenizer, self.backend, text)
    }

    /// Errors: Sandy's template does not render `chat`.
    pub fn render(&self, chat: &Value) -> Result<SandyChat> {
        let input_ids = self.formatter.encode(&self.tokenizer, chat)?;
        let generated_spans = self.adapter.generated_spans(&input_ids);
        let rendered = SandyChat {
            input_ids,
            generated_spans,
        };
        Ok(rendered.keep_every(self.keep_every))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chat(length: u32, generated_spans: &[Range<usize>]) -> SandyChat {
        SandyChat {
            input_ids: (100..100 + length).collect(),
            generated_spans: generated_spans.to_vec(),
        }
    }

    fn every(keep_every: usize) -> NonZeroUsize {
        NonZeroUsize::new(keep_every).unwrap()
    }

    #[test]
    fn keeping_every_token_changes_nothing() {
        let kept = chat(10, &[2..5, 7..9]).keep_every(every(1));

        assert_eq!(kept, chat(10, &[2..5, 7..9]));
    }

    #[test]
    fn a_chat_without_answers_keeps_every_kth_token() {
        let kept = chat(10, &[]).keep_every(every(3));

        assert_eq!(
            kept,
            SandyChat {
                input_ids: vec![102, 105, 108],
                generated_spans: vec![],
            }
        );
    }

    #[test]
    fn earlier_answers_stay_whole_and_move_to_their_place_in_the_kept_tokens() {
        let kept = chat(16, &[3..5, 9..13]).keep_every(every(4));

        assert_eq!(
            kept,
            SandyChat {
                input_ids: vec![103, 104, 107, 109, 110, 111, 112, 115],
                generated_spans: vec![0..2, 3..7],
            }
        );
    }

    #[test]
    fn a_prefix_of_a_chat_keeps_a_prefix_of_its_tokens() {
        let whole = chat(30, &[10..14, 21..25]).keep_every(every(4));
        let first_answer = [Range { start: 10, end: 14 }];
        let prefixes = [
            chat(9, &[]),
            chat(14, &first_answer),
            chat(19, &first_answer),
        ];

        for prefix in prefixes {
            let kept = prefix.keep_every(every(4));

            assert!(whole.input_ids.starts_with(&kept.input_ids), "{kept:?}");
            assert!(
                whole.generated_spans.starts_with(&kept.generated_spans),
                "{kept:?}"
            );
        }
    }
}
