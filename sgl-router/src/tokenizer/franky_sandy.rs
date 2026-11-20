// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use super::adapter;
use super::chat_formatter::ChatFormatter;
use super::chat_template_adapter::{self, ChatTemplateAdapter};
use crate::config::{FrankySandyConfig, TokenizerConfig};
use anyhow::{Context, Result};
use dynamo_tokenizers::{EncodeSegment, Tokenizer};
use serde_json::Value;
use std::num::NonZeroUsize;
use std::ops::Range;
use std::sync::Arc;

/// A Franky Sandy that tokenizes on its own: its tokenizer, chat template and
/// the markers of its earlier answers, kept apart from the served model's.
pub struct FrankySandyTokens {
    tokenizer: Arc<Tokenizer>,
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
            keep_every: config.keep_every,
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
        let rendered = SandyChat {
            input_ids,
            generated_spans,
        };
        Ok(rendered.keep_every(self.keep_every))
    }
}

/// What a lossy decode of ids that end inside a character ends with.
const INCOMPLETE_CHARACTER: char = '\u{FFFD}';

/// Ricky's text of an answer as his ids of it arrive, decoded with special
/// tokens kept, so that his end token becomes text too. Each call gives the
/// text his ids of the answer so far add: up to a character they end inside
/// of, which waits for the call that finishes it, or with `to_the_end` all of
/// it.
pub struct AnswerText {
    tokenizer: Arc<Tokenizer>,
    /// The text the calls so far gave.
    done: String,
}

impl AnswerText {
    /// `tokenizer`: Ricky's.
    pub fn new(tokenizer: Arc<Tokenizer>) -> Self {
        Self {
            tokenizer,
            done: String::new(),
        }
    }

    /// The text that `answer_ids`, Ricky's ids of the answer so far, add since
    /// the last call: their whole text without one trailing U+FFFD, the
    /// character they end inside of, or with `to_the_end` all of it.
    ///
    /// Errors: the tokenizer fails, or his text of more ids does not continue
    /// his text of fewer.
    pub fn new_text(&mut self, answer_ids: &[u32], to_the_end: bool) -> Result<String> {
        let mut text = adapter::decode_complete(&self.tokenizer, answer_ids, false)?;
        if !to_the_end && text.ends_with(INCOMPLETE_CHARACTER) {
            text.pop();
        }
        let new_text = text
            .strip_prefix(self.done.as_str())
            .with_context(|| {
                format!(
                    "Ricky's text of more tokens, {text:?}, does not continue his text of \
                     fewer, {:?}",
                    self.done
                )
            })?
            .to_owned();
        self.done = text;
        Ok(new_text)
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

    fn tiny_tokenizer() -> Arc<Tokenizer> {
        adapter::load("tests/fixtures/tiny_tokenizer.json").unwrap()
    }

    #[test]
    fn a_character_split_between_blocks_waits_for_the_tokens_that_finish_it() {
        let tokenizer = tiny_tokenizer();
        let ids = adapter::encode(&tokenizer, "a\u{1F600}b").unwrap();
        // The fixture has no merges: one token per byte.
        assert_eq!(ids.len(), 6);
        let mut text = AnswerText::new(tokenizer);

        assert_eq!(text.new_text(&ids[..3], false).unwrap(), "a");
        assert_eq!(text.new_text(&ids[..4], false).unwrap(), "");
        assert_eq!(text.new_text(&ids[..5], false).unwrap(), "\u{1F600}");
        assert_eq!(text.new_text(&ids, false).unwrap(), "b");
        assert_eq!(text.new_text(&ids, true).unwrap(), "");
    }

    #[test]
    fn the_end_of_the_answer_gives_an_unfinished_character_too() {
        let tokenizer = tiny_tokenizer();
        let ids = adapter::encode(&tokenizer, "a\u{1F600}").unwrap();
        let mut text = AnswerText::new(tokenizer);

        assert_eq!(text.new_text(&ids[..3], true).unwrap(), "a\u{FFFD}");
    }

    #[test]
    fn special_tokens_stay_in_the_text() {
        let tokenizer = tiny_tokenizer();
        let mut ids = adapter::encode(&tokenizer, "ab").unwrap();
        // The fixture's one special token.
        ids.push(256);
        let mut text = AnswerText::new(tokenizer);

        assert_eq!(text.new_text(&ids, true).unwrap(), "ab<|endoftext|>");
    }

    #[test]
    fn texts_given_block_by_block_join_into_the_whole_text() {
        let tokenizer = tiny_tokenizer();
        let whole = "Hello, w\u{00F6}rld \u{1F600}! ".repeat(20);
        let ids = adapter::encode(&tokenizer, &whole).unwrap();
        let mut text = AnswerText::new(tokenizer);

        let mut joined = String::new();
        for end in (0..ids.len()).step_by(7) {
            joined.push_str(&text.new_text(&ids[..end], false).unwrap());
        }
        joined.push_str(&text.new_text(&ids, true).unwrap());

        assert_eq!(joined, whole);
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
