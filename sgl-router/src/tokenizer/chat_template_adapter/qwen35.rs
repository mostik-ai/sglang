// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use super::ChatTemplateAdapter;
use crate::tokenizer::adapter::encode;
use anyhow::{bail, Result};
use dynamo_tokenizers::Tokenizer;
use std::ops::Range;

/// Qwen3.5's chat template. An answer starts after `<|im_start|>assistant\n` and
/// the `<think>\n\n</think>\n\n` or `<think>\n` that may follow it, and ends
/// with its `<|im_end|>`.
pub struct Qwen35 {
    answer_header: Vec<u32>,
    empty_reasoning: Vec<u32>,
    reasoning_opening: Vec<u32>,
    answer_end: u32,
}

impl Qwen35 {
    /// Looks up the template's markers in `tokenizer`.
    ///
    /// Errors: `tokenizer` has no single token for `<|im_start|>`, `<|im_end|>`,
    /// `<think>` or `</think>`.
    pub fn from_tokenizer(tokenizer: &Tokenizer) -> Result<Self> {
        for marker in ["<|im_start|>", "<think>", "</think>"] {
            single_token(tokenizer, marker)?;
        }
        Ok(Self {
            answer_header: encode(tokenizer, "<|im_start|>assistant\n")?,
            empty_reasoning: encode(tokenizer, "<think>\n\n</think>\n\n")?,
            reasoning_opening: encode(tokenizer, "<think>\n")?,
            answer_end: single_token(tokenizer, "<|im_end|>")?,
        })
    }

    fn answer_start(&self, input_ids: &[u32], header_end: usize) -> usize {
        let after_header = &input_ids[header_end..];
        let opening = [&self.empty_reasoning, &self.reasoning_opening]
            .into_iter()
            .find(|opening| after_header.starts_with(opening));
        header_end + opening.map_or(0, Vec::len)
    }
}

impl ChatTemplateAdapter for Qwen35 {
    fn generated_spans(&self, input_ids: &[u32]) -> Vec<Range<usize>> {
        let mut spans = Vec::new();
        let mut position = 0;
        while let Some(header) = find(&input_ids[position..], &self.answer_header) {
            let start = self.answer_start(input_ids, position + header + self.answer_header.len());
            let Some(length) = input_ids[start..]
                .iter()
                .position(|&id| id == self.answer_end)
            else {
                break;
            };
            let end = start + length + 1;
            spans.push(start..end);
            position = end;
        }
        spans
    }
}

fn single_token(tokenizer: &Tokenizer, marker: &str) -> Result<u32> {
    match encode(tokenizer, marker)?.as_slice() {
        [id] => Ok(*id),
        ids => bail!(
            "the tokenizer has no {marker:?} token (it encodes it as {ids:?}), \
             so it is not a Qwen3.5 tokenizer"
        ),
    }
}

fn find(input_ids: &[u32], marker: &[u32]) -> Option<usize> {
    input_ids
        .windows(marker.len())
        .position(|window| window == marker)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::adapter::load;
    use std::sync::Arc;

    struct RenderedChat {
        tokenizer: Arc<Tokenizer>,
        input_ids: Vec<u32>,
        generated: Vec<Range<usize>>,
    }

    impl RenderedChat {
        fn new() -> Self {
            Self {
                tokenizer: load("tests/fixtures/tiny_qwen/tokenizer.json").unwrap(),
                input_ids: Vec::new(),
                generated: Vec::new(),
            }
        }

        fn prompt(mut self, text: &str) -> Self {
            self.input_ids
                .extend(encode(&self.tokenizer, text).unwrap());
            self
        }

        fn generated(mut self, text: &str) -> Self {
            let start = self.input_ids.len();
            self = self.prompt(text);
            self.generated.push(start..self.input_ids.len());
            self
        }

        fn spans(&self) -> Vec<Range<usize>> {
            Qwen35::from_tokenizer(&self.tokenizer)
                .unwrap()
                .generated_spans(&self.input_ids)
        }
    }

    #[test]
    fn an_answer_with_reasoning_starts_after_the_reasoning_opening() {
        let chat = RenderedChat::new()
            .prompt("<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n<think>\n")
            .generated("Add them.\n</think>\n\nIt is 4.<|im_end|>")
            .prompt("\n<|im_start|>user\n<tool_response>\nok\n</tool_response><|im_end|>\n")
            .prompt("<|im_start|>assistant\n<think>\n");

        assert_eq!(chat.spans(), chat.generated);
    }

    #[test]
    fn an_answer_with_thinking_off_starts_after_the_empty_reasoning() {
        let chat = RenderedChat::new()
            .prompt(
                "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            )
            .generated("Hello!<|im_end|>")
            .prompt("\n<|im_start|>user\n<tool_response>\nok\n</tool_response><|im_end|>\n")
            .prompt("<|im_start|>assistant\n<think>\n\n</think>\n\n");

        assert_eq!(chat.spans(), chat.generated);
    }

    #[test]
    fn an_answer_before_a_newer_user_message_starts_after_its_header() {
        let chat = RenderedChat::new()
            .prompt("<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n")
            .generated("Hello!<|im_end|>")
            .prompt("\n<|im_start|>user\nBye<|im_end|>\n<|im_start|>assistant\n<think>\n");

        assert_eq!(chat.spans(), chat.generated);
    }

    #[test]
    fn tool_calls_belong_to_their_answer() {
        let chat = RenderedChat::new()
            .prompt("<|im_start|>user\nWeather?<|im_end|>\n<|im_start|>assistant\n<think>\n")
            .generated(
                "Call the tool.\n</think>\n\n<tool_call>\n<function=weather>\n\
                 <parameter=city>\nParis\n</parameter>\n</function>\n</tool_call><|im_end|>",
            )
            .prompt("\n<|im_start|>user\n<tool_response>\nSunny\n</tool_response><|im_end|>\n")
            .prompt("<|im_start|>assistant\n<think>\n");

        assert_eq!(chat.spans(), chat.generated);
    }

    #[test]
    fn every_answer_of_several_turns_is_a_span() {
        let chat = RenderedChat::new()
            .prompt("<|im_start|>system\nBe brief.<|im_end|>\n")
            .prompt("<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n")
            .generated("Hello!<|im_end|>")
            .prompt("\n<|im_start|>user\nWeather?<|im_end|>\n<|im_start|>assistant\n<think>\n")
            .generated("Call the tool.\n</think>\n\n<tool_call>\n</tool_call><|im_end|>")
            .prompt("\n<|im_start|>user\n<tool_response>\nSunny\n</tool_response><|im_end|>\n")
            .prompt("<|im_start|>assistant\n<think>\n")
            .generated("Report it.\n</think>\n\nSunny.<|im_end|>")
            .prompt("\n<|im_start|>user\n<tool_response>\nok\n</tool_response><|im_end|>\n")
            .prompt("<|im_start|>assistant\n<think>\n");

        assert_eq!(chat.spans(), chat.generated);
    }

    #[test]
    fn the_generation_prompt_is_not_an_answer() {
        let chat = RenderedChat::new()
            .prompt("<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n");

        assert_eq!(chat.spans(), chat.generated);
    }

    #[test]
    fn a_header_without_its_end_is_not_an_answer() {
        let chat = RenderedChat::new()
            .prompt("<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n")
            .generated("Hello!<|im_end|>")
            .prompt("\n<|im_start|>user\nBye<|im_end|>\n<|im_start|>assistant\n<think>\nSo far")
            .prompt(" the client wrote this.");

        assert_eq!(chat.spans(), chat.generated);
    }

    #[test]
    fn a_tokenizer_without_the_markers_is_refused() {
        let tokenizer = load("tests/fixtures/tiny_tokenizer.json").unwrap();

        let error = Qwen35::from_tokenizer(&tokenizer).err().unwrap();

        assert!(error.to_string().contains("<|im_start|>"), "{error}");
    }
}
