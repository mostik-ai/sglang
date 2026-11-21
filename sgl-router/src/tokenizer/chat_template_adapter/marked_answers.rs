// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use super::ChatTemplateAdapter;
use crate::tokenizer::adapter::encode;
use anyhow::{bail, Result};
use dynamo_tokenizers::Tokenizer;
use std::ops::Range;

/// A chat template that opens each answer with a header and one of the
/// openings a generation prompt ends with, and closes it with one token. The
/// answer is what follows the opening, its closing token included.
pub struct MarkedAnswers {
    answer_header: Vec<u32>,
    openings: Vec<Vec<u32>>,
    answer_end: u32,
    text_end: Option<u32>,
}

/// A template's markers, as text.
struct Markers {
    template: &'static str,
    /// Markers the template's tokenizer must have as tokens of their own.
    special_tokens: &'static [&'static str],
    answer_header: &'static str,
    /// Tried in order, so a longer opening comes before its prefixes.
    openings: &'static [&'static str],
    answer_end: &'static str,
    /// Where the answer's text ends when the template closes the answer with
    /// more than the model's EOS.
    text_end: Option<&'static str>,
}

impl MarkedAnswers {
    /// Qwen3.5: after `<|im_start|>assistant\n` and its `<think>\n\n</think>\n\n`
    /// or `<think>\n`, up to its `<|im_end|>`.
    ///
    /// Errors: `tokenizer` has no token of its own for a marker.
    pub fn qwen35(tokenizer: &Tokenizer) -> Result<Self> {
        Self::from_markers(
            tokenizer,
            &Markers {
                template: "Qwen3.5",
                special_tokens: &["<|im_start|>", "<think>", "</think>", "<|im_end|>"],
                answer_header: "<|im_start|>assistant\n",
                openings: &["<think>\n\n</think>\n\n", "<think>\n"],
                answer_end: "<|im_end|>",
                text_end: None,
            },
        )
    }

    /// DeepSeek V4 and V4.1: after `<｜Assistant｜>` and its `<think>` or
    /// `</think>`, up to its `<｜end▁of▁sentence｜>`. Reasoning an earlier
    /// answer keeps belongs to it.
    ///
    /// Errors: `tokenizer` has no token of its own for a marker.
    pub fn deepseek_v4(tokenizer: &Tokenizer) -> Result<Self> {
        Self::from_markers(
            tokenizer,
            &Markers {
                template: "DeepSeek V4",
                special_tokens: &[
                    "<｜Assistant｜>",
                    "<think>",
                    "</think>",
                    "<｜end▁of▁sentence｜>",
                ],
                answer_header: "<｜Assistant｜>",
                openings: &["<think>", "</think>"],
                answer_end: "<｜end▁of▁sentence｜>",
                text_end: None,
            },
        )
    }

    /// Kimi K3: after `<|open|>message role="assistant"<|sep|>` and its think or
    /// response opening, up to its `<|end_of_msg|>`. Its text ends at the first
    /// `<|close|>`, which starts closing the response and the message.
    ///
    /// Errors: `tokenizer` has no token of its own for a marker.
    pub fn kimi_k3(tokenizer: &Tokenizer) -> Result<Self> {
        Self::from_markers(
            tokenizer,
            &Markers {
                template: "Kimi K3",
                special_tokens: &["<|open|>", "<|sep|>", "<|close|>", "<|end_of_msg|>"],
                answer_header: "<|open|>message role=\"assistant\"<|sep|>",
                openings: &["<|open|>think<|sep|>", "<|open|>response<|sep|>"],
                answer_end: "<|end_of_msg|>",
                text_end: Some("<|close|>"),
            },
        )
    }

    fn from_markers(tokenizer: &Tokenizer, markers: &Markers) -> Result<Self> {
        for marker in markers.special_tokens {
            single_token(tokenizer, marker, markers.template)?;
        }
        let openings = markers
            .openings
            .iter()
            .map(|opening| encode(tokenizer, opening))
            .collect::<Result<_>>()?;
        Ok(Self {
            answer_header: encode(tokenizer, markers.answer_header)?,
            openings,
            answer_end: single_token(tokenizer, markers.answer_end, markers.template)?,
            text_end: markers
                .text_end
                .map(|marker| single_token(tokenizer, marker, markers.template))
                .transpose()?,
        })
    }

    fn answer_start(&self, input_ids: &[u32], header_end: usize) -> usize {
        let after_header = &input_ids[header_end..];
        let opening = self
            .openings
            .iter()
            .find(|opening| after_header.starts_with(opening));
        header_end + opening.map_or(0, Vec::len)
    }
}

impl ChatTemplateAdapter for MarkedAnswers {
    fn text_end(&self) -> Option<u32> {
        self.text_end
    }

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

fn single_token(tokenizer: &Tokenizer, marker: &str, template: &str) -> Result<u32> {
    match encode(tokenizer, marker)?.as_slice() {
        [id] => Ok(*id),
        ids => bail!(
            "the tokenizer has no {marker:?} token (it encodes it as {ids:?}), \
             so it is not a {template} tokenizer"
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
        template: fn(&Tokenizer) -> Result<MarkedAnswers>,
        input_ids: Vec<u32>,
        generated: Vec<Range<usize>>,
    }

    impl RenderedChat {
        fn qwen35() -> Self {
            Self::new(
                "tests/fixtures/tiny_qwen/tokenizer.json",
                MarkedAnswers::qwen35,
            )
        }

        fn deepseek_v4() -> Self {
            Self::new(
                "tests/fixtures/tiny_deepseek_v4/tokenizer.json",
                MarkedAnswers::deepseek_v4,
            )
        }

        fn new(path: &str, template: fn(&Tokenizer) -> Result<MarkedAnswers>) -> Self {
            Self {
                tokenizer: load(path).unwrap(),
                template,
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
            (self.template)(&self.tokenizer)
                .unwrap()
                .generated_spans(&self.input_ids)
        }
    }

    #[test]
    fn an_answer_with_reasoning_starts_after_the_reasoning_opening() {
        let chat = RenderedChat::qwen35()
            .prompt("<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n<think>\n")
            .generated("Add them.\n</think>\n\nIt is 4.<|im_end|>")
            .prompt("\n<|im_start|>user\n<tool_response>\nok\n</tool_response><|im_end|>\n")
            .prompt("<|im_start|>assistant\n<think>\n");

        assert_eq!(chat.spans(), chat.generated);
    }

    #[test]
    fn an_answer_with_thinking_off_starts_after_the_empty_reasoning() {
        let chat = RenderedChat::qwen35()
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
        let chat = RenderedChat::qwen35()
            .prompt("<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n")
            .generated("Hello!<|im_end|>")
            .prompt("\n<|im_start|>user\nBye<|im_end|>\n<|im_start|>assistant\n<think>\n");

        assert_eq!(chat.spans(), chat.generated);
    }

    #[test]
    fn tool_calls_belong_to_their_answer() {
        let chat = RenderedChat::qwen35()
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
        let chat = RenderedChat::qwen35()
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
        let chat = RenderedChat::qwen35()
            .prompt("<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n");

        assert_eq!(chat.spans(), chat.generated);
    }

    #[test]
    fn a_header_without_its_end_is_not_an_answer() {
        let chat = RenderedChat::qwen35()
            .prompt("<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n")
            .generated("Hello!<|im_end|>")
            .prompt("\n<|im_start|>user\nBye<|im_end|>\n<|im_start|>assistant\n<think>\nSo far")
            .prompt(" the client wrote this.");

        assert_eq!(chat.spans(), chat.generated);
    }

    #[test]
    fn a_deepseek_answer_starts_after_its_reasoning_switch() {
        let chat = RenderedChat::deepseek_v4()
            .prompt("<｜begin▁of▁sentence｜><｜User｜>Hi<｜Assistant｜></think>")
            .generated("Hello!<｜end▁of▁sentence｜>")
            .prompt("<｜User｜>Weather?<｜Assistant｜><think>")
            .generated("Call it.</think>\n\n<calls>Paris</calls><｜end▁of▁sentence｜>")
            .prompt("<｜User｜><tool_result>Sunny</tool_result><｜Assistant｜><think>");

        assert_eq!(chat.spans(), chat.generated);
    }

    #[test]
    fn a_tokenizer_without_the_markers_is_refused() {
        let tokenizer = load("tests/fixtures/tiny_tokenizer.json").unwrap();

        for (template, marker) in [
            (
                MarkedAnswers::qwen35 as fn(&Tokenizer) -> Result<MarkedAnswers>,
                "<|im_start|>",
            ),
            (MarkedAnswers::deepseek_v4, "<｜Assistant｜>"),
            (MarkedAnswers::kimi_k3, "<|open|>"),
        ] {
            let error = template(&tokenizer).err().unwrap();

            assert!(error.to_string().contains(marker), "{error}");
        }
    }
}
