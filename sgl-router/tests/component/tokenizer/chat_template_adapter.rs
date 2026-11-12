// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use serde_json::{json, Value};
use sgl_router::config::ChatTemplateAdapterKind;
use sgl_router::tokenizer::adapter;
use sgl_router::tokenizer::chat_formatter::ChatFormatter;
use sgl_router::tokenizer::chat_template_adapter::{self, ChatTemplateAdapter};
use std::path::Path;
use std::slice;
use std::sync::Arc;

struct TemplateSamples {
    kind: ChatTemplateAdapterKind,
    model_id: &'static str,
    tokenizer_dir_variable: &'static str,
    answer_end: &'static str,
    turn_after_answer: Value,
    chats: Vec<Value>,
}

struct SampleRenderer {
    formatter: ChatFormatter,
    tokenizer: Arc<dynamo_tokenizers::Tokenizer>,
    adapter: Arc<dyn ChatTemplateAdapter>,
    answer_end: u32,
}

impl SampleRenderer {
    fn load(samples: &TemplateSamples, tokenizer_dir: &Path) -> Self {
        let tokenizer_path = tokenizer_dir.join("tokenizer.json");
        let tokenizer_path = tokenizer_path.to_str().unwrap();
        let tokenizer = adapter::load(tokenizer_path).unwrap();
        let formatter = ChatFormatter::load(samples.model_id, tokenizer_path)
            .unwrap()
            .expect("the model ships a chat template");
        let chat_adapter = chat_template_adapter::build(samples.kind, &tokenizer).unwrap();
        let answer_end = match adapter::encode(&tokenizer, samples.answer_end)
            .unwrap()
            .as_slice()
        {
            [id] => *id,
            ids => panic!("{} is not one token: {ids:?}", samples.answer_end),
        };
        Self {
            formatter,
            tokenizer,
            adapter: chat_adapter,
            answer_end,
        }
    }

    fn render(&self, chat: &Value, messages: &[Value]) -> Vec<u32> {
        let mut request = chat.clone();
        request["messages"] = json!(messages);
        self.formatter.encode(&self.tokenizer, &request).unwrap()
    }
}

fn qwen35() -> TemplateSamples {
    let weather_tool = json!({"type": "function", "function": {
        "name": "get_weather",
        "description": "The current weather of a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }});
    let weather_call = |city: &str| {
        json!([{"id": "call_1", "type": "function", "function": {
            "name": "get_weather",
            "arguments": {"city": city},
        }}])
    };
    TemplateSamples {
        kind: ChatTemplateAdapterKind::Qwen35,
        model_id: "Qwen/Qwen3.5-35B-A3B",
        tokenizer_dir_variable: "QWEN35_TOKENIZER_DIR",
        answer_end: "<|im_end|>",
        // chat_template.jinja keeps an answer's reasoning before a tool turn, not a user turn.
        turn_after_answer: json!({"role": "tool", "content": "ok"}),
        chats: vec![
            json!({"messages": [
                {"role": "user", "content": "What is 17 * 23?"},
                {"role": "assistant", "reasoning_content": "17 * 20 + 17 * 3 = 340 + 51.",
                 "content": "17 * 23 = 391."},
            ]}),
            json!({"chat_template_kwargs": {"enable_thinking": false}, "messages": [
                {"role": "system", "content": "You are terse."},
                {"role": "user", "content": "Name a prime number."},
                {"role": "assistant", "content": "7"},
            ]}),
            json!({"tools": [weather_tool], "messages": [
                {"role": "user", "content": "What is the weather in Paris?"},
                {"role": "assistant", "reasoning_content": "I need the weather tool.",
                 "content": "", "tool_calls": weather_call("Paris")},
                {"role": "tool", "tool_call_id": "call_1", "content": "Sunny, 21 C"},
                {"role": "assistant", "reasoning_content": "The tool says it is sunny.",
                 "content": "It is sunny and 21 C in Paris."},
            ]}),
            json!({"messages": [
                {"role": "user", "content": "Hi!"},
                {"role": "assistant", "reasoning_content": "Greet back.",
                 "content": "Hello! How can I help?"},
                {"role": "user", "content": "Count to three."},
                {"role": "assistant", "reasoning_content": "Easy.", "content": "1, 2, 3."},
                {"role": "user", "content": "Now backwards."},
                {"role": "assistant", "reasoning_content": "Reverse it.", "content": "3, 2, 1."},
            ]}),
            json!({"chat_template_kwargs": {"enable_thinking": false}, "tools": [weather_tool],
                   "messages": [
                {"role": "user", "content": "Weather in Oslo?"},
                {"role": "assistant", "content": "Let me check.",
                 "tool_calls": weather_call("Oslo")},
                {"role": "tool", "tool_call_id": "call_1", "content": "Snow, -3 C"},
                {"role": "assistant", "content": "It is snowing and -3 C in Oslo."},
                {"role": "user", "content": "Thanks!"},
                {"role": "assistant", "content": "You are welcome."},
            ]}),
        ],
    }
}

#[test]
fn every_answer_of_the_sample_chats_is_the_span_the_model_wrote() {
    for samples in [qwen35()] {
        let Ok(tokenizer_dir) = std::env::var(samples.tokenizer_dir_variable) else {
            eprintln!(
                "skip {:?}: set {} to the model's tokenizer directory",
                samples.kind, samples.tokenizer_dir_variable
            );
            continue;
        };
        let renderer = SampleRenderer::load(&samples, Path::new(&tokenizer_dir));
        for chat in &samples.chats {
            let messages = chat["messages"].as_array().unwrap();
            let answer_indices = messages
                .iter()
                .enumerate()
                .filter(|(_, message)| message["role"] == "assistant")
                .map(|(index, _)| index);
            for (answer_count, index) in (1..).zip(answer_indices) {
                let prompt = renderer.render(chat, &messages[..index]);
                let through_answer = [
                    &messages[..=index],
                    slice::from_ref(&samples.turn_after_answer),
                ];
                let rendered = renderer.render(chat, &through_answer.concat());
                assert!(
                    rendered.starts_with(&prompt),
                    "{chat}, answer {answer_count}"
                );
                let answer_length = rendered[prompt.len()..]
                    .iter()
                    .position(|&id| id == renderer.answer_end)
                    .unwrap();
                let span = prompt.len()..prompt.len() + answer_length + 1;

                let spans = renderer.adapter.generated_spans(&rendered);

                assert_eq!(spans.len(), answer_count, "{chat}");
                assert_eq!(spans.last(), Some(&span), "{chat}, answer {answer_count}");
            }
        }
    }
}
