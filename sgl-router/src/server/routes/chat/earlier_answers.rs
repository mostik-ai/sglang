// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use super::preparation::{invalid_request, PreparedRequest, RoutingFields, CHAT_PATH};
use crate::discovery::ModelId;
use crate::policies::has_caller_input_ids;
use crate::server::app_context::AppContext;
use crate::server::error::ApiError;
use axum::http::HeaderMap;
use bytes::Bytes;
use serde_json::Value;

/// A chat for Franky workers, prepared as any chat. When it has earlier answers
/// but no `input_ids`, the engines get the ids the router rendered, with the
/// runs of those answers as `franky_generated_spans`.
///
/// Errors: `ApiError::BadRequest` for `franky_generated_spans` without
/// `input_ids`, and for earlier answers the router cannot mark: without
/// `--franky-chat-adapter`, or in a chat it leaves to the engine to render.
pub(super) fn prepare_franky_chat(
    ctx: &AppContext,
    model: ModelId,
    fields: RoutingFields,
    headers: &HeaderMap,
    body: Bytes,
    policy_needs_request_tokens: bool,
) -> Result<PreparedRequest, ApiError> {
    let chat: Value = serde_json::from_slice(&body).map_err(|_| invalid_request())?;
    if !router_marks_earlier_answers(&chat)? {
        return PreparedRequest::openai(
            ctx,
            CHAT_PATH,
            model,
            fields,
            headers,
            body,
            policy_needs_request_tokens,
        );
    }
    let Some(adapter) = ctx.franky_chat_adapter.as_deref() else {
        return Err(cannot_mark_answers(&model, "with --franky-chat-adapter"));
    };
    let mut request = PreparedRequest::openai(
        ctx,
        CHAT_PATH,
        model,
        fields,
        headers,
        body,
        policy_needs_request_tokens,
    )?;
    let Some(rendered_ids) = request.rendered_chat_ids() else {
        return Err(cannot_mark_answers(
            &request.model,
            "in the chats it renders, and it leaves this one to the engine",
        ));
    };
    let spans = adapter.generated_spans(rendered_ids);
    if !spans.is_empty() {
        request.add_franky_generated_spans(&spans)?;
    }
    Ok(request)
}

fn router_marks_earlier_answers(chat: &Value) -> Result<bool, ApiError> {
    if has_caller_input_ids(chat) {
        return Ok(false);
    }
    if chat
        .get("franky_generated_spans")
        .is_some_and(|spans| !spans.is_null())
    {
        return Err(ApiError::BadRequest(
            "franky_generated_spans are positions in input_ids, so a chat that sends them \
             must send input_ids too"
                .into(),
        ));
    }
    Ok(has_assistant_message(chat))
}

fn has_assistant_message(chat: &Value) -> bool {
    chat["messages"].as_array().is_some_and(|messages| {
        messages.iter().any(|message| {
            message["role"]
                .as_str()
                .is_some_and(|role| role.eq_ignore_ascii_case("assistant"))
        })
    })
}

fn cannot_mark_answers(model: &ModelId, only_when: &str) -> ApiError {
    ApiError::BadRequest(format!(
        "model {model} is served by Franky workers, and this chat has earlier assistant \
         messages, whose tokens the router marks for Ricky only {only_when}; send \
         input_ids with franky_generated_spans instead"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn messages_with_an_answer() -> Value {
        json!([
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "Bye"},
        ])
    }

    #[test]
    fn the_router_marks_the_assistant_messages_of_chats_without_input_ids() {
        let messages = messages_with_an_answer();
        for (chat, marks) in [
            (json!({"messages": messages}), true),
            (
                json!({"messages": [{"role": "user", "content": "Hi"}, {"role": "Assistant", "content": "Hello!"}]}),
                true,
            ),
            (
                json!({"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}]}),
                false,
            ),
            (
                json!({"messages": [{"role": "user", "content": "Hi"}], "franky_generated_spans": null}),
                false,
            ),
            (json!({"messages": messages, "input_ids": [1, 2, 3]}), false),
            (
                json!({"messages": messages, "input_ids": [1, 2, 3], "franky_generated_spans": [[1, 2]]}),
                false,
            ),
        ] {
            assert_eq!(
                router_marks_earlier_answers(&chat).unwrap(),
                marks,
                "{chat}"
            );
        }
    }

    #[test]
    fn spans_without_input_ids_are_refused() {
        let chat =
            json!({"messages": messages_with_an_answer(), "franky_generated_spans": [[1, 2]]});

        let error = router_marks_earlier_answers(&chat).unwrap_err();

        assert!(
            matches!(&error, ApiError::BadRequest(message) if message.contains("input_ids")),
            "{error:?}"
        );
    }
}
