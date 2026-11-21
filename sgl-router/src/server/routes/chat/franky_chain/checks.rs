// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::config::FrankyChainConfig;
use crate::server::error::ApiError;
use serde_json::{Map, Value};

/// A request field and when it asks for something: each field has its own
/// rule, since the lowering of chats and completions writes some of them
/// switched off on every request (`return_logprob: false`, `top_logprobs_num: 0`).
type FieldRule = (&'static str, fn(&Value) -> bool);

const CONSTRAINED_DECODING_FIELDS: [FieldRule; 4] = [
    ("json_schema", is_given),
    ("regex", is_given),
    ("ebnf", is_given),
    ("structural_tag", is_given),
];
const FIELDS_CHAINS_CANNOT_JOIN: [FieldRule; 4] = [
    ("return_hidden_states", is_true),
    ("return_routed_experts", is_true),
    ("return_flat_raw_top_logprobs", is_true),
    ("session_params", is_given),
];
/// Per-token output, which Sandy's text has none of in Ricky's terms.
const PER_TOKEN_OUTPUT_FIELDS: [FieldRule; 3] = [
    ("return_logprob", is_true),
    ("top_logprobs_num", is_positive),
    ("token_ids_logprob", is_given),
];

/// Refuses a `/generate` body that a chain under `config` cannot serve.
///
/// Errors: `ApiError::BadRequest` naming what the chain cannot serve.
pub(super) fn check_chains_can_serve(
    body: &Map<String, Value>,
    config: &FrankyChainConfig,
) -> Result<(), ApiError> {
    let empty = Map::new();
    let sampling = body
        .get("sampling_params")
        .and_then(Value::as_object)
        .unwrap_or(&empty);
    if asked(sampling, &CONSTRAINED_DECODING_FIELDS).is_some() {
        return Err(refused(
            "constrained decoding, whose state each request of a chain would restart",
        ));
    }
    if let Some(field) = asked(body, &FIELDS_CHAINS_CANNOT_JOIN) {
        return Err(refused(&format!("`{field}`")));
    }
    match asked(body, &PER_TOKEN_OUTPUT_FIELDS).filter(|_| config.sandy_call_tokens > 0) {
        Some(field) => Err(refused(&format!("`{field}` while Sandy writes"))),
        None => Ok(()),
    }
}

/// The first of `rules`' fields that `fields` asks for.
fn asked(fields: &Map<String, Value>, rules: &[FieldRule]) -> Option<&'static str> {
    rules
        .iter()
        .find(|(field, asks)| fields.get(*field).is_some_and(asks))
        .map(|(field, _)| *field)
}

pub(super) fn refused(what: &str) -> ApiError {
    ApiError::BadRequest(format!(
        "Franky chains (--franky-chain) cannot serve {what}"
    ))
}

fn is_true(value: &Value) -> bool {
    value.as_bool() == Some(true)
}

fn is_positive(value: &Value) -> bool {
    value.as_f64().is_some_and(|number| number > 0.0)
}

fn is_given(value: &Value) -> bool {
    match value {
        Value::String(text) => !text.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(fields) => !fields.is_empty(),
        Value::Null | Value::Bool(_) | Value::Number(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn object(value: Value) -> Map<String, Value> {
        match value {
            Value::Object(fields) => fields,
            _ => panic!("not an object"),
        }
    }

    #[test]
    fn chains_refuse_what_their_requests_cannot_carry() {
        let writes = FrankyChainConfig {
            sandy_call_tokens: 8,
            ..FrankyChainConfig::default()
        };
        for (body, config) in [
            (
                json!({"sampling_params": {"regex": "a+"}}),
                FrankyChainConfig::default(),
            ),
            (
                json!({"return_hidden_states": true}),
                FrankyChainConfig::default(),
            ),
            (json!({"return_logprob": true}), writes.clone()),
        ] {
            assert!(
                check_chains_can_serve(&object(body.clone()), &config).is_err(),
                "{body}"
            );
        }
        let logprobs = object(json!({"return_logprob": true, "stream": true}));
        assert!(check_chains_can_serve(&logprobs, &FrankyChainConfig::default()).is_ok());
    }

    #[test]
    fn fields_that_ask_for_nothing_pass_while_sandy_writes() {
        let writes = FrankyChainConfig {
            sandy_call_tokens: 8,
            ..FrankyChainConfig::default()
        };
        // As the chat and completion lowering write every request.
        let lowered_chat = object(json!({
            "return_logprob": false,
            "top_logprobs_num": 0,
            "logprob_start_len": -1,
            "token_ids_logprob": [],
            "sampling_params": {"regex": null, "json_schema": ""},
            "session_params": {},
        }));
        assert!(check_chains_can_serve(&lowered_chat, &writes).is_ok());
        for body in [
            json!({"top_logprobs_num": 2}),
            json!({"token_ids_logprob": [5]}),
        ] {
            assert!(
                check_chains_can_serve(&object(body.clone()), &writes).is_err(),
                "{body}"
            );
        }
    }
}
