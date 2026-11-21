// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::config::FrankyChainConfig;
use crate::server::error::ApiError;
use serde_json::{Map, Value};

const CONSTRAINED_DECODING_FIELDS: [&str; 4] = ["json_schema", "regex", "ebnf", "structural_tag"];
const FIELDS_CHAINS_CANNOT_JOIN: [&str; 4] = [
    "return_hidden_states",
    "return_routed_experts",
    "return_flat_raw_top_logprobs",
    "session_params",
];
/// Per-token output, which Sandy's text has none of in Ricky's terms.
const PER_TOKEN_OUTPUT_FIELDS: [&str; 3] =
    ["return_logprob", "top_logprobs_num", "token_ids_logprob"];

/// Refuses a `/generate` body that a chain under `config` cannot serve.
///
/// Errors: `ApiError::BadRequest` naming what the chain cannot serve.
pub(super) fn check_chains_can_serve(
    body: &Map<String, Value>,
    config: &FrankyChainConfig,
) -> Result<(), ApiError> {
    let sampling = body.get("sampling_params");
    let constrained = CONSTRAINED_DECODING_FIELDS.iter().any(|field| {
        sampling
            .and_then(|params| params.get(field))
            .is_some_and(is_set)
    });
    if constrained {
        return Err(refused(
            "constrained decoding, whose state each request of a chain would restart",
        ));
    }
    if let Some(field) = FIELDS_CHAINS_CANNOT_JOIN
        .iter()
        .find(|field| body.get(**field).is_some_and(is_set))
    {
        return Err(refused(&format!("`{field}`")));
    }
    let sandy_writes = config.sandy_call_tokens > 0;
    match PER_TOKEN_OUTPUT_FIELDS
        .iter()
        .find(|field| sandy_writes && body.get(**field).is_some_and(is_set))
    {
        Some(field) => Err(refused(&format!("`{field}` while Sandy writes"))),
        None => Ok(()),
    }
}

pub(super) fn refused(what: &str) -> ApiError {
    ApiError::BadRequest(format!(
        "Franky chains (--franky-chain) cannot serve {what}"
    ))
}

/// Whether a request field asks for something. Null, false, zero and empty
/// values ask for nothing, and the chat lowering writes some of them on every
/// request (`return_logprob: false`, `top_logprobs_num: 0`).
fn is_set(value: &Value) -> bool {
    match value {
        Value::Null | Value::Bool(false) => false,
        Value::Bool(true) => true,
        Value::Number(number) => number.as_f64() != Some(0.0),
        Value::String(text) => !text.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(fields) => !fields.is_empty(),
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
        // As the chat lowering writes every request.
        let lowered_chat = object(json!({
            "return_logprob": false,
            "top_logprobs_num": 0,
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
