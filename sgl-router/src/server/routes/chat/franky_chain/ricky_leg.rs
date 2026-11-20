// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use super::answer_limits::WriteLimits;
use serde_json::{json, Map, Value};

/// What a Ricky leg carries for Franky beside the client's fields.
#[derive(Debug, Default)]
pub(super) struct RickyLegFields {
    /// `franky_chain`: the chain, the leg's first block and Ricky's blocks;
    /// `None` for a chain without blocks, whose engines hold nothing.
    pub(super) chain: Option<Value>,
    /// `franky_tap_runs`; `None` for a leg that reads no taps.
    pub(super) tap_runs: Option<Value>,
    /// `franky_sandy_input_ids`, when Sandy tokenizes on its own.
    pub(super) sandy_input_ids: Option<Value>,
}

/// The `/generate` body of a Ricky leg that continues the trace `input_ids`
/// within `limits`, with the client's sampling and stop rules.
pub(super) fn ricky_leg_body(
    client_body: &Map<String, Value>,
    input_ids: Value,
    limits: WriteLimits,
    first_leg: bool,
    franky: RickyLegFields,
) -> Map<String, Value> {
    let mut body = client_body.clone();
    body.insert("input_ids".into(), input_ids);
    let sampling = body.entry("sampling_params").or_insert_with(|| json!({}));
    limits.set_on(sampling);
    if !first_leg && body.contains_key("logprob_start_len") {
        body.insert("logprob_start_len".into(), (-1).into());
    }
    let fields = [
        ("franky_chain", franky.chain),
        ("franky_tap_runs", franky.tap_runs),
        ("franky_sandy_input_ids", franky.sandy_input_ids),
    ];
    for (name, value) in fields {
        if let Some(value) = value {
            body.insert(name.into(), value);
        }
    }
    body
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_leg_continues_the_trace_with_what_is_left_and_its_franky_fields() {
        let mut client = Map::new();
        client.insert("sampling_params".into(), json!({"temperature": 0.0}));
        client.insert("logprob_start_len".into(), json!(0));
        let franky = RickyLegFields {
            chain: Some(json!({"id": "c", "block": 3})),
            tap_runs: Some(json!([[4, 4]])),
            sandy_input_ids: None,
        };

        let limits = WriteLimits {
            max: Some(10),
            min: None,
        };
        let body = ricky_leg_body(&client, json!([1, 2, 3, 4]), limits, false, franky);

        assert_eq!(body["input_ids"], json!([1, 2, 3, 4]));
        assert_eq!(
            body["sampling_params"],
            json!({"temperature": 0.0, "max_new_tokens": 10})
        );
        assert_eq!(body["logprob_start_len"], json!(-1));
        assert_eq!(body["franky_chain"], json!({"id": "c", "block": 3}));
        assert_eq!(body["franky_tap_runs"], json!([[4, 4]]));
        assert!(!body.contains_key("franky_sandy_input_ids"));
    }

    #[test]
    fn without_a_limit_a_leg_writes_up_to_the_context() {
        let limits = WriteLimits {
            max: None,
            min: None,
        };
        let body = ricky_leg_body(
            &Map::new(),
            json!([1]),
            limits,
            true,
            RickyLegFields::default(),
        );

        assert_eq!(body["sampling_params"], json!({"max_new_tokens": null}));
    }
}
