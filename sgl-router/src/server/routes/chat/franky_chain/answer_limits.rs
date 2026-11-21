// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use serde_json::Value;

/// The client's token limits of the whole answer; each request of a chain
/// gets what is left of them.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct AnswerLimits {
    max: Option<u64>,
    min: Option<u64>,
}

/// `max_new_tokens` and `min_new_tokens` of one request of a chain; `max` is
/// `None` for a request that writes up to the context.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct WriteLimits {
    pub(super) max: Option<u64>,
    pub(super) min: Option<u64>,
}

impl AnswerLimits {
    /// The limits in the client's `sampling_params`.
    pub(super) fn of_sampling(sampling: &Value) -> Self {
        Self {
            max: sampling["max_new_tokens"].as_u64(),
            min: sampling["min_new_tokens"].as_u64(),
        }
    }

    /// The tokens left of the answer after `written`; `None` without a limit.
    pub(super) fn left(&self, written: u64) -> Option<u64> {
        self.max.map(|max| max.saturating_sub(written))
    }

    /// The limits of a request that writes after `written` tokens of the
    /// answer, at most `cap` of them.
    pub(super) fn of_request(&self, written: u64, cap: Option<u64>) -> WriteLimits {
        let max = match (self.left(written), cap) {
            (Some(left), Some(cap)) => Some(left.min(cap)),
            (left, cap) => left.or(cap),
        };
        let min = self.min.map(|min| {
            let left = min.saturating_sub(written);
            max.map_or(left, |max| left.min(max))
        });
        WriteLimits { max, min }
    }
}

impl WriteLimits {
    /// Sets the limits in a request's `sampling_params`.
    pub(super) fn set_on(&self, sampling: &mut Value) {
        sampling["max_new_tokens"] = self.max.map_or(Value::Null, Value::from);
        if let Some(min) = self.min {
            sampling["min_new_tokens"] = min.into();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn each_request_gets_what_is_left_of_the_answers_limits() {
        let limits =
            AnswerLimits::of_sampling(&json!({"max_new_tokens": 100, "min_new_tokens": 70}));

        assert_eq!(limits.left(30), Some(70));
        assert_eq!(
            limits.of_request(30, Some(64)),
            WriteLimits {
                max: Some(64),
                min: Some(40)
            }
        );
        assert_eq!(
            limits.of_request(90, None),
            WriteLimits {
                max: Some(10),
                min: Some(0)
            }
        );
    }

    #[test]
    fn without_limits_a_request_writes_up_to_its_cap_or_the_context() {
        let limits = AnswerLimits::of_sampling(&json!({"temperature": 0.0}));

        assert_eq!(limits.left(30), None);
        assert_eq!(
            limits.of_request(30, Some(64)),
            WriteLimits {
                max: Some(64),
                min: None
            }
        );
        let mut sampling = json!({"temperature": 0.0});
        limits.of_request(30, None).set_on(&mut sampling);
        assert_eq!(
            sampling,
            json!({"temperature": 0.0, "max_new_tokens": null})
        );
    }
}
