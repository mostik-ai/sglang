// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! The SSE events of a chain's stream.

use axum::body::Body;
use axum::http::Response;
use bytes::Bytes;
use serde_json::{json, Value};
use tokio::sync::mpsc;

pub(super) type Events = mpsc::Sender<Result<Bytes, axum::Error>>;

const DONE_EVENT: &[u8] = b"data: [DONE]\n\n";

pub(super) fn done_event() -> Bytes {
    Bytes::from_static(DONE_EVENT)
}

pub(super) fn json_event(frame: &Value) -> Bytes {
    data_event(frame.to_string().as_bytes())
}

pub(super) fn data_event(data: &[u8]) -> Bytes {
    [b"data: ".as_slice(), data, b"\n\n"].concat().into()
}

pub(super) fn error_event(message: &str) -> Bytes {
    json_event(&json!({"error": {"message": message}}))
}

/// The error event of a request that failed with `response`: its own error
/// object when it sent one.
pub(super) async fn error_event_from_response(response: Response<Body>) -> Bytes {
    let status = response.status();
    let body = match axum::body::to_bytes(response.into_body(), usize::MAX).await {
        Ok(body) => body,
        Err(error) => return error_event(&format!(
            "a Franky chain's request failed with {status}, and reading its body failed: {error}"
        )),
    };
    match serde_json::from_slice::<Value>(&body) {
        Ok(error) if error.get("error").is_some() => data_event(&body),
        _ => error_event(&format!(
            "a Franky chain's request failed with {status}: {}",
            String::from_utf8_lossy(&body)
        )),
    }
}

/// Ends a stream with an error event.
pub(super) async fn fail_stream(events: &Events, message: &str) {
    let _ = events.send(Ok(error_event(message))).await;
    let _ = events.send(Ok(done_event())).await;
}
