// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use super::super::openai::sse_data;
use axum::body::Body;
use bytes::Bytes;
use futures::stream::{self, BoxStream};
use futures::{Stream, StreamExt};
use serde_json::Value;

pub(super) type EngineFrames = BoxStream<'static, Result<EngineFrame, axum::Error>>;

/// One frame of an engine's response.
pub(super) enum EngineFrame {
    Frame(Value),
    /// An error frame of a streamed response, which ends it.
    Error(Bytes),
}

/// The frames of `body`: its one frame when unary, its SSE frames when
/// streamed.
pub(super) fn engine_frames(body: Body, streamed: bool) -> EngineFrames {
    if !streamed {
        return stream::once(async move {
            let bytes = axum::body::to_bytes(body, usize::MAX).await?;
            let frame = serde_json::from_slice(&bytes).map_err(axum::Error::new)?;
            Ok(EngineFrame::Frame(frame))
        })
        .boxed();
    }
    streamed_frames(body).boxed()
}

fn streamed_frames(body: Body) -> impl Stream<Item = Result<EngineFrame, axum::Error>> {
    let not_done = |data: &Result<Bytes, axum::Error>| {
        futures::future::ready(!matches!(data, Ok(data) if data.as_ref() == b"[DONE]"))
    };
    sse_data(body).filter(not_done).map(|data| {
        let data = data?;
        Ok(match serde_json::from_slice::<Value>(&data) {
            Ok(frame) if frame.get("error").is_none() => EngineFrame::Frame(frame),
            _ => EngineFrame::Error(data),
        })
    })
}
