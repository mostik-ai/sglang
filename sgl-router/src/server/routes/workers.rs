// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Workers that clients add and remove at run time (`--worker-api`), in the
//! shape Miles registers the engines it launches: `POST /workers` with
//! `{"url", "worker_type", "bootstrap_port"}`, `GET /workers`, and
//! `DELETE /workers/{id}`.

use crate::discovery::{DiscoveryEvent, WorkerId, WorkerMode, WorkerSpec};
use crate::server::app_context::AppContext;
use crate::server::error::ApiError;
use axum::extract::{Path, State};
use axum::Json;
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;

/// How long `POST /workers` waits for the manager to register the worker, which
/// first reads its `/server_info`.
const REGISTER_TIMEOUT: Duration = Duration::from_secs(120);
const REGISTER_POLL: Duration = Duration::from_millis(100);

#[derive(Deserialize)]
pub struct AddWorker {
    url: String,
    #[serde(default)]
    worker_type: Option<String>,
    #[serde(default)]
    bootstrap_port: Option<u16>,
}

/// Registers a worker and answers once the router routes to it. Its role
/// (including Sandy) and models come from its `/server_info`, as for a static URL.
pub async fn add_worker(
    State(ctx): State<Arc<AppContext>>,
    Json(body): Json<AddWorker>,
) -> Result<Json<Value>, ApiError> {
    let events = worker_events(&ctx)?;
    let mode = match body.worker_type.as_deref() {
        None | Some("regular") => WorkerMode::Plain,
        Some("prefill") => WorkerMode::Prefill,
        Some("decode") => WorkerMode::Decode,
        Some(other) => {
            return Err(ApiError::BadRequest(format!(
                "worker_type is regular, prefill or decode, not {other:?}"
            )))
        }
    };
    let id = api_worker_id(&body.url);
    let spec = WorkerSpec {
        id: id.clone(),
        url: body.url.clone(),
        mode,
        bootstrap_port: body.bootstrap_port,
        ..Default::default()
    };
    events
        .send(DiscoveryEvent::Added(spec))
        .await
        .map_err(|_| ApiError::Internal(anyhow::anyhow!("the worker manager stopped")))?;
    let registered = tokio::time::timeout(REGISTER_TIMEOUT, async {
        while !ctx
            .registry
            .get(&id)
            .is_some_and(|worker| !worker.model_ids.is_empty())
        {
            tokio::time::sleep(REGISTER_POLL).await;
        }
    })
    .await;
    if registered.is_err() {
        return Err(ApiError::Internal(anyhow::anyhow!(
            "worker {} did not register within {REGISTER_TIMEOUT:?}; check its /server_info",
            body.url
        )));
    }
    Ok(Json(json!({ "id": id.0, "url": body.url })))
}

/// Every registered worker, static or added, with the id `DELETE` takes.
pub async fn list_workers(State(ctx): State<Arc<AppContext>>) -> Json<Value> {
    let workers: Vec<Value> = ctx
        .registry
        .all()
        .iter()
        .map(|worker| {
            json!({
                "id": worker.id.0,
                "url": worker.url,
                "mode": worker.mode(),
                "model_ids": worker.model_ids,
            })
        })
        .collect();
    Json(json!({ "workers": workers }))
}

pub async fn remove_worker(
    State(ctx): State<Arc<AppContext>>,
    Path(id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let events = worker_events(&ctx)?;
    events
        .send(DiscoveryEvent::Removed {
            id: WorkerId(id.clone()),
        })
        .await
        .map_err(|_| ApiError::Internal(anyhow::anyhow!("the worker manager stopped")))?;
    Ok(Json(json!({ "id": id })))
}

fn worker_events(ctx: &AppContext) -> Result<&mpsc::Sender<DiscoveryEvent>, ApiError> {
    ctx.worker_events.as_ref().ok_or_else(|| {
        ApiError::BadRequest("this router takes no workers at run time; start it with --worker-api".into())
    })
}

/// An id that fits one URL path segment, so `DELETE /workers/{id}` reaches it.
fn api_worker_id(url: &str) -> WorkerId {
    let id = url
        .trim_end_matches('/')
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();
    WorkerId(id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_api_worker_id_fits_one_path_segment() {
        assert_eq!(
            api_worker_id("http://10.0.0.1:30000/").0,
            "http---10-0-0-1-30000"
        );
    }
}
