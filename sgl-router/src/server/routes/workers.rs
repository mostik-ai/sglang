// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

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

const REGISTER_TIMEOUT: Duration = Duration::from_secs(120);
const REGISTER_POLL_INTERVAL: Duration = Duration::from_millis(100);

/// The body of `POST /workers`, as Miles sends it for an engine it launched.
#[derive(Deserialize)]
pub struct AddWorker {
    url: String,
    worker_type: Option<String>,
    bootstrap_port: Option<u16>,
}

impl AddWorker {
    fn spec(&self) -> Result<WorkerSpec, ApiError> {
        Ok(WorkerSpec {
            id: worker_id_for_url(&self.url),
            url: self.url.clone(),
            mode: worker_mode(self.worker_type.as_deref())?,
            bootstrap_port: self.bootstrap_port,
            ..Default::default()
        })
    }
}

/// Registers a worker and answers `{"id", "url"}` once the router routes to it.
/// `worker_type` is `regular` (the default), `prefill` or `decode`; the
/// worker's role (including Sandy) and models come from its `/server_info`, as
/// for a static URL.
pub async fn add_worker(
    State(ctx): State<Arc<AppContext>>,
    Json(body): Json<AddWorker>,
) -> Result<Json<Value>, ApiError> {
    let events = worker_api_events(&ctx)?;
    let spec = body.spec()?;
    let id = spec.id.clone();
    send_to_manager(events, DiscoveryEvent::Added(spec)).await?;
    wait_until_registered(&ctx, &id, &body.url).await?;
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

/// Asks the worker manager to remove worker `id`, and answers before the
/// worker leaves the registry.
pub async fn remove_worker(
    State(ctx): State<Arc<AppContext>>,
    Path(id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let events = worker_api_events(&ctx)?;
    let removal = DiscoveryEvent::Removed {
        id: WorkerId(id.clone()),
    };
    send_to_manager(events, removal).await?;
    Ok(Json(json!({ "id": id })))
}

fn worker_api_events(ctx: &AppContext) -> Result<&mpsc::Sender<DiscoveryEvent>, ApiError> {
    ctx.worker_api_events.as_ref().ok_or_else(|| {
        ApiError::BadRequest(
            "this router takes no workers at run time; start it with --worker-api".into(),
        )
    })
}

async fn send_to_manager(
    events: &mpsc::Sender<DiscoveryEvent>,
    event: DiscoveryEvent,
) -> Result<(), ApiError> {
    events
        .send(event)
        .await
        .map_err(|_| ApiError::Internal(anyhow::anyhow!("the worker manager stopped")))
}

async fn wait_until_registered(ctx: &AppContext, id: &WorkerId, url: &str) -> Result<(), ApiError> {
    let registered = tokio::time::timeout(REGISTER_TIMEOUT, async {
        while ctx
            .registry
            .get(id)
            .is_none_or(|worker| worker.model_ids.is_empty())
        {
            tokio::time::sleep(REGISTER_POLL_INTERVAL).await;
        }
    })
    .await;
    registered.map_err(|_| {
        ApiError::Internal(anyhow::anyhow!(
            "worker {url} did not register within {REGISTER_TIMEOUT:?}; check its /server_info"
        ))
    })
}

fn worker_mode(worker_type: Option<&str>) -> Result<WorkerMode, ApiError> {
    match worker_type {
        None | Some("regular") => Ok(WorkerMode::Plain),
        Some("prefill") => Ok(WorkerMode::Prefill),
        Some("decode") => Ok(WorkerMode::Decode),
        Some(other) => Err(ApiError::BadRequest(format!(
            "worker_type is regular, prefill or decode, not {other:?}"
        ))),
    }
}

fn worker_id_for_url(url: &str) -> WorkerId {
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
    use crate::discovery::ModelId;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tokio::sync::Notify;
    use tower::ServiceExt;

    const ADDED_ID: &str = "http---10-0-0-1-30000";

    struct FakeManager {
        ctx: Arc<AppContext>,
        removed: Arc<Notify>,
    }

    impl FakeManager {
        fn start() -> Self {
            let (events, mut received) = mpsc::channel(8);
            let mut ctx = AppContext::stub();
            ctx.worker_api_events = Some(events);
            let ctx = Arc::new(ctx);
            let registry = Arc::clone(&ctx.registry);
            let removed = Arc::new(Notify::new());
            let notify_removed = Arc::clone(&removed);
            tokio::spawn(async move {
                while let Some(event) = received.recv().await {
                    match event {
                        DiscoveryEvent::Added(spec) => registry
                            .add(WorkerSpec {
                                model_ids: vec![ModelId("stub-model".into())],
                                ..spec
                            })
                            .expect("worker accepted"),
                        DiscoveryEvent::Removed { id } => {
                            registry.remove(&id);
                            notify_removed.notify_one();
                        }
                        _ => {}
                    }
                }
            });
            Self { ctx, removed }
        }
    }

    async fn send(ctx: &Arc<AppContext>, request: Request<Body>) -> (StatusCode, String) {
        let response = crate::server::app::build_router(Arc::clone(ctx))
            .oneshot(request)
            .await
            .unwrap();
        let status = response.status();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        (status, String::from_utf8(body.to_vec()).unwrap())
    }

    fn add_request() -> Request<Body> {
        Request::post("/workers")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"url": "http://10.0.0.1:30000", "worker_type": "regular"}"#,
            ))
            .unwrap()
    }

    #[test]
    fn a_worker_id_fits_one_path_segment() {
        assert_eq!(worker_id_for_url("http://10.0.0.1:30000/").0, ADDED_ID);
    }

    #[tokio::test]
    async fn an_added_worker_is_routed_to_listed_and_removed_by_its_id() {
        let FakeManager { ctx, removed } = FakeManager::start();

        let (added, _) = send(&ctx, add_request()).await;
        let (listed, workers) =
            send(&ctx, Request::get("/workers").body(Body::empty()).unwrap()).await;
        let (deleted, _) = send(
            &ctx,
            Request::delete(format!("/workers/{ADDED_ID}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        tokio::time::timeout(Duration::from_secs(5), removed.notified())
            .await
            .expect("the manager removes the worker");

        assert_eq!(
            (added, listed, deleted),
            (StatusCode::OK, StatusCode::OK, StatusCode::OK)
        );
        assert!(workers.contains(ADDED_ID), "{workers}");
        assert!(ctx.registry.get(&WorkerId(ADDED_ID.into())).is_none());
    }

    #[tokio::test]
    async fn a_router_without_the_worker_api_refuses_to_add_workers() {
        let ctx = Arc::new(AppContext::stub());

        let (status, body) = send(&ctx, add_request()).await;

        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert!(body.contains("--worker-api"), "{body}");
    }

    #[tokio::test]
    async fn an_unknown_worker_type_is_refused() {
        let FakeManager { ctx, removed: _ } = FakeManager::start();
        let request = Request::post("/workers")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"url": "http://10.0.0.1:30000", "worker_type": "sandy"}"#,
            ))
            .unwrap();

        let (status, body) = send(&ctx, request).await;

        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert!(body.contains("worker_type"), "{body}");
        assert!(ctx.registry.all().is_empty());
    }
}
