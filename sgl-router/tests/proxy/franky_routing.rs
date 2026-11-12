// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Franky routing at the HTTP layer with mock engines: a Ricky worker that reads
//! taps gets a Sandy, and the Sandy and every Ricky leg carry one `franky_room`.

use crate::common::mock_worker::MockWorker;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{json, Value};
use sgl_router::config::{
    Config, DiscoveryBackend, InflightLoadConfig, ModelConfig, ObservabilityConfig, PolicyKind,
    ProxyConfig, ServerConfig, StaticUrlsDiscoveryConfig,
};
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::factory::build_registry_with_defaults;
use sgl_router::policies_reorg::factory::{build_resolver, build_sandy_group};
use sgl_router::proxy::Proxy;
use sgl_router::server::app::build_router;
use sgl_router::server::app_context::{AppContext, ChatRouting};
use sgl_router::state::kv_events::KvEventIndex;
use sgl_router::tokenizer::TokenizerRegistry;
use sgl_router::workers::{EngineProfile, WireProtocol, WorkerRegistry};
use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::Duration;
use tower::ServiceExt;

const SANDY_PORT: u16 = 8996;
const PREFILL_PORT: u16 = 8997;

fn config(max_attempts: u32, policy: PolicyKind) -> Config {
    Config {
        server: ServerConfig {
            host: "0".into(),
            port: 0,
            ..Default::default()
        },
        observability: ObservabilityConfig::default(),
        model: ModelConfig {
            id: "tiny".into(),
            tokenizer_path: Some("tests/fixtures/tiny_tokenizer.json".into()),
            disable_input_ids_forwarding: false,
            tokenizer: Default::default(),
            policy,
            decode_policy: Default::default(),
            dp_aware: false,
            dp_rank_policy: Default::default(),
            bucket_config: None,
            reorg_buckets: None,
            reorg_admission: Default::default(),
            circuit_breaker: None,
            cache_aware: None,
            sticky: None,
            affinity: None,
            fused: None,
            eligibility: None,
            sampling_overrides: Default::default(),
            default_chat_template_kwargs: Default::default(),
        },
        discovery: DiscoveryBackend::StaticUrls(StaticUrlsDiscoveryConfig {
            urls: vec!["http://placeholder:0".into()],
        }),
        proxy: ProxyConfig {
            max_attempts: NonZeroU32::new(max_attempts).unwrap(),
            ..Default::default()
        },
        router_inflight_load: InflightLoadConfig::default(),
    }
}

/// A router over Ricky workers that read taps and over Sandy workers, with
/// legacy round-robin or reorg power-of-two routing.
fn franky_ctx(
    ricky: &[(&str, &str, WorkerMode)],
    sandy: &[(&str, &str)],
    max_attempts: u32,
    reorg: bool,
) -> Arc<AppContext> {
    let policy = if reorg {
        PolicyKind::PowerOfTwo
    } else {
        PolicyKind::RoundRobin
    };
    let cfg = config(max_attempts, policy);
    let registry = Arc::new(WorkerRegistry::default());
    for (id, url, mode) in ricky {
        let spec = WorkerSpec {
            id: WorkerId((*id).into()),
            url: (*url).into(),
            mode: *mode,
            model_ids: vec![ModelId("tiny".into())],
            bootstrap_port: (*mode == WorkerMode::Prefill).then_some(PREFILL_PORT),
            ..Default::default()
        };
        let profile = EngineProfile {
            needs_taps: true,
            ..EngineProfile::from(WireProtocol::default())
        };
        registry.add_with_cb(spec, None, profile).unwrap();
    }
    for (id, url) in sandy {
        let spec = WorkerSpec {
            id: WorkerId((*id).into()),
            url: (*url).into(),
            mode: WorkerMode::Sandy,
            model_ids: vec![ModelId("tiny".into())],
            bootstrap_port: Some(SANDY_PORT),
            ..Default::default()
        };
        registry.add(spec).unwrap();
    }
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let policies = Arc::new(build_registry_with_defaults(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(Duration::from_secs(5)).unwrap());
    let mut ctx = AppContext::new(cfg, tokenizers, proxy, registry, policies);
    let state = KvEventIndex::new();
    let sandy_group = if reorg {
        let (resolver, sandy_group, _) = build_resolver(&ctx.config.model, &state, None).unwrap();
        ctx.chat_routing = ChatRouting::Reorg([(ModelId("tiny".into()), resolver)].into());
        sandy_group
    } else {
        build_sandy_group(&ctx.config.model, &state, None)
            .unwrap()
            .0
    };
    ctx.sandy = Some(sandy_group);
    Arc::new(ctx)
}

fn chat() -> Request<Body> {
    Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({"model": "tiny", "messages": [{"role": "user", "content": "hi"}]}).to_string(),
        ))
        .unwrap()
}

fn without(mut body: Value, fields: &[&str]) -> Value {
    for field in fields {
        body.as_object_mut().unwrap().remove(*field);
    }
    body
}

/// Request outcomes recorded against the worker at `url` in `mode`, each as
/// `"<outcome> <count>"`.
fn outcomes(ctx: &AppContext, url: &str, mode: &str) -> Vec<String> {
    let prefix = format!(
        r#"sgl_router_worker_requests_total{{worker_url="{url}",model_id="tiny",mode="{mode}",outcome=""#
    );
    let metrics = ctx.metrics.render();
    metrics
        .lines()
        .filter_map(|line| line.strip_prefix(&prefix))
        .map(|rest| rest.replace("\"}", ""))
        .collect()
}

/// Sandy gets Ricky's own prompt, and both get one room on Sandy's tap port.
#[tokio::test]
async fn a_plain_ricky_and_its_sandy_share_one_room() {
    for reorg in [false, true] {
        let ricky = MockWorker::start(vec![]).await;
        let sandy = MockWorker::start(vec![]).await;
        let ctx = franky_ctx(
            &[("r", &ricky.url, WorkerMode::Plain)],
            &[("s", &sandy.url)],
            1,
            reorg,
        );
        let res = build_router(ctx).oneshot(chat()).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK, "reorg={reorg}");

        let ricky_body = ricky.captured_json().await;
        let sandy_body = sandy.captured_json().await;
        let room = &ricky_body["franky_room"];
        assert_eq!(room["bootstrap_host"], json!("127.0.0.1"), "reorg={reorg}");
        assert_eq!(room["bootstrap_port"], json!(SANDY_PORT), "reorg={reorg}");
        assert!(room["room"].as_u64().unwrap() <= i64::MAX as u64);
        assert_eq!(&sandy_body["franky_room"], room, "reorg={reorg}");
        assert_eq!(sandy_body, ricky_body, "reorg={reorg}");
    }
}

/// Both PD legs carry the Sandy's room next to their own bootstrap room, and
/// Sandy computes the same prompt outside PD.
#[tokio::test]
async fn pd_legs_and_their_sandy_share_one_room() {
    for reorg in [false, true] {
        let prefill = MockWorker::start(vec![]).await;
        let decode = MockWorker::start(vec![]).await;
        let sandy = MockWorker::start(vec![]).await;
        let ctx = franky_ctx(
            &[
                ("p", &prefill.url, WorkerMode::Prefill),
                ("d", &decode.url, WorkerMode::Decode),
            ],
            &[("s", &sandy.url)],
            1,
            reorg,
        );
        let res = build_router(ctx).oneshot(chat()).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK, "reorg={reorg}");

        let prefill_body = prefill.captured_json().await;
        let decode_body = decode.captured_json().await;
        let sandy_body = sandy.captured_json().await;
        assert_eq!(prefill_body, decode_body, "reorg={reorg}");
        assert_eq!(prefill_body["bootstrap_port"], json!(PREFILL_PORT));
        assert_eq!(
            prefill_body["franky_room"]["bootstrap_port"],
            json!(SANDY_PORT)
        );
        let pd_fields = ["bootstrap_host", "bootstrap_port", "bootstrap_room"];
        assert_eq!(
            sandy_body,
            without(prefill_body, &pd_fields),
            "reorg={reorg}"
        );
    }
}

/// A Ricky that reads taps never runs without them: with no Sandy the request
/// is refused before any worker sees it.
#[tokio::test]
async fn a_ricky_that_reads_taps_needs_a_sandy() {
    for reorg in [false, true] {
        let ricky = MockWorker::start(vec![]).await;
        let ctx = franky_ctx(&[("r", &ricky.url, WorkerMode::Plain)], &[], 1, reorg);
        let res = build_router(ctx).oneshot(chat()).await.unwrap();
        assert_eq!(
            res.status(),
            StatusCode::SERVICE_UNAVAILABLE,
            "reorg={reorg}"
        );
        assert_eq!(
            res.headers().get("x-router-error-code").unwrap(),
            "no_sandy_workers_available"
        );
        assert!(ricky.captured.lock().unwrap().last_body.is_none());
    }
}

/// A Sandy failure ends the request as a PD prefill failure does, and the
/// retry never sends the request to that Sandy again.
#[tokio::test]
async fn a_failed_sandy_is_blamed_and_not_retried() {
    for reorg in [false, true] {
        // Ricky answers after Sandy fails, as a real one waits for the taps; the
        // second Ricky leaves the retry a worker to pair with that Sandy again.
        let ricky = MockWorker::start_hanging(Duration::from_millis(500)).await;
        let other_ricky = MockWorker::start_hanging(Duration::from_millis(500)).await;
        let sandy =
            MockWorker::start_returning_error(StatusCode::INTERNAL_SERVER_ERROR, json!({})).await;
        let ctx = franky_ctx(
            &[
                ("r", &ricky.url, WorkerMode::Plain),
                ("r2", &other_ricky.url, WorkerMode::Plain),
            ],
            &[("s", &sandy.url)],
            2,
            reorg,
        );
        let res = build_router(Arc::clone(&ctx))
            .oneshot(chat())
            .await
            .unwrap();
        assert_eq!(
            res.headers().get("x-router-error-code").unwrap(),
            "prefill_failed",
            "reorg={reorg}"
        );
        assert_eq!(
            outcomes(&ctx, &sandy.url, "sandy"),
            ["error 1"],
            "reorg={reorg}"
        );
    }
}
