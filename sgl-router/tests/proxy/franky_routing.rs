// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Franky routing at the HTTP layer with mock engines: a Ricky worker that reads
//! taps gets a Sandy, and the Sandy and every Ricky leg carry one `franky_room`.

use crate::common::mock_worker::MockWorker;
use axum::body::Body;
use axum::extract::State;
use axum::http::{Request, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::Json;
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
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tower::ServiceExt;

const SANDY_PORT: u16 = 8996;
const PREFILL_PORT: u16 = 8997;

fn config(max_attempts: u32, policy: PolicyKind, franky_block_size: Option<u32>) -> Config {
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
            franky_block_size: franky_block_size.map(|size| NonZeroU32::new(size).unwrap()),
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
    franky_ctx_with_blocks(ricky, sandy, max_attempts, reorg, None, None)
}

fn franky_ctx_with_blocks(
    ricky: &[(&str, &str, WorkerMode)],
    sandy: &[(&str, &str)],
    max_attempts: u32,
    reorg: bool,
    franky_block_size: Option<u32>,
    franky_boundary_token_id: Option<u32>,
) -> Arc<AppContext> {
    let policy = if reorg {
        PolicyKind::PowerOfTwo
    } else {
        PolicyKind::RoundRobin
    };
    let mut cfg = config(max_attempts, policy, franky_block_size);
    cfg.proxy.franky_boundary_token_id = franky_boundary_token_id;
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

#[derive(Clone, Default)]
struct BlockEngine {
    stop_at: usize,
    prompt_len: Arc<Mutex<Option<usize>>>,
    requests: Arc<Mutex<Vec<Value>>>,
    chain_ends: Arc<Mutex<Vec<Value>>>,
    chain_ended: Arc<tokio::sync::Notify>,
}

impl BlockEngine {
    async fn start(stop_at: usize) -> (Self, String, Shutdown) {
        let engine = Self {
            stop_at,
            ..Default::default()
        };
        let app = axum::Router::new()
            .route("/generate", post(Self::generate))
            .route("/franky/end_chain", post(Self::end_chain))
            .with_state(engine.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let (shutdown, stopped) = tokio::sync::oneshot::channel::<()>();
        tokio::spawn(async move {
            axum::serve(listener, app)
                .with_graceful_shutdown(async {
                    let _ = stopped.await;
                })
                .await
                .unwrap();
        });
        (engine, url, shutdown)
    }

    async fn generate(State(engine): State<Self>, Json(request): Json<Value>) -> Response {
        engine.requests.lock().unwrap().push(request.clone());
        let input = request["input_ids"].as_array().unwrap().len();
        let prompt = *engine.prompt_len.lock().unwrap().get_or_insert(input);
        let generated = input - prompt;
        let max_new_tokens = request["sampling_params"]["max_new_tokens"]
            .as_u64()
            .unwrap() as usize;
        let count = max_new_tokens.min(engine.stop_at.saturating_sub(generated));
        let mut ids: Vec<usize> = (input..input + count).collect();
        let mut finish = if generated + count >= engine.stop_at {
            json!({"type": "stop", "matched": 0})
        } else {
            json!({"type": "length", "length": count})
        };
        let stops = request["sampling_params"]["stop_token_ids"].as_array();
        let stopped = stops.and_then(|stops| ids.iter().position(|&id| stops.contains(&json!(id))));
        if let Some(position) = stopped {
            ids.truncate(position + 1);
            finish = json!({"type": "stop", "matched": ids[position]});
        }
        let frame = |ids: &[usize], finish: Value| {
            json!({
                "text": ids.iter().map(|id| format!("<{id}>")).collect::<String>(),
                "output_ids": ids,
                "meta_info": {
                    "id": request["rid"],
                    "prompt_tokens": input,
                    "completion_tokens": ids.len(),
                    "finish_reason": finish,
                },
            })
        };
        if request["stream"] != json!(true) {
            return Json(frame(&ids, finish)).into_response();
        }
        let mut events: Vec<String> = (1..ids.len())
            .map(|end| format!("data: {}\n\n", frame(&ids[..end], Value::Null)))
            .collect();
        events.push(format!("data: {}\n\n", frame(&ids, finish)));
        events.push("data: [DONE]\n\n".into());
        let chunks = events.into_iter().map(Ok::<_, std::io::Error>);
        let mut response = Response::new(Body::from_stream(futures::stream::iter(chunks)));
        response.headers_mut().insert(
            "content-type",
            axum::http::HeaderValue::from_static("text/event-stream"),
        );
        response
    }

    async fn end_chain(State(engine): State<Self>, Json(request): Json<Value>) -> StatusCode {
        engine.chain_ends.lock().unwrap().push(request);
        engine.chain_ended.notify_one();
        StatusCode::OK
    }

    fn requests(&self) -> Vec<Value> {
        self.requests.lock().unwrap().clone()
    }

    fn chain_ends(&self) -> Vec<Value> {
        self.chain_ends.lock().unwrap().clone()
    }

    async fn wait_for_chain_end(&self) -> Vec<Value> {
        tokio::time::timeout(Duration::from_secs(5), self.chain_ended.notified())
            .await
            .expect("the engine got no chain end within 5 seconds");
        self.chain_ends()
    }
}

fn generate(body: Value) -> Request<Body> {
    Request::post("/generate")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

type Shutdown = tokio::sync::oneshot::Sender<()>;

async fn blocks_router(stop_at: usize) -> (Arc<AppContext>, [BlockEngine; 2], [Shutdown; 2]) {
    let (ricky, ricky_url, ricky_stop) = BlockEngine::start(stop_at).await;
    let (sandy, sandy_url, sandy_stop) = BlockEngine::start(usize::MAX).await;
    let ctx = franky_ctx_with_blocks(
        &[("r", &ricky_url, WorkerMode::Plain)],
        &[("s", &sandy_url)],
        1,
        false,
        Some(4),
        None,
    );
    (ctx, [ricky, sandy], [ricky_stop, sandy_stop])
}

/// A router running blocks of 4 that also end at `boundary`.
async fn boundary_router(
    stop_at: usize,
    boundary: u32,
) -> (Arc<AppContext>, [BlockEngine; 2], [Shutdown; 2]) {
    let (ricky, ricky_url, ricky_stop) = BlockEngine::start(stop_at).await;
    let (sandy, sandy_url, sandy_stop) = BlockEngine::start(usize::MAX).await;
    let ctx = franky_ctx_with_blocks(
        &[("r", &ricky_url, WorkerMode::Plain)],
        &[("s", &sandy_url)],
        1,
        false,
        Some(4),
        Some(boundary),
    );
    (ctx, [ricky, sandy], [ricky_stop, sandy_stop])
}

async fn answer_of(res: Response) -> Value {
    serde_json::from_slice(
        &axum::body::to_bytes(res.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap()
}

#[tokio::test]
async fn blocks_run_until_ricky_stops_and_answer_once() {
    let (ctx, [ricky, sandy], _stop) = blocks_router(10).await;
    let body = json!({"input_ids": [1, 2, 3], "sampling_params": {"max_new_tokens": 100}});

    let res = build_router(ctx).oneshot(generate(body)).await.unwrap();

    assert_eq!(res.status(), StatusCode::OK);
    let answer: Value = serde_json::from_slice(
        &axum::body::to_bytes(res.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(answer["output_ids"], json!((3..13).collect::<Vec<_>>()));
    assert_eq!(answer["meta_info"]["completion_tokens"], 10);
    assert_eq!(answer["meta_info"]["prompt_tokens"], 3);
    assert_eq!(answer["meta_info"]["finish_reason"]["type"], "stop");
    let requests = ricky.requests();
    assert_eq!(requests.len(), 3);
    let chain = requests[0]["franky_chain"]["id"]
        .as_str()
        .unwrap()
        .to_string();
    for (block, request) in requests.iter().enumerate() {
        let context: Vec<u64> = [1, 2, 3]
            .into_iter()
            .chain(3..3 + 4 * block as u64)
            .collect();
        assert_eq!(request["input_ids"], json!(context));
        assert_eq!(request["rid"], json!(format!("{chain}-{}", block + 1)));
        assert_eq!(
            request["franky_chain"],
            json!({"id": chain, "block": block + 1, "block_size": 4, "last_block": false})
        );
        assert_eq!(request["sampling_params"]["max_new_tokens"], 4);
    }
    assert_eq!(sandy.requests().len(), 3);
    assert_eq!(sandy.requests()[2]["input_ids"], requests[2]["input_ids"]);
    for engine in [&ricky, &sandy] {
        assert_eq!(
            engine.wait_for_chain_end().await,
            [json!({"chain_id": chain})]
        );
    }
}

#[tokio::test]
async fn a_chain_at_its_token_limit_marks_its_last_block() {
    let (ctx, [ricky, _sandy], _stop) = blocks_router(100).await;
    let body = json!({"input_ids": [1, 2, 3], "sampling_params": {"max_new_tokens": 6}});

    let res = build_router(ctx).oneshot(generate(body)).await.unwrap();

    assert_eq!(res.status(), StatusCode::OK);
    let requests = ricky.requests();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[1]["sampling_params"]["max_new_tokens"], 2);
    assert_eq!(requests[1]["franky_chain"]["last_block"], true);
    assert!(ricky.chain_ends().is_empty());
}

#[tokio::test]
async fn a_streamed_chain_is_one_stream() {
    let (ctx, [_ricky, _sandy], _stop) = blocks_router(6).await;
    let body =
        json!({"input_ids": [1, 2, 3], "stream": true, "sampling_params": {"max_new_tokens": 100}});

    let res = build_router(ctx).oneshot(generate(body)).await.unwrap();

    assert_eq!(res.status(), StatusCode::OK);
    let text = String::from_utf8(
        axum::body::to_bytes(res.into_body(), usize::MAX)
            .await
            .unwrap()
            .to_vec(),
    )
    .unwrap();
    let frames: Vec<Value> = text
        .split("\n\n")
        .filter_map(|event| event.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).unwrap())
        .collect();
    assert_eq!(text.matches("[DONE]").count(), 1);
    let lengths: Vec<_> = frames
        .iter()
        .map(|frame| frame["output_ids"].as_array().unwrap().len())
        .collect();
    assert_eq!(lengths, [1, 2, 3, 4, 5, 6]);
    let finishes: Vec<_> = frames
        .iter()
        .map(|frame| frame["meta_info"]["finish_reason"]["type"].clone())
        .collect();
    assert_eq!(
        finishes[..5],
        [
            Value::Null,
            Value::Null,
            Value::Null,
            Value::Null,
            Value::Null
        ]
    );
    assert_eq!(finishes[5], "stop");
    assert_eq!(frames[5]["text"], "<3><4><5><6><7><8>");
}

#[tokio::test]
async fn requests_blocks_cannot_serve_are_refused() {
    let (ctx, [ricky, _sandy], _stop) = blocks_router(10).await;
    for body in [
        json!({"input_ids": [1, 2], "sampling_params": {"max_new_tokens": 8, "n": 2}}),
        json!({"input_ids": [1, 2], "sampling_params": {"max_new_tokens": 8, "regex": "a+"}}),
        json!({"input_ids": [1, 2], "return_hidden_states": true}),
    ] {
        let res = build_router(Arc::clone(&ctx))
            .oneshot(generate(body.clone()))
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::BAD_REQUEST, "{body}");
    }
    let res = build_router(Arc::clone(&ctx))
        .oneshot(chat())
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    assert!(ricky.requests().is_empty());
}

/// Sandy reads her own view of a `/generate` prompt and Ricky his; the field
/// that carries hers reaches no engine.
#[tokio::test]
async fn a_sandy_reads_her_own_view_of_the_prompt() {
    let ricky = MockWorker::start(vec![]).await;
    let sandy = MockWorker::start(vec![]).await;
    let ctx = franky_ctx(
        &[("r", &ricky.url, WorkerMode::Plain)],
        &[("s", &sandy.url)],
        1,
        false,
    );
    let body = json!({
        "input_ids": [1, 2, 3, 4],
        "franky_sandy_input_ids": [1, 9, 9, 4],
        "sampling_params": {"max_new_tokens": 4},
    });

    let res = build_router(ctx).oneshot(generate(body)).await.unwrap();

    assert_eq!(res.status(), StatusCode::OK);
    let ricky_body = ricky.captured_json().await;
    let sandy_body = sandy.captured_json().await;
    assert_eq!(ricky_body["input_ids"], json!([1, 2, 3, 4]));
    assert_eq!(sandy_body["input_ids"], json!([1, 9, 9, 4]));
    for body in [&ricky_body, &sandy_body] {
        assert!(body.get("franky_sandy_input_ids").is_none(), "{body}");
    }
    assert_eq!(
        without(sandy_body, &["input_ids"]),
        without(ricky_body, &["input_ids"])
    );
}

/// Ricky's cached KV holds what his readers added, so it depends on Sandy's
/// view: one Ricky prompt under two views gets two cache keys on both engines,
/// the same view the same key, and a client's own key stays in front.
#[tokio::test]
async fn each_sandy_view_keys_both_engines_caches() {
    async fn keys(sandy_view: Value, client_key: Option<&str>) -> (String, String) {
        let ricky = MockWorker::start(vec![]).await;
        let sandy = MockWorker::start(vec![]).await;
        let ctx = franky_ctx(
            &[("r", &ricky.url, WorkerMode::Plain)],
            &[("s", &sandy.url)],
            1,
            false,
        );
        let mut body = json!({
            "input_ids": [1, 2, 3, 4],
            "franky_sandy_input_ids": sandy_view,
            "sampling_params": {"max_new_tokens": 4},
        });
        if let Some(key) = client_key {
            body["extra_key"] = key.into();
        }
        let res = build_router(ctx).oneshot(generate(body)).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let key = |body: Value| body["extra_key"].as_str().unwrap().to_owned();
        (
            key(ricky.captured_json().await),
            key(sandy.captured_json().await),
        )
    }

    let (ricky_first, sandy_first) = keys(json!([1, 9, 9, 4]), None).await;
    let (ricky_same, _) = keys(json!([1, 9, 9, 4]), None).await;
    let (ricky_other, _) = keys(json!([1, 8, 8, 4]), None).await;
    let (ricky_keyed, _) = keys(json!([1, 9, 9, 4]), Some("eval:3")).await;

    assert_eq!(ricky_first, sandy_first);
    assert_eq!(ricky_first, ricky_same);
    assert_ne!(ricky_first, ricky_other);
    assert_eq!(ricky_keyed, format!("eval:3|{ricky_first}"));
}

/// Every block gives Sandy her own view and every token Ricky generated so far,
/// and Ricky his view and the same tokens.
#[tokio::test]
async fn every_block_gives_sandy_her_view_and_rickys_tokens_so_far() {
    let (ctx, [ricky, sandy], _stop) = blocks_router(10).await;
    let body = json!({
        "input_ids": [1, 2, 3],
        "franky_sandy_input_ids": [7, 8, 9],
        "sampling_params": {"max_new_tokens": 100},
    });

    let res = build_router(ctx).oneshot(generate(body)).await.unwrap();

    assert_eq!(res.status(), StatusCode::OK);
    let (ricky_requests, sandy_requests) = (ricky.requests(), sandy.requests());
    assert_eq!((ricky_requests.len(), sandy_requests.len()), (3, 3));
    for (block, (ricky_request, sandy_request)) in
        ricky_requests.iter().zip(&sandy_requests).enumerate()
    {
        let generated: Vec<u64> = (3..3 + 4 * block as u64).collect();
        let ricky_context: Vec<u64> = [1, 2, 3].into_iter().chain(generated.clone()).collect();
        let sandy_context: Vec<u64> = [7, 8, 9].into_iter().chain(generated).collect();
        assert_eq!(ricky_request["input_ids"], json!(ricky_context));
        assert_eq!(sandy_request["input_ids"], json!(sandy_context));
        assert_eq!(sandy_request["franky_chain"], ricky_request["franky_chain"]);
        for request in [ricky_request, sandy_request] {
            assert!(request.get("franky_sandy_input_ids").is_none(), "{request}");
        }
    }
}

/// A view of another length, a prompt sent as text, or a view that is not token
/// ids is refused before any engine sees it.
#[tokio::test]
async fn a_sandy_view_that_cannot_align_with_rickys_prompt_is_refused() {
    let (ctx, [ricky, sandy], _stop) = blocks_router(10).await;
    for body in [
        json!({"input_ids": [1, 2, 3], "franky_sandy_input_ids": [7, 8]}),
        json!({"text": "one two three", "franky_sandy_input_ids": [7, 8, 9]}),
        json!({"input_ids": [1, 2, 3], "franky_sandy_input_ids": "seven eight nine"}),
    ] {
        let res = build_router(Arc::clone(&ctx))
            .oneshot(generate(body.clone()))
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::BAD_REQUEST, "{body}");
    }
    assert!(ricky.requests().is_empty());
    assert!(sandy.requests().is_empty());
}

/// The boundary token ends a block before its size and the next block starts a
/// new answer: a new chain whose blocks list the answer before it, with the old
/// chain ended on both engines; the client still gets one answer.
#[tokio::test]
async fn a_boundary_token_ends_an_answer_and_the_next_block_starts_another() {
    let (ctx, [ricky, sandy], _stop) = boundary_router(10, 5).await;
    let body = json!({
        "input_ids": [1, 2, 3],
        "franky_sandy_input_ids": [7, 8, 9],
        "sampling_params": {"max_new_tokens": 100},
    });

    let res = build_router(ctx).oneshot(generate(body)).await.unwrap();

    assert_eq!(res.status(), StatusCode::OK);
    let answer = answer_of(res).await;
    assert_eq!(answer["output_ids"], json!((3..13).collect::<Vec<_>>()));
    assert_eq!(
        answer["meta_info"]["finish_reason"],
        json!({"type": "stop", "matched": 0})
    );
    let requests = ricky.requests();
    let chains: Vec<_> = requests
        .iter()
        .map(|request| {
            (
                request["franky_chain"]["id"].clone(),
                request["franky_chain"]["block"].clone(),
            )
        })
        .collect();
    assert_eq!(requests.len(), 3);
    let (first, second) = (chains[0].0.clone(), chains[1].0.clone());
    assert_ne!(first, second);
    assert_eq!(
        chains,
        [
            (first.clone(), json!(1)),
            (second.clone(), json!(1)),
            (second.clone(), json!(2))
        ]
    );
    assert_eq!(
        requests[1]["rid"],
        json!(format!("{}-1", second.as_str().unwrap()))
    );
    assert!(requests[0].get("franky_generated_spans").is_none());
    for request in &requests[1..] {
        assert_eq!(request["franky_generated_spans"], json!([[3, 6]]));
    }
    assert_eq!(requests[1]["input_ids"], json!([1, 2, 3, 3, 4, 5]));
    for request in &requests {
        assert_eq!(request["sampling_params"]["stop_token_ids"], json!([5]));
    }
    let sandy_requests = sandy.requests();
    assert_eq!(sandy_requests[1]["input_ids"], json!([7, 8, 9, 3, 4, 5]));
    assert_eq!(
        sandy_requests[1]["franky_chain"],
        requests[1]["franky_chain"]
    );
    assert_eq!(sandy_requests[1]["franky_generated_spans"], json!([[3, 6]]));
    for engine in [&ricky, &sandy] {
        let ended = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let ends = engine.chain_ends();
                if ends.len() == 2 {
                    return ends;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("both chains end on the engine");
        assert_eq!(
            ended,
            [json!({"chain_id": first}), json!({"chain_id": second})]
        );
    }
}

/// A client that stops on the boundary token ends its answer there.
#[tokio::test]
async fn a_client_that_stops_on_the_boundary_token_ends_its_answer_there() {
    let (ctx, [ricky, _sandy], _stop) = boundary_router(10, 5).await;
    let body = json!({
        "input_ids": [1, 2, 3],
        "sampling_params": {"max_new_tokens": 100, "stop_token_ids": [5]},
    });

    let res = build_router(ctx).oneshot(generate(body)).await.unwrap();

    assert_eq!(res.status(), StatusCode::OK);
    let answer = answer_of(res).await;
    assert_eq!(answer["output_ids"], json!([3, 4, 5]));
    assert_eq!(
        answer["meta_info"]["finish_reason"],
        json!({"type": "stop", "matched": 5})
    );
    let requests = ricky.requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0]["sampling_params"]["stop_token_ids"], json!([5]));
}

/// ignore_eos turns stops off, so a request asking for it under a boundary
/// token is refused.
#[tokio::test]
async fn ignore_eos_under_a_boundary_token_is_refused() {
    let (ctx, [ricky, _sandy], _stop) = boundary_router(10, 5).await;
    let body = json!({"input_ids": [1, 2, 3], "sampling_params": {"max_new_tokens": 8, "ignore_eos": true}});

    let res = build_router(ctx).oneshot(generate(body)).await.unwrap();

    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    assert!(ricky.requests().is_empty());
}
