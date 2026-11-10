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
    franky_ctx_with_blocks(ricky, sandy, max_attempts, reorg, None)
}

fn franky_ctx_with_blocks(
    ricky: &[(&str, &str, WorkerMode)],
    sandy: &[(&str, &str)],
    max_attempts: u32,
    reorg: bool,
    franky_block_size: Option<u32>,
) -> Arc<AppContext> {
    let policy = if reorg {
        PolicyKind::PowerOfTwo
    } else {
        PolicyKind::RoundRobin
    };
    let cfg = config(max_attempts, policy, franky_block_size);
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

/// An engine for blocks: a Ricky generates the next ids after its prompt, up to
/// the block's `max_new_tokens`, and stops once the chain generated `stop_at`;
/// a Sandy answers no tokens. Both record every request and chain end.
#[derive(Clone, Default)]
struct BlockEngine {
    stop_at: usize,
    prompt_len: Arc<Mutex<Option<usize>>>,
    requests: Arc<Mutex<Vec<Value>>>,
    chain_ends: Arc<Mutex<Vec<Value>>>,
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
        let ids: Vec<usize> = (input..input + count).collect();
        let finish = if generated + count >= engine.stop_at {
            json!({"type": "stop", "matched": 0})
        } else {
            json!({"type": "length", "length": count})
        };
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
        StatusCode::OK
    }

    fn requests(&self) -> Vec<Value> {
        self.requests.lock().unwrap().clone()
    }

    async fn chain_ends(&self) -> Vec<Value> {
        // The router ends a chain in the background once the response is done.
        for _ in 0..50 {
            let ends = self.chain_ends.lock().unwrap().clone();
            if !ends.is_empty() {
                return ends;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        Vec::new()
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
    );
    (ctx, [ricky, sandy], [ricky_stop, sandy_stop])
}

/// Each block sends the prompt and everything generated so far, under one
/// chain, to the same Ricky and Sandy, until Ricky stops; the client gets one
/// answer, and both engines are told to end the chain.
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
            json!({"id": chain, "block": block + 1, "last_block": false})
        );
        assert_eq!(request["sampling_params"]["max_new_tokens"], 4);
    }
    assert_eq!(sandy.requests().len(), 3);
    assert_eq!(sandy.requests()[2]["input_ids"], requests[2]["input_ids"]);
    for engine in [&ricky, &sandy] {
        assert_eq!(engine.chain_ends().await, [json!({"chain_id": chain})]);
    }
}

/// A chain that reaches its token limit marks its last block, after which no
/// engine holds anything, so the chain needs no end.
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
    assert!(ricky.chain_ends().await.is_empty());
}

/// A streamed chain is one stream: frames continue across blocks, a block cut
/// has no finish reason, and one `[DONE]` ends it.
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

/// Blocks refuse what they cannot join or would restart.
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
