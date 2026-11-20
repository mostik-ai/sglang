// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Franky routing at the HTTP layer with mock engines: a Ricky worker that reads
//! taps gets a Sandy, the Sandy and every Ricky leg carry one `franky_room`,
//! and chains run Sandy calls and Ricky legs over one trace.

use crate::common::mock_worker::MockWorker;
use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, Request, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::Json;
use serde_json::{json, Value};
use sgl_router::config::{
    Config, DiscoveryBackend, FrankyChainConfig, InflightLoadConfig, ModelConfig,
    ObservabilityConfig, PolicyKind, ProxyConfig, ServerConfig, StaticUrlsDiscoveryConfig,
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

fn config(max_attempts: u32, policy: PolicyKind, franky_chain: FrankyChainConfig) -> Config {
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
            franky_chain,
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
    franky_ctx_with_chain(
        ricky,
        sandy,
        max_attempts,
        reorg,
        FrankyChainConfig::default(),
    )
}

fn franky_ctx_with_chain(
    ricky: &[(&str, &str, WorkerMode)],
    sandy: &[(&str, &str)],
    max_attempts: u32,
    reorg: bool,
    franky_chain: FrankyChainConfig,
) -> Arc<AppContext> {
    let policy = if reorg {
        PolicyKind::PowerOfTwo
    } else {
        PolicyKind::RoundRobin
    };
    let cfg = config(max_attempts, policy, franky_chain);
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

fn with_request_id(mut request: Request<Body>, request_id: Option<&str>) -> Request<Body> {
    if let Some(request_id) = request_id {
        request
            .headers_mut()
            .insert("x-request-id", request_id.parse().unwrap());
    }
    request
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

/// Sandy gets Ricky's own prompt as a leg that writes nothing, and both get
/// one room on Sandy's tap port.
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
        assert_eq!(sandy_body["franky_taps_only"], true, "reorg={reorg}");
        assert_eq!(
            without(sandy_body, &["franky_taps_only"]),
            ricky_body,
            "reorg={reorg}"
        );
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
            without(sandy_body, &["franky_taps_only"]),
            without(prefill_body, &pd_fields),
            "reorg={reorg}"
        );
    }
}

/// The client's `x-request-id` reaches the Sandy and both PD legs, so each
/// engine's record names the client's request; without one, none is sent.
#[tokio::test]
async fn every_leg_carries_the_clients_request_id() {
    for request_id in [Some("client-7"), None] {
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
            true,
        );

        let res = build_router(ctx)
            .oneshot(with_request_id(chat(), request_id))
            .await
            .unwrap();

        assert_eq!(res.status(), StatusCode::OK);
        for worker in [&prefill, &decode, &sandy] {
            worker.captured_json().await;
            let captured = worker.captured.lock().unwrap();
            assert_eq!(
                captured.headers.get("x-request-id").map(String::as_str),
                request_id,
                "{}",
                worker.url
            );
        }
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

/// A mock engine of a Franky chain over a prompt of `prompt_len` tokens. Each
/// token it writes is its position in the trace. As Ricky it ends a leg for a
/// Sandy call at the end of every `every_blocks`-th block of the chain and
/// stops once the answer has `stop_at` tokens; as Sandy it writes what a call
/// asks and nothing for a tap leg.
#[derive(Clone, Default)]
struct ChainEngine {
    prompt_len: usize,
    stop_at: usize,
    requests: Arc<Mutex<Vec<Value>>>,
    request_ids: Arc<Mutex<Vec<Option<String>>>>,
    request_came: Arc<tokio::sync::Notify>,
    chain_ends: Arc<Mutex<Vec<Value>>>,
    chain_ended: Arc<tokio::sync::Notify>,
}

const PROMPT: [u64; 3] = [1, 2, 3];

impl ChainEngine {
    async fn start(stop_at: usize) -> (Self, String, Shutdown) {
        let engine = Self {
            prompt_len: PROMPT.len(),
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

    /// How many tokens a request writes after `written` tokens of the answer,
    /// and why it stops.
    fn write(&self, request: &Value, written: usize) -> (usize, Value) {
        if request["franky_taps_only"] == json!(true) {
            return (0, json!({"type": "length", "length": 0}));
        }
        let max_new_tokens = request["sampling_params"]["max_new_tokens"]
            .as_u64()
            .map_or(usize::MAX, |max| max as usize);
        let to_stop = self.stop_at.saturating_sub(written);
        let count = max_new_tokens.min(to_stop);
        let chain = &request["franky_chain"];
        if let Some(block_size) = chain["blocks"]["block_size"].as_u64() {
            let every_blocks = chain["blocks"]["decider"]["every_blocks"]
                .as_u64()
                .unwrap_or(1);
            let first_block = chain["block"].as_u64().unwrap();
            let call_block = first_block.next_multiple_of(every_blocks);
            let call_at = ((call_block - first_block + 1) * block_size) as usize;
            if call_at < count {
                let next_block = call_block + 1;
                return (
                    call_at,
                    json!({"type": "franky_sandy_call", "next_block": next_block}),
                );
            }
        }
        if count == to_stop {
            (count, json!({"type": "stop", "matched": 0}))
        } else {
            (count, json!({"type": "length", "length": count}))
        }
    }

    async fn generate(
        State(engine): State<Self>,
        headers: HeaderMap,
        Json(request): Json<Value>,
    ) -> Response {
        engine.requests.lock().unwrap().push(request.clone());
        let request_id = headers
            .get("x-request-id")
            .map(|id| id.to_str().unwrap().to_owned());
        engine.request_ids.lock().unwrap().push(request_id);
        engine.request_came.notify_one();
        let input = request["input_ids"].as_array().unwrap().len();
        let (count, finish) = engine.write(&request, input - engine.prompt_len);
        let ids: Vec<usize> = (input..input + count).collect();
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

    /// The `x-request-id` of each request, once `count` requests came: a
    /// Sandy's leg may still be on its way when the chain answers.
    async fn wait_for_request_ids(&self, count: usize) -> Vec<Option<String>> {
        let came = async {
            while self.request_ids.lock().unwrap().len() < count {
                self.request_came.notified().await;
            }
        };
        tokio::time::timeout(Duration::from_secs(5), came)
            .await
            .expect("the engine got too few requests within 5 seconds");
        self.request_ids.lock().unwrap().clone()
    }

    async fn wait_for_chain_end(&self) -> Vec<Value> {
        let ended = async {
            while self.chain_ends.lock().unwrap().is_empty() {
                self.chain_ended.notified().await;
            }
        };
        tokio::time::timeout(Duration::from_secs(5), ended)
            .await
            .expect("the engine got no chain end within 5 seconds");
        self.chain_ends.lock().unwrap().clone()
    }
}

fn generate(body: Value) -> Request<Body> {
    Request::post("/generate")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

type Shutdown = tokio::sync::oneshot::Sender<()>;

fn blocks(block_size: u64, every_blocks: u64) -> serde_json::Map<String, Value> {
    let blocks = json!({
        "block_size": block_size,
        "decider": {"type": "call_token_or_every_blocks", "every_blocks": every_blocks},
    });
    blocks.as_object().unwrap().clone()
}

fn chain_config(sandy_call_tokens: u32, taps: bool, block_size: u64) -> FrankyChainConfig {
    FrankyChainConfig {
        sandy_call_tokens,
        taps,
        blocks: Some(blocks(block_size, 1)),
    }
}

/// A chain router over one Ricky that stops after `stop_at` tokens of the
/// answer and one Sandy.
async fn chain_router(
    stop_at: usize,
    chain: FrankyChainConfig,
) -> (Arc<AppContext>, [ChainEngine; 2], [Shutdown; 2]) {
    let (ricky, ricky_url, ricky_stop) = ChainEngine::start(stop_at).await;
    let (sandy, sandy_url, sandy_stop) = ChainEngine::start(usize::MAX).await;
    let ctx = franky_ctx_with_chain(
        &[("r", &ricky_url, WorkerMode::Plain)],
        &[("s", &sandy_url)],
        1,
        false,
        chain,
    );
    (ctx, [ricky, sandy], [ricky_stop, sandy_stop])
}

async fn json_answer(response: Response) -> Value {
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    serde_json::from_slice(&body).unwrap()
}

async fn streamed_frames(response: Response) -> (Vec<Value>, usize) {
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();
    let frames = text
        .split("\n\n")
        .filter_map(|event| event.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).unwrap())
        .collect();
    (frames, text.matches("[DONE]").count())
}

fn trace(answer: std::ops::Range<u64>) -> Value {
    json!(PROMPT.into_iter().chain(answer).collect::<Vec<_>>())
}

/// With Sandy writing nothing, each Ricky leg writes a block beside a Sandy
/// tap leg over the trace, and reads all of its taps.
#[tokio::test]
async fn ricky_blocks_read_the_taps_of_a_sandy_leg_over_the_trace() {
    let (ctx, [ricky, sandy], _stop) = chain_router(10, chain_config(0, true, 4)).await;
    let body = json!({"input_ids": PROMPT, "sampling_params": {"max_new_tokens": 100}});

    let res = build_router(ctx).oneshot(generate(body)).await.unwrap();

    assert_eq!(res.status(), StatusCode::OK);
    let answer = json_answer(res).await;
    assert_eq!(answer["output_ids"], json!((3..13).collect::<Vec<_>>()));
    assert_eq!(answer["meta_info"]["completion_tokens"], 10);
    assert_eq!(answer["meta_info"]["prompt_tokens"], 3);
    assert_eq!(answer["meta_info"]["finish_reason"]["type"], "stop");
    let legs = ricky.requests();
    assert_eq!(legs.len(), 3);
    let chain = legs[0]["franky_chain"]["id"].as_str().unwrap().to_string();
    let mut runs = Vec::new();
    for (leg, request) in legs.iter().enumerate() {
        let start = 3 + 4 * leg as u64;
        runs.push(json!([start, start]));
        assert_eq!(request["input_ids"], trace(3..start));
        assert_eq!(request["rid"], json!(format!("{chain}-r{}", leg + 1)));
        assert_eq!(
            request["franky_chain"],
            json!({"id": chain, "block": leg + 1, "blocks": blocks(4, 1)})
        );
        assert_eq!(request["franky_tap_runs"], json!(runs));
        assert_eq!(
            request["sampling_params"]["max_new_tokens"],
            100 - 4 * leg as u64
        );
    }
    let tap_legs = sandy.requests();
    assert_eq!(tap_legs.len(), 3);
    for (tap_leg, leg) in tap_legs.iter().zip(&legs) {
        assert_eq!(tap_leg["input_ids"], leg["input_ids"]);
        assert_eq!(tap_leg["franky_room"], leg["franky_room"]);
        assert_eq!(tap_leg["franky_taps_only"], true);
        assert_eq!(
            tap_leg["franky_chain"],
            json!({"id": chain, "block": leg["franky_chain"]["block"]})
        );
        assert!(tap_leg.get("franky_tap_runs").is_none());
    }
    for engine in [&ricky, &sandy] {
        assert_eq!(
            engine.wait_for_chain_end().await,
            [json!({"chain_id": chain})]
        );
    }
}

/// Sandy writes before each leg; the leg reads the room of Sandy's call, whose
/// taps cover the trace and Sandy's text, from the text's start on.
#[tokio::test]
async fn sandy_writes_before_each_leg_and_the_leg_reads_its_room() {
    let (ctx, [ricky, sandy], _stop) = chain_router(12, chain_config(2, true, 4)).await;
    let body = json!({"input_ids": PROMPT, "sampling_params": {"max_new_tokens": 100}});

    let res = build_router(ctx).oneshot(generate(body)).await.unwrap();

    assert_eq!(res.status(), StatusCode::OK);
    let answer = json_answer(res).await;
    assert_eq!(answer["output_ids"], json!((3..15).collect::<Vec<_>>()));
    assert_eq!(
        answer["text"],
        (3..15).map(|id| format!("<{id}>")).collect::<String>()
    );
    assert_eq!(answer["meta_info"]["completion_tokens"], 12);
    assert_eq!(answer["meta_info"]["prompt_tokens"], 3);
    assert_eq!(answer["meta_info"]["finish_reason"]["type"], "stop");
    let calls = sandy.requests();
    let legs = ricky.requests();
    assert_eq!((calls.len(), legs.len()), (2, 2));
    let chain = calls[0]["franky_chain"]["id"].as_str().unwrap().to_string();
    assert_eq!(calls[0]["input_ids"], trace(3..3));
    assert_eq!(calls[1]["input_ids"], trace(3..9));
    assert_eq!(legs[0]["input_ids"], trace(3..5));
    assert_eq!(legs[1]["input_ids"], trace(3..11));
    assert_eq!(legs[0]["franky_tap_runs"], json!([[3, 5]]));
    assert_eq!(legs[1]["franky_tap_runs"], json!([[3, 5], [9, 11]]));
    for (index, (call, leg)) in calls.iter().zip(&legs).enumerate() {
        let block = [1, 2][index];
        assert_eq!(call["rid"], json!(format!("{chain}-s{}", 2 * index + 1)));
        assert_eq!(leg["rid"], json!(format!("{chain}-r{}", 2 * index + 2)));
        assert_eq!(call["franky_chain"], json!({"id": chain, "block": block}));
        assert_eq!(
            leg["franky_chain"],
            json!({"id": chain, "block": block, "blocks": blocks(4, 1)})
        );
        assert_eq!(call["stream"], false);
        assert_eq!(call["sampling_params"]["max_new_tokens"], 2);
        assert!(call.get("franky_tap_runs").is_none());
        assert!(call["franky_room"]["room"].is_u64());
        assert_eq!(leg["franky_room"], call["franky_room"]);
        assert!(leg.get("franky_taps_only").is_none());
    }
    assert_ne!(calls[0]["franky_room"], calls[1]["franky_room"]);
    assert_eq!(legs[0]["sampling_params"]["max_new_tokens"], 98);
    assert_eq!(legs[1]["sampling_params"]["max_new_tokens"], 92);
    for engine in [&ricky, &sandy] {
        assert_eq!(
            engine.wait_for_chain_end().await,
            [json!({"chain_id": chain})]
        );
    }
}

/// A Sandy call that writes all that is left of the answer ends it.
#[tokio::test]
async fn sandys_text_at_the_answers_limit_ends_the_answer() {
    let (ctx, [ricky, sandy], _stop) = chain_router(100, chain_config(4, true, 4)).await;
    let body = json!({"input_ids": PROMPT, "sampling_params": {"max_new_tokens": 3}});

    let res = build_router(ctx).oneshot(generate(body)).await.unwrap();

    assert_eq!(res.status(), StatusCode::OK);
    let answer = json_answer(res).await;
    assert_eq!(answer["output_ids"], json!([3, 4, 5]));
    assert_eq!(answer["meta_info"]["finish_reason"]["type"], "length");
    assert_eq!(answer["meta_info"]["completion_tokens"], 3);
    assert_eq!(sandy.requests()[0]["sampling_params"]["max_new_tokens"], 3);
    assert!(ricky.requests().is_empty());
    sandy.wait_for_chain_end().await;
}

/// Without taps no request carries a room or tap runs.
#[tokio::test]
async fn text_only_chains_send_no_rooms_or_tap_runs() {
    let (ctx, [ricky, sandy], _stop) = chain_router(12, chain_config(2, false, 4)).await;
    let body = json!({"input_ids": PROMPT, "sampling_params": {"max_new_tokens": 100}});

    let res = build_router(ctx).oneshot(generate(body)).await.unwrap();

    assert_eq!(res.status(), StatusCode::OK);
    assert_eq!(
        json_answer(res).await["output_ids"],
        json!((3..15).collect::<Vec<_>>())
    );
    for request in sandy.requests().iter().chain(&ricky.requests()) {
        for field in ["franky_room", "franky_tap_runs", "franky_taps_only"] {
            assert!(request.get(field).is_none(), "{field} in {request}");
        }
    }
}

/// A streamed chain is one stream: Sandy's text comes as one frame, every
/// frame carries the answer so far, and only the last one finishes it.
#[tokio::test]
async fn a_streamed_chain_with_sandys_text_is_one_stream() {
    let (ctx, [_ricky, _sandy], _stop) = chain_router(12, chain_config(2, true, 4)).await;
    let body = json!({
        "input_ids": PROMPT,
        "stream": true,
        "sampling_params": {"max_new_tokens": 100},
    });

    let res = build_router(ctx).oneshot(generate(body)).await.unwrap();

    assert_eq!(res.status(), StatusCode::OK);
    assert_eq!(res.headers()["content-type"], "text/event-stream");
    let (frames, done) = streamed_frames(res).await;
    assert_eq!(done, 1);
    let lengths: Vec<_> = frames
        .iter()
        .map(|frame| frame["output_ids"].as_array().unwrap().len())
        .collect();
    assert_eq!(lengths, [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]);
    let last = frames.len() - 1;
    for frame in &frames[..last] {
        assert_eq!(frame["meta_info"]["finish_reason"], Value::Null, "{frame}");
    }
    assert_eq!(frames[last]["meta_info"]["finish_reason"]["type"], "stop");
    assert_eq!(frames[last]["meta_info"]["completion_tokens"], 12);
    assert_eq!(
        frames[last]["text"],
        (3..15).map(|id| format!("<{id}>")).collect::<String>()
    );
}

#[tokio::test]
async fn every_request_of_a_chain_carries_the_clients_request_id() {
    for request_id in [Some("client-7"), None] {
        let (ctx, [ricky, sandy], _stop) = chain_router(10, chain_config(0, true, 4)).await;
        let body = json!({"input_ids": PROMPT, "sampling_params": {"max_new_tokens": 100}});

        let res = build_router(ctx)
            .oneshot(with_request_id(generate(body), request_id))
            .await
            .unwrap();

        assert_eq!(res.status(), StatusCode::OK);
        let expected = vec![request_id.map(str::to_owned); 3];
        assert_eq!(ricky.wait_for_request_ids(3).await, expected);
        assert_eq!(sandy.wait_for_request_ids(3).await, expected);
    }
}

#[tokio::test]
async fn requests_chains_cannot_serve_are_refused() {
    let (ctx, [ricky, sandy], _stop) = chain_router(10, chain_config(2, true, 4)).await;
    for body in [
        json!({"input_ids": [1, 2], "sampling_params": {"max_new_tokens": 8, "n": 2}}),
        json!({"input_ids": [1, 2], "sampling_params": {"max_new_tokens": 8, "regex": "a+"}}),
        json!({"input_ids": [1, 2], "return_hidden_states": true}),
        json!({"input_ids": [1, 2], "return_logprob": true}),
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
    assert!(sandy.requests().is_empty());
}

/// A leg that reads a Sandy call's room gets one attempt: the room sends its
/// taps once.
#[tokio::test]
async fn a_leg_reading_a_sandy_calls_room_is_not_retried() {
    let ricky = MockWorker::start_returning_error(StatusCode::BAD_GATEWAY, json!({})).await;
    let other_ricky = MockWorker::start_returning_error(StatusCode::BAD_GATEWAY, json!({})).await;
    let (sandy, sandy_url, _sandy_stop) = ChainEngine::start(usize::MAX).await;
    let ctx = franky_ctx_with_chain(
        &[
            ("r", &ricky.url, WorkerMode::Plain),
            ("r2", &other_ricky.url, WorkerMode::Plain),
        ],
        &[("s", &sandy_url)],
        2,
        false,
        chain_config(2, true, 4),
    );
    let body = json!({"input_ids": PROMPT, "sampling_params": {"max_new_tokens": 100}});

    let res = build_router(ctx).oneshot(generate(body)).await.unwrap();

    assert_eq!(res.status(), StatusCode::BAD_GATEWAY);
    assert_eq!(sandy.requests().len(), 1);
    let tried = [&ricky, &other_ricky]
        .iter()
        .filter(|worker| worker.captured.lock().unwrap().last_body.is_some())
        .count();
    assert_eq!(tried, 1);
}

/// A Ricky that answers 502 got no taps: its retry goes to another Sandy.
#[tokio::test]
async fn a_ricky_without_its_taps_blames_the_sandy() {
    let ricky = MockWorker::start_returning_error(StatusCode::BAD_GATEWAY, json!({})).await;
    let sandy = MockWorker::start(vec![]).await;
    let other_sandy = MockWorker::start(vec![]).await;
    let ctx = franky_ctx(
        &[("r", &ricky.url, WorkerMode::Plain)],
        &[("s", &sandy.url), ("s2", &other_sandy.url)],
        2,
        true,
    );

    let res = build_router(ctx).oneshot(chat()).await.unwrap();

    assert_eq!(res.status(), StatusCode::BAD_GATEWAY);
    for worker in [&sandy, &other_sandy] {
        worker.captured_json().await;
    }
}
