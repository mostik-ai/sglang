// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::common::cache_aware_fixture::{self, MODEL as DEEPSEEK_V4};
use crate::common::kimi_k3_fixture;
use crate::common::mock_worker::MockWorker;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{json, Value};
use sgl_router::config::{ChatTemplateAdapterKind, Config, FrankySandyConfig, PolicyKind};
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::factory::build_registry_with_defaults;
use sgl_router::policies_reorg::factory::build_sandy_group;
use sgl_router::proxy::Proxy;
use sgl_router::server::app::build_router;
use sgl_router::server::app_context::AppContext;
use sgl_router::state::kv_events::KvEventIndex;
use sgl_router::tokenizer::adapter::encode;
use sgl_router::tokenizer::chat_template_adapter::{self, ChatTemplateAdapter};
use sgl_router::tokenizer::franky_sandy::FrankySandyTokens;
use sgl_router::tokenizer::TokenizerRegistry;
use sgl_router::workers::{EngineProfile, WireProtocol, WorkerRegistry};
use std::num::NonZeroUsize;
use std::ops::Range;
use std::sync::Arc;
use std::time::Duration;
use tower::ServiceExt;

const QWEN: &str = "tiny-qwen";
const TINY_QWEN_TOKENIZER: &str = "tests/fixtures/tiny_qwen/tokenizer.json";
const SANDY_PORT: u16 = 8996;
const CHAT_PATH: &str = "/v1/chat/completions";

#[derive(Clone, Copy)]
enum Engine {
    Ricky,
    Sandy,
    Solo,
}

struct FinalTokenAdapter;

impl ChatTemplateAdapter for FinalTokenAdapter {
    fn generated_spans(&self, input_ids: &[u32]) -> Vec<Range<usize>> {
        vec![Range {
            start: input_ids.len() - 1,
            end: input_ids.len(),
        }]
    }
}

fn config(model: &str, tokenizer_path: &str) -> Config {
    let mut config = cache_aware_fixture::config();
    config.model.id = model.into();
    config.model.tokenizer_path = Some(tokenizer_path.into());
    config.model.policy = PolicyKind::RoundRobin;
    config.model.cache_aware = None;
    config
}

fn router_context(config: Config, engines: &[(&MockWorker, Engine)]) -> AppContext {
    let registry = Arc::new(WorkerRegistry::default());
    for &(worker, engine) in engines {
        let spec = WorkerSpec {
            id: WorkerId(worker.url.clone()),
            url: worker.url.clone(),
            mode: match engine {
                Engine::Sandy => WorkerMode::Sandy,
                Engine::Ricky | Engine::Solo => WorkerMode::Plain,
            },
            model_ids: vec![ModelId(config.model.id.clone())],
            bootstrap_port: matches!(engine, Engine::Sandy).then_some(SANDY_PORT),
            ..Default::default()
        };
        let profile = EngineProfile {
            openai: Some(Arc::default()),
            needs_taps: matches!(engine, Engine::Ricky),
            ..WireProtocol::default().into()
        };
        registry.add_with_cb(spec, None, profile).unwrap();
    }
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&config).unwrap());
    let policies = Arc::new(build_registry_with_defaults(&config).unwrap());
    let proxy = Arc::new(Proxy::new(Duration::from_secs(5)).unwrap());
    let mut ctx = AppContext::new(config, tokenizers, proxy, registry, policies);
    let (sandy, _) = build_sandy_group(&ctx.config.model, &KvEventIndex::new(), None).unwrap();
    ctx.sandy = Some(sandy);
    ctx
}

fn qwen35(ctx: &AppContext) -> Arc<dyn ChatTemplateAdapter> {
    let tokenizer = ctx.tokenizers.get(&ctx.config.model.id).unwrap();
    chat_template_adapter::build(ChatTemplateAdapterKind::Qwen35, &tokenizer).unwrap()
}

fn ids(ctx: &AppContext, text: &str) -> Vec<u32> {
    let tokenizer = ctx.tokenizers.get(&ctx.config.model.id).unwrap();
    encode(&tokenizer, text).unwrap()
}

fn chat_with_an_answer(model: &str) -> Value {
    json!({"model": model, "messages": [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "Bye"},
    ]})
}

struct FrankyPair {
    ricky: MockWorker,
    sandy: MockWorker,
    ctx: AppContext,
}

impl FrankyPair {
    async fn start(config: Config) -> Self {
        let ricky = MockWorker::start(vec![]).await;
        let sandy = MockWorker::start(vec![]).await;
        let ctx = router_context(config, &[(&ricky, Engine::Ricky), (&sandy, Engine::Sandy)]);
        Self { ricky, sandy, ctx }
    }
}

async fn post(ctx: Arc<AppContext>, path: &str, body: &Value) -> (StatusCode, String) {
    let request = Request::post(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = build_router(ctx).oneshot(request).await.unwrap();
    let status = response.status();
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    (status, String::from_utf8(body.to_vec()).unwrap())
}

#[tokio::test]
async fn a_franky_chat_carries_its_rendered_ids_and_the_spans_of_its_answers() {
    let FrankyPair {
        ricky,
        sandy,
        mut ctx,
    } = FrankyPair::start(config(QWEN, TINY_QWEN_TOKENIZER)).await;
    ctx.franky_chat_adapter = Some(qwen35(&ctx));
    let before = ids(
        &ctx,
        "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n",
    );
    let answer = ids(&ctx, "Hello!<|im_end|>");
    let after = ids(
        &ctx,
        "\n<|im_start|>user\nBye<|im_end|>\n<|im_start|>assistant\n<think>\n",
    );

    let (status, _) = post(Arc::new(ctx), CHAT_PATH, &chat_with_an_answer(QWEN)).await;

    assert_eq!(status, StatusCode::OK);
    let ricky_body = ricky.captured_json().await;
    let rendered_ids = [before.clone(), answer.clone(), after].concat();
    assert_eq!(ricky_body["input_ids"], json!(rendered_ids));
    assert_eq!(
        ricky_body["franky_generated_spans"],
        json!([[before.len(), before.len() + answer.len()]])
    );
    assert_eq!(
        ricky_body["messages"],
        chat_with_an_answer(QWEN)["messages"]
    );
    assert_eq!(sandy.captured_json().await, ricky_body);
}

#[tokio::test]
async fn a_lowered_franky_chat_carries_the_spans_in_its_generate_body() {
    let FrankyPair {
        ricky,
        sandy,
        mut ctx,
    } = FrankyPair::start(config(DEEPSEEK_V4, "tests/fixtures/tiny_tokenizer.json")).await;
    ctx.franky_chat_adapter = Some(Arc::new(FinalTokenAdapter));

    let (status, _) = post(Arc::new(ctx), CHAT_PATH, &chat_with_an_answer(DEEPSEEK_V4)).await;

    assert_eq!(status, StatusCode::OK);
    let ricky_body = ricky.captured_json().await;
    let prompt_length = ricky_body["input_ids"].as_array().unwrap().len();
    assert_eq!(
        ricky_body["franky_generated_spans"],
        json!([[prompt_length - 1, prompt_length]])
    );
    assert!(ricky_body.get("messages").is_none(), "{ricky_body}");
    assert_eq!(sandy.captured_json().await, ricky_body);
}

#[tokio::test]
async fn a_franky_chat_without_earlier_answers_carries_no_spans() {
    let FrankyPair {
        ricky,
        sandy: _sandy,
        mut ctx,
    } = FrankyPair::start(config(QWEN, TINY_QWEN_TOKENIZER)).await;
    ctx.franky_chat_adapter = Some(qwen35(&ctx));
    let chat = json!({"model": QWEN, "messages": [{"role": "user", "content": "Hi"}]});

    let (status, _) = post(Arc::new(ctx), CHAT_PATH, &chat).await;

    assert_eq!(status, StatusCode::OK);
    let ricky_body = ricky.captured_json().await;
    assert!(ricky_body["input_ids"].is_array(), "{ricky_body}");
    assert!(
        ricky_body.get("franky_generated_spans").is_none(),
        "{ricky_body}"
    );
}

#[tokio::test]
async fn the_spans_a_chat_sends_with_its_input_ids_pass_through() {
    let FrankyPair {
        ricky,
        sandy,
        mut ctx,
    } = FrankyPair::start(config(QWEN, TINY_QWEN_TOKENIZER)).await;
    ctx.franky_chat_adapter = Some(qwen35(&ctx));
    let mut chat = chat_with_an_answer(QWEN);
    chat["input_ids"] = json!([1, 2, 3, 4, 5, 6]);
    chat["franky_generated_spans"] = json!([[2, 4]]);

    let (status, _) = post(Arc::new(ctx), CHAT_PATH, &chat).await;

    assert_eq!(status, StatusCode::OK);
    let ricky_body = ricky.captured_json().await;
    assert_eq!(ricky_body["input_ids"], chat["input_ids"]);
    assert_eq!(
        ricky_body["franky_generated_spans"],
        chat["franky_generated_spans"]
    );
    assert_eq!(sandy.captured_json().await, ricky_body);
}

#[tokio::test]
async fn the_spans_a_generate_request_sends_pass_through() {
    let FrankyPair { ricky, sandy, ctx } =
        FrankyPair::start(config(QWEN, TINY_QWEN_TOKENIZER)).await;
    let generate = json!({"input_ids": [1, 2, 3, 4, 5, 6], "franky_generated_spans": [[2, 4]]});

    let (status, _) = post(Arc::new(ctx), "/generate", &generate).await;

    assert_eq!(status, StatusCode::OK);
    let ricky_body = ricky.captured_json().await;
    assert_eq!(ricky_body["input_ids"], generate["input_ids"]);
    assert_eq!(
        ricky_body["franky_generated_spans"],
        generate["franky_generated_spans"]
    );
    assert_eq!(sandy.captured_json().await, ricky_body);
}

#[tokio::test]
async fn the_spans_a_completion_sends_reach_its_generate_body() {
    let FrankyPair { ricky, sandy, ctx } =
        FrankyPair::start(config(QWEN, TINY_QWEN_TOKENIZER)).await;
    let completion = json!({
        "model": QWEN,
        "prompt": [1, 2, 3, 4, 5, 6],
        "max_tokens": 8,
        "franky_generated_spans": [[2, 4]],
    });

    let (status, _) = post(Arc::new(ctx), "/v1/completions", &completion).await;

    assert_eq!(status, StatusCode::OK);
    let ricky_body = ricky.captured_json().await;
    assert_eq!(ricky_body["input_ids"], completion["prompt"]);
    assert_eq!(
        ricky_body["franky_generated_spans"],
        completion["franky_generated_spans"]
    );
    assert!(ricky_body["sampling_params"].is_object(), "{ricky_body}");
    assert_eq!(sandy.captured_json().await, ricky_body);
}

#[tokio::test]
async fn franky_chats_whose_answers_the_router_cannot_mark_are_refused() {
    let mut spans_without_input_ids = chat_with_an_answer(QWEN);
    spans_without_input_ids["franky_generated_spans"] = json!([[3, 5]]);
    let mut left_to_the_engine = chat_with_an_answer(QWEN);
    left_to_the_engine["chat_template_kwargs"] = json!({"enable_thinking": false});
    for (chat, with_adapter, names) in [
        (spans_without_input_ids, true, "input_ids"),
        (chat_with_an_answer(QWEN), false, "--franky-chat-adapter"),
        (left_to_the_engine, true, "leaves this one to the engine"),
    ] {
        let FrankyPair {
            ricky,
            sandy,
            mut ctx,
        } = FrankyPair::start(config(QWEN, TINY_QWEN_TOKENIZER)).await;
        ctx.franky_chat_adapter = with_adapter.then(|| qwen35(&ctx));

        let (status, error) = post(Arc::new(ctx), CHAT_PATH, &chat).await;

        assert_eq!(status, StatusCode::BAD_REQUEST, "{chat}");
        assert!(error.contains(names), "{error}");
        assert!(ricky.captured.lock().unwrap().last_body.is_none());
        assert!(sandy.captured.lock().unwrap().last_body.is_none());
    }
}

#[tokio::test]
async fn chats_for_solo_workers_are_forwarded_as_before() {
    let mut spans_without_input_ids = chat_with_an_answer(QWEN);
    spans_without_input_ids["franky_generated_spans"] = json!([[3, 5]]);
    for chat in [chat_with_an_answer(QWEN), spans_without_input_ids] {
        let worker = MockWorker::start(vec![]).await;
        let mut ctx = router_context(
            config(QWEN, TINY_QWEN_TOKENIZER),
            &[(&worker, Engine::Solo)],
        );
        ctx.franky_chat_adapter = Some(qwen35(&ctx));

        let (status, _) = post(Arc::new(ctx), CHAT_PATH, &chat).await;

        assert_eq!(status, StatusCode::OK, "{chat}");
        let body = worker.captured_json().await;
        assert!(body["input_ids"].is_array(), "{body}");
        assert_eq!(
            body.get("franky_generated_spans"),
            chat.get("franky_generated_spans")
        );
    }
}

/// A Franky pair whose Sandy is a Kimi K3 that tokenizes on its own, serving
/// the lowered DeepSeek V4 model; Ricky's adapter marks the prompt's last
/// token as its one earlier answer.
async fn pair_with_kimi_sandy() -> (FrankyPair, tempfile::TempDir) {
    let mut pair =
        FrankyPair::start(config(DEEPSEEK_V4, "tests/fixtures/tiny_tokenizer.json")).await;
    pair.ctx.franky_chat_adapter = Some(Arc::new(FinalTokenAdapter));
    let kimi = kimi_k3_fixture::tokenizer();
    let sandy = FrankySandyConfig {
        tokenizer_path: kimi.path().join("tiktoken.model").display().to_string(),
        chat_adapter: ChatTemplateAdapterKind::KimiK3,
        keep_every: NonZeroUsize::MIN,
    };
    pair.ctx.franky_sandy = Some(Arc::new(
        FrankySandyTokens::load(&sandy, Default::default()).unwrap(),
    ));
    (pair, kimi)
}

#[tokio::test]
async fn a_sandy_with_its_own_tokenizer_reads_its_rendering_and_ricky_its_spans() {
    let (FrankyPair { ricky, sandy, ctx }, _kimi) = pair_with_kimi_sandy().await;

    let (status, _) = post(Arc::new(ctx), CHAT_PATH, &chat_with_an_answer(DEEPSEEK_V4)).await;

    assert_eq!(status, StatusCode::OK);
    let ricky_body = ricky.captured_json().await;
    let sandy_body = sandy.captured_json().await;
    let ricky_ids = ricky_body["input_ids"].as_array().unwrap();
    let sandy_ids = sandy_body["input_ids"].as_array().unwrap();
    assert_ne!(sandy_ids, ricky_ids);
    assert_eq!(ricky_body["franky_sandy_input_ids"], json!(sandy_ids));
    assert_eq!(
        ricky_body["franky_generated_spans"],
        json!([[ricky_ids.len() - 1, ricky_ids.len()]])
    );
    let [sandy_span] = ricky_body["franky_sandy_spans"]
        .as_array()
        .unwrap()
        .as_slice()
    else {
        panic!("{ricky_body}");
    };
    assert!(
        sandy_span[1].as_u64().unwrap() < sandy_ids.len() as u64,
        "{ricky_body}"
    );
    for ricky_field in [
        "franky_generated_spans",
        "franky_sandy_spans",
        "franky_sandy_input_ids",
    ] {
        assert!(sandy_body.get(ricky_field).is_none(), "{sandy_body}");
    }
    assert_eq!(sandy_body["franky_room"], ricky_body["franky_room"]);
}

#[tokio::test]
async fn a_sandy_with_its_own_tokenizer_refuses_what_the_router_cannot_render_for_it() {
    let mut with_input_ids = chat_with_an_answer(DEEPSEEK_V4);
    with_input_ids["input_ids"] = json!([1, 2, 3]);
    // Ricky's adapter finds an answer in every chat, Sandy's only in chats with one.
    let without_answers =
        json!({"model": DEEPSEEK_V4, "messages": [{"role": "user", "content": "Hi"}]});
    let completion = json!({"model": DEEPSEEK_V4, "prompt": "Hi", "max_tokens": 8});
    let generate = json!({"input_ids": [1, 2, 3]});
    let generate_text = json!({"text": "Hi", "franky_sandy_input_ids": [7, 8]});
    for (path, body, names) in [
        (
            CHAT_PATH,
            with_input_ids,
            "sends Sandy's as franky_sandy_input_ids",
        ),
        (CHAT_PATH, without_answers, "earlier answers"),
        ("/v1/completions", completion, "renders only chats"),
        (
            "/generate",
            generate,
            "franky_sandy_input_ids beside input_ids",
        ),
        ("/generate", generate_text, "one prompt's input_ids"),
    ] {
        let (FrankyPair { ricky, sandy, ctx }, _kimi) = pair_with_kimi_sandy().await;

        let (status, error) = post(Arc::new(ctx), path, &body).await;

        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert!(error.contains(names), "{error}");
        assert!(ricky.captured.lock().unwrap().last_body.is_none());
        assert!(sandy.captured.lock().unwrap().last_body.is_none());
    }
}

/// A trainer keeps both sides' ids over a session's turns and sends them: Ricky
/// gets both sides' ids and earlier answers, Sandy her own ids alone.
#[tokio::test]
async fn a_generate_request_with_both_sides_ids_reaches_each_side() {
    let (FrankyPair { ricky, sandy, ctx }, _kimi) = pair_with_kimi_sandy().await;
    let body = json!({
        "input_ids": [5, 6, 7, 8, 9, 10],
        "franky_sandy_input_ids": [40, 41, 42, 43],
        "franky_generated_spans": [[2, 4]],
        "franky_sandy_spans": [[1, 3]],
        "franky_sandy_block_starts": [[1, 2]],
        "sampling_params": {"max_new_tokens": 2},
    });

    let (status, response) = post(Arc::new(ctx), "/generate", &body).await;

    assert_eq!(status, StatusCode::OK, "{response}");
    let ricky_body = ricky.captured_json().await;
    let sandy_body = sandy.captured_json().await;
    assert_eq!(ricky_body["input_ids"], body["input_ids"]);
    for field in [
        "franky_sandy_input_ids",
        "franky_generated_spans",
        "franky_sandy_spans",
        "franky_sandy_block_starts",
    ] {
        assert_eq!(ricky_body[field], body[field], "{ricky_body}");
        assert!(sandy_body.get(field).is_none(), "{sandy_body}");
    }
    assert_eq!(sandy_body["input_ids"], json!([40, 41, 42, 43]));
    assert_eq!(sandy_body["franky_room"], ricky_body["franky_room"]);
}

/// A unary answer comes with Sandy's tokens of the answer, which follow her
/// prompt's ids, and no block starts after its first and only block.
#[tokio::test]
async fn a_unary_answer_carries_sandys_tokens_of_it() {
    let (
        FrankyPair {
            ricky: _ricky,
            sandy: _sandy,
            ctx,
        },
        kimi,
    ) = pair_with_kimi_sandy().await;
    let ricky_tokenizer = ctx.tokenizers.get(DEEPSEEK_V4).unwrap();
    let body = json!({
        "input_ids": [5, 6, 7],
        "franky_sandy_input_ids": [40, 41, 42, 43],
        "sampling_params": {"max_new_tokens": 2},
    });

    let (status, response) = post(Arc::new(ctx), "/generate", &body).await;

    assert_eq!(status, StatusCode::OK, "{response}");
    let response: Value = serde_json::from_str(&response).unwrap();
    // The mock engine answers [1, 2].
    let text =
        sgl_router::tokenizer::adapter::decode_complete(&ricky_tokenizer, &[1, 2], false).unwrap();
    let sandy = FrankySandyTokens::load(
        &FrankySandyConfig {
            tokenizer_path: kimi.path().join("tiktoken.model").display().to_string(),
            chat_adapter: ChatTemplateAdapterKind::KimiK3,
            keep_every: NonZeroUsize::MIN,
        },
        Default::default(),
    )
    .unwrap();
    let answer = sandy.encode_answer_text(&text).unwrap();
    assert_eq!(
        response["meta_info"]["franky_sandy_output_ids"],
        json!(answer)
    );
    assert_eq!(
        response["meta_info"]["franky_sandy_block_starts"],
        json!([])
    );
}

/// A chat with `input_ids` that sends Sandy's too, as Miles' session server
/// does, reaches each side with its own ids; the router renders nothing.
#[tokio::test]
async fn a_chat_with_both_sides_ids_reaches_each_side() {
    let (FrankyPair { ricky, sandy, ctx }, _kimi) = pair_with_kimi_sandy().await;
    let mut chat = chat_with_an_answer(DEEPSEEK_V4);
    chat["input_ids"] = json!([5, 6, 7, 8]);
    chat["franky_sandy_input_ids"] = json!([40, 41, 42]);
    chat["franky_generated_spans"] = json!([[1, 3]]);
    chat["franky_sandy_spans"] = json!([[1, 2]]);

    let (status, response) = post(Arc::new(ctx), CHAT_PATH, &chat).await;

    assert_eq!(status, StatusCode::OK, "{response}");
    let ricky_body = ricky.captured_json().await;
    let sandy_body = sandy.captured_json().await;
    assert_eq!(ricky_body["input_ids"], json!([5, 6, 7, 8]));
    assert_eq!(ricky_body["franky_sandy_input_ids"], json!([40, 41, 42]));
    assert_eq!(ricky_body["franky_generated_spans"], json!([[1, 3]]));
    assert_eq!(ricky_body["franky_sandy_spans"], json!([[1, 2]]));
    assert_eq!(sandy_body["input_ids"], json!([40, 41, 42]));
    assert!(
        sandy_body.get("franky_sandy_input_ids").is_none(),
        "{sandy_body}"
    );
}
