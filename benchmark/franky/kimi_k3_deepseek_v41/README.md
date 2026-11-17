# Franky: Kimi K3 + DeepSeek V4.1 Flash

Kimi K3 is Sandy (prefill only, its taps feed the readers) and DeepSeek V4.1
Flash is Ricky (answers). The two models tokenize differently, so the Franky
config sets `"shared_tokenizer": false` and sgl-router renders every chat for
both models.

| File | What |
| --- | --- |
| `franky-config.json` | 20 encoder/reader pairs: K3 layers 5, 9, 14, 19, ..., 88, 93 to DeepSeek layers 2, 4, ..., 40; tap width 256; 64-token tap pages; K3 read through its attention-residual bank, DeepSeek's added to all four hyper-connection streams |
| `seed_weights.py` | Seeded encoder and reader weights for a pair without trained ones |

## Layout

Two B300 nodes. Node A runs Sandy; node B runs Ricky as PD (prefill and decode,
TP4 + EP4 each) and the router. Every NIXL agent keeps a progress thread
polling, and a Ricky prefill rank runs three (PD's KV, taps from Sandy, the relay
to decode), so node B's eight schedulers keep about 27 cores busy even when idle.
Run the pair on bare metal or plain containers: under a sandboxed runtime (gVisor,
as on Modal) syscalls beside those pollers are slow (`getpid` took ~50 µs under
load), and decode's HTTP process, which spends its time on per-token metrics and
stream writes, then caps Franky near 3 requests/s whatever the prompt length.

```bash
python seed_weights.py --config franky-config.json \
  --sandy-model /models/Kimi-K3 --ricky-model /models/DeepSeek-V4.1-Flash \
  --output-dir /models/franky-k3-dsv41
```

Sandy, node A (the K3 cookbook's B300 prefill cell, with 64-token pages and FP8
KV; on network storage each rank prefetches its share of the 1.6 TB checkpoint
once, instead of every rank reading all of it). A Sandy token costs 13.8 KB of
K3 KV and 10.2 KB of taps per GPU, so at the cell's `--mem-fraction-static 0.85`
the KV pool holds 720k tokens and refuses a prompt at DeepSeek V4.1's 1M
context; 0.90 with 288 Mamba state slots (72 running requests, the cell has 69)
holds about 1.3M:

```bash
python -m sglang.launch_server --model-path /models/Kimi-K3 --trust-remote-code \
  --tp-size 8 --mem-fraction-static 0.90 --max-mamba-cache-size 288 \
  --page-size 64 --kv-cache-dtype fp8_e4m3 \
  --chunked-prefill-size 16384 --max-prefill-tokens 16384 \
  --disable-custom-all-reduce --enable-symm-mem --weight-loader-prefetch-checkpoints \
  --franky-side sandy --franky-config franky-config.json \
  --franky-weights /models/franky-k3-dsv41/sandy.safetensors \
  --franky-bootstrap-port 8996 --host 0.0.0.0 --port 30000
```

One TP8 node prefills about 22k tokens/s, half of what one DeepSeek PD node takes at
8k prompts, so prefill-heavy traffic backs up behind Sandy. Pipeline stages split
K3's MLA KV and the taps by layer, where TP8 keeps both on every GPU. Sandy alone on
one B300 node at `--mem-fraction-static 0.90`, prompts without a room:

| Layout | KV tokens | 1k, c1 | 8k, c1 | 1k, c64 | 8k, c64 | 64k, c4 |
| --- | --- | --- | --- | --- | --- | --- |
| TP8, `--max-mamba-cache-size 288` | 1.28M | 264 ms | 410 ms | 19.5k tok/s | 22.2k tok/s | 18.9k tok/s |
| PP2 x TP4 | 2.20M | 286 ms | 533 ms | 26.2k tok/s | 32.4k tok/s | 25.1k tok/s |
| PP4 x TP2 | 3.58M | 314 ms | 817 ms | 21.7k tok/s | 39.1k tok/s | 29.6k tok/s |
| PP8 x TP1 | 6.68M | 482 ms | 1353 ms | 18.2k tok/s | 41.6k tok/s | 26.0k tok/s |

PP2 x TP4 keeps a lone request's latency near TP8's with 1.5x the prefill throughput
and 1.7x the prefix cache; deeper pipelines trade latency for more of both. Replace
`--tp-size 8` with `--tp-size 4 --pp-size 2 --disable-flashinfer-autotune` and drop
`--enable-symm-mem`. A stage must not end at a layer Franky reads after, since the
next stage holds that layer's attention-residual mix: set `SGLANG_PP_LAYER_PARTITION`
to `47,46` (PP2), `24,23,22,24` (PP4) or `12,12,11,12,11,11,12,12` (PP8). Sandy
refuses a split that ends a stage at a part's layer when it loads. For prefill-heavy
traffic, use PP8 x TP1, the K3 cookbook's Long-Context prefill strategy: it nearly
doubles TP8's prefill throughput, close to the ~45k tokens/s one DeepSeek V4.1 PD
node takes at 8k/1k c64, at ~220 ms more for a lone 1k request. Beyond that, run
more Sandy engines per Ricky; the router spreads chats over them by cache.

Ricky, node B (the cookbook's B300 High-Throughput flags per worker, with
`--mem-fraction-static 0.85`: at SGLang's default a B300 decode worker runs out of
memory while it starts). Taps cost 10 KB per token against DeepSeek's 0.96 KB of
KV, and prefill holds them only for the requests it is prefilling while its
prefix cache keeps KV, so `--franky-tap-pool-ratio 0.1` gives prefill 51M tokens
of KV instead of 9.6M (DeepSeek alone: 111M). Decode keeps no prefix cache, so it
holds taps for every slot:

```bash
python -m sglang.launch_server --model-path /models/DeepSeek-V4.1-Flash --trust-remote-code \
  --served-model-name deepseek-ai/DeepSeek-V4.1-Flash \
  --tp 4 --ep-size 4 --max-running-requests 256 --mem-fraction-static 0.85 \
  --disaggregation-mode prefill --disaggregation-transfer-backend nixl \
  --disaggregation-bootstrap-port 8998 \
  --franky-side ricky --franky-config franky-config.json --franky-tap-pool-ratio 0.1 \
  --franky-weights /models/franky-k3-dsv41/ricky.safetensors \
  --franky-bootstrap-port 8997 --host 0.0.0.0 --port 30100 --base-gpu-id 0
```

```bash
python -m sglang.launch_server --model-path /models/DeepSeek-V4.1-Flash --trust-remote-code \
  --served-model-name deepseek-ai/DeepSeek-V4.1-Flash \
  --tp 4 --ep-size 4 --max-running-requests 256 --mem-fraction-static 0.85 \
  --disaggregation-mode decode --disaggregation-transfer-backend nixl \
  --franky-side ricky --franky-config franky-config.json \
  --franky-weights /models/franky-k3-dsv41/ricky.safetensors \
  --franky-bootstrap-port 8997 --host 0.0.0.0 --port 30200 --base-gpu-id 4
```

Router, node B:

```bash
sgl-router --host 0.0.0.0 --port 31000 \
  --model-id deepseek-ai/DeepSeek-V4.1-Flash \
  --tokenizer-path /models/DeepSeek-V4.1-Flash/tokenizer.json \
  --worker-urls http://<node A>:30000 http://127.0.0.1:30100 http://127.0.0.1:30200 \
  --chat-routing reorg --policy cache_aware \
  --request-timeout-secs 3600 --stream-idle-timeout-secs 3600 \
  --franky-chat-adapter deepseek-v4 \
  --franky-sandy-tokenizer-path /models/Kimi-K3/tiktoken.model \
  --franky-sandy-chat-adapter kimi-k3
```

The router serves the model it is given with `--model-id`, so both Ricky workers
serve under that name (`--served-model-name`); Sandy joins it under any name. It
also hands each worker URL's host to the other engines as the bootstrap host, so
a worker on another node than its peers needs an address they can reach, not
`127.0.0.1`.

A Franky stream sends nothing until Sandy's taps land, so behind a busy Sandy the
router's default 180 s stream idle timeout cuts requests that are only slow; the
aborts then count against the decode worker and open its breaker for every request.
Keep `--stream-idle-timeout-secs` at the request timeout.

Send a few chats through the router before taking traffic: the first Franky
requests open the NIXL connections and compile kernels (about 10 s each).

NIXL needs `UCX_NET_DEVICES` set to the devices of `NCCL_IB_HCA`, each with
`:1`, on every engine. RoCE rails (Emma) also need `UCX_IB_GID_INDEX=3`;
InfiniBand rails have no GID 3, and with it set UCX finds no usable device.

When Sandy falls behind, Ricky's requests wait minutes between PD's bootstrap and
the KV transfer, and past PD's 300 s defaults Ricky fails them. Set
`SGLANG_DISAGGREGATION_WAITING_TIMEOUT=3600` and
`SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=3600` on both Ricky workers, as long as
the router's request timeout. On the Mooncake trace with one PP2 x TP4 Sandy,
a run whose TTFT p90 stayed at 270 s served all 2,728 requests, and one at 443 s
lost 1,161 of them this way.

## Battery

AIPerf 0.12.0 against the router's chat route, as in the GLM Kolupalka runs,
without the user-centric points (Franky keeps no host or storage KV tier):

| Point | AIPerf options |
| --- | --- |
| Synth 1k/1k, 8k/1k x c1, 4, 16, 64, 256 | `--isl {1024,8192} --osl 1024 --isl-stddev 0 --osl-stddev 0 --concurrency C --request-count 10C --warmup-request-count 2C` |
| AgentX x c1, 8, 32 | `--scenario inferencex-agentx-mvp --public-dataset semianalysis-cc-traces-weka-062126 --concurrency C --benchmark-duration 900` |
| Mooncake trace | `--custom-dataset-type mooncake_trace --input-file conversation_trace.jsonl --fixed-schedule --fixed-schedule-auto-offset --fixed-schedule-end-offset 900000` |

Every point also passes `--endpoint-type chat --streaming --use-server-token-count
--extra-inputs '{"ignore_eos": true}'`. The baseline is the same Ricky PD
without `--franky-*` flags and without Sandy.
