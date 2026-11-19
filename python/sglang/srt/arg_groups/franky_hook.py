from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from sglang.srt.arg_groups.overrides import (
    declare_resolution,
    model_config_of,
    resolving_view,
)
from sglang.srt.franky.config import FrankyConfig
from sglang.srt.franky.parts.reader.aligned.config import AlignedReaderConfig
from sglang.srt.franky.parts.reader.query.config import QueryReaderConfig
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    default_cuda_graph_config,
)

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.server_args import ServerArgs


_READER_PAGE_SIZES = (16, 32, 64, 128)


def handle_franky(server_args: ServerArgs) -> None:
    """Refuse what Franky does not support."""
    cfg = resolving_view(server_args)
    if cfg.franky_side == "null":
        return
    if cfg.franky_config is None:
        raise ValueError(f"--franky-side {cfg.franky_side} needs --franky-config.")
    config = FrankyConfig.load(cfg.franky_config)
    model_config = model_config_of(cfg)
    side_checks = _sandy_checks if cfg.franky_side == "sandy" else _ricky_checks
    checks = {
        **_shared_checks(cfg=cfg, model_config=model_config),
        **_kv_pool_checks(cfg=cfg),
        **side_checks(cfg=cfg, config=config, model_config=model_config),
    }
    refusals = [reason for reason, is_refused in checks.items() if is_refused]
    if refusals:
        raise ValueError(
            f"--franky-side {cfg.franky_side} does not support: {'; '.join(refusals)}."
        )
    if cfg.franky_side == "ricky" and cfg.disaggregation_mode == "decode":
        # A retracted request's KV backup holds no taps, so PD aborts it instead.
        declare_resolution(
            server_args,
            "handle_franky",
            disaggregation_decode_retraction_backup="none",
        )
    if cfg.franky_side == "sandy":
        # Sandy never decodes: a decode graph would only cost startup time and memory.
        graphs = cfg.cuda_graph_config or default_cuda_graph_config()
        declare_resolution(
            server_args,
            "handle_franky",
            disable_overlap_schedule=True,
            cuda_graph_config=dataclasses.replace(
                graphs,
                decode=dataclasses.replace(graphs.decode, backend=Backend.DISABLED),
            ),
        )


def _shared_checks(*, cfg: Any, model_config: ModelConfig) -> dict[str, bool]:
    return {
        "speculative decoding": cfg.speculative_algorithm is not None,
        "LoRA": cfg.enable_lora,
        "--dp-size above 1, run one more worker instead": cfg.dp_size > 1,
        # Parts must see each layer's whole residual, for all of its tokens, once.
        "--enable-attn-tp-input-scattered": cfg.enable_attn_tp_input_scattered,
        "--enable-layernorm-sp": cfg.enable_layernorm_sp,
        "--enable-two-batch-overlap": cfg.enable_two_batch_overlap,
        # Split prefill runs the layers outside the model's layer loop.
        "--enable-pdmux": cfg.enable_pdmux,
        # Taps are striped over attention TP ranks and move between them.
        f"--attn-cp-size {cfg.attn_cp_size}": cfg.attn_cp_size > 1,
        # KV loaded from outside the forward would have no taps.
        "--enable-hierarchical-cache": cfg.enable_hierarchical_cache,
        "--enable-lmcache": cfg.enable_lmcache,
        f"--franky-chain-hold-seconds {cfg.franky_chain_hold_seconds}": (
            cfg.franky_chain_hold_seconds <= 0
        ),
        f"--franky-tap-pool-ratio {cfg.franky_tap_pool_ratio}, it must be above 0": (
            cfg.franky_tap_pool_ratio <= 0
        ),
    }


def _kv_pool_checks(*, cfg: Any) -> dict[str, bool]:
    """KV pools that Franky has not run with."""
    return {
        "--enable-unified-memory": cfg.enable_unified_memory,
        "--enable-hisparse": cfg.enable_hisparse,
        "--enable-kv-cache-sharding": cfg.enable_kv_cache_sharding,
        "--dcp-size above 1": cfg.dcp_size > 1,
    }


def _sandy_checks(
    *, cfg: Any, config: FrankyConfig, model_config: ModelConfig
) -> dict[str, bool]:
    last_layer = config.encoder.layers[-1]
    num_layers = model_config.num_hidden_layers
    return {
        f"--disaggregation-mode {cfg.disaggregation_mode}": (
            cfg.disaggregation_mode != "null"
        ),
        f"encoder layer {last_layer}, the model has {num_layers} layers": (
            last_layer > num_layers
        ),
        # Sandy's taps live at its KV slots.
        f"--franky-tap-pool-ratio {cfg.franky_tap_pool_ratio}": (
            cfg.franky_tap_pool_ratio != 1
        ),
        f"page size {cfg.page_size} after model and backend rules, it must be a "
        f"multiple of tap_page_size {config.tap_page_size}": (
            cfg.page_size % config.tap_page_size != 0
        ),
    }


def _ricky_checks(
    *, cfg: Any, config: FrankyConfig, model_config: ModelConfig
) -> dict[str, bool]:
    last_layer = config.reader.layers[-1]
    num_layers = model_config.num_hidden_layers
    tap_size = config.encoder.output_size
    pd = cfg.disaggregation_mode != "null"
    decode = cfg.disaggregation_mode == "decode"
    retraction_backup = cfg.disaggregation_decode_retraction_backup
    return {
        # Taps go with the KV over PD's NIXL classes only.
        f"--disaggregation-transfer-backend {cfg.disaggregation_transfer_backend}": (
            pd and cfg.disaggregation_transfer_backend != "nixl"
        ),
        # Every Ricky stage holds all taps, so PD stages would relay them twice.
        "--pp-size above 1 with --disaggregation-mode": pd and cfg.pp_size > 1,
        # Decode KV that one request computed saw only that request's taps.
        "--disaggregation-decode-enable-radix-cache": (
            decode and cfg.disaggregation_decode_enable_radix_cache
        ),
        f"--disaggregation-decode-retraction-backup {retraction_backup}": (
            decode and retraction_backup not in (None, "none")
        ),
        # A chunk ends at the taps that have landed.
        "--chunked-prefill-size -1": not decode and cfg.chunked_prefill_size <= 0,
        # A disabled cache admits ignore-EOS requests past the landed taps.
        "--disable-radix-cache": not decode and cfg.disable_radix_cache,
        f"reader layer {last_layer}, the model has {num_layers} layers": (
            last_layer > num_layers
        ),
        # Query readers attend over taps with FlashAttention, whose heads are this
        # size; an aligned reader maps one tap and attends over none.
        f"encoder output_size {tap_size}, readers need a multiple of 8 up to 256": (
            isinstance(config.reader, QueryReaderConfig)
            and (tap_size % 8 != 0 or tap_size > 256)
        ),
        # An aligned reader reads the tap at a row's own token, which only a Sandy
        # that reads Ricky's tokens wrote.
        "an aligned reader with shared_tokenizer false": (
            isinstance(config.reader, AlignedReaderConfig)
            and not config.shared_tokenizer
        ),
        # Readers attend over tap pages with TRT-LLM's decode kernels on Blackwell.
        f"tap_page_size {config.tap_page_size}, readers need 16, 32, 64 or 128": (
            config.tap_page_size not in _READER_PAGE_SIZES
        ),
        # Readers take each prefill's lengths from the host, and a full graph
        # replays the lengths it was captured with.
        "--cuda-graph-backend-prefill full": (
            cfg.cuda_graph_config.prefill.backend == Backend.FULL
        ),
    }
