from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import torch
from safetensors.torch import load_file

from sglang.srt.franky.cache.kv_allocator import FrankyKvAllocator
from sglang.srt.franky.cache.kv_options import FrankyKvOptions
from sglang.srt.franky.cache.tap_cache import TAP_DTYPE, FrankyTapCache
from sglang.srt.franky.config import FrankyConfig
from sglang.srt.franky.parts.franky_mixin import FrankyMixin
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch

if TYPE_CHECKING:
    from torch import nn

    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.model_executor.forward_batch_info import (
        ForwardBatch,
        PPProxyTensors,
    )


class FrankyParts(ABC):
    """This rank's Franky parts inside its model. The model calls them through
    ``FrankyMixin``.
    """

    def __init__(
        self, *, config: FrankyConfig, layer_indices: frozenset[int], device: str
    ) -> None:
        self.config = config
        self.device = device
        self.tap_cache: Optional[FrankyTapCache] = None
        # The layers of this PP stage that have a part, counted from 0.
        self._layer_indices = layer_indices

    @staticmethod
    def attach(
        *,
        model: nn.Module,
        side: str,
        config_path: str,
        weights_path: str,
        dtype: torch.dtype,
        device: str,
    ) -> FrankyParts:
        """Build this side's parts and set them on the one ``FrankyMixin`` module
        inside ``model``.
        """
        # Imported here, not at the top: both subclasses import this module.
        from sglang.srt.franky.parts.ricky_parts import RickyParts
        from sglang.srt.franky.parts.sandy_parts import SandyParts

        language_model = _franky_module_of(model)
        parts_cls = SandyParts if side == "sandy" else RickyParts
        parts = parts_cls.build(
            model=model,
            language_model=language_model,
            config=FrankyConfig.load(config_path),
            weights=load_file(weights_path),
            dtype=dtype,
            device=device,
        )
        language_model.franky_parts = parts
        return parts

    @classmethod
    @abstractmethod
    def build(
        cls,
        *,
        model: nn.Module,
        language_model: FrankyMixin,
        config: FrankyConfig,
        weights: dict[str, torch.Tensor],
        dtype: torch.dtype,
        device: str,
    ) -> FrankyParts:
        """Build this rank's parts from the config and the weights file."""
        raise NotImplementedError

    def kv_options(self) -> FrankyKvOptions:
        return FrankyKvOptions(
            tap_bytes_per_token=(
                len(self._tap_encoder_ids())
                * self.config.encoder.output_size
                * TAP_DTYPE.itemsize
            ),
            paged_allocator_cls=FrankyKvAllocator,
        )

    def init_tap_cache(
        self,
        *,
        num_tokens: int,
        kv_allocator: BaseTokenToKVPoolAllocator,
        req_to_token_pool: ReqToTokenPool,
    ) -> None:
        """Build the tap cache, once the KV budget has set ``num_tokens``."""
        if not isinstance(kv_allocator, FrankyKvAllocator):
            raise ValueError(
                "Franky needs the plain paged KV allocator, got "
                f"{type(kv_allocator).__name__}."
            )
        self.tap_cache = FrankyTapCache(
            encoder_ids=self._tap_encoder_ids(),
            output_size=self.config.encoder.output_size,
            page_size=self.config.tap_page_size,
            num_tokens=num_tokens,
            req_to_token=req_to_token_pool.req_to_token,
            device=self.device,
            attention_workspace=self._attention_workspace(),
        )

    def after_layer(
        self,
        *,
        layer_idx: int,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """For models whose layers return ``hidden_states`` and keep the residual
        in ``forward_batch``: the layer's output waits there to be added.
        """
        if layer_idx not in self._layer_indices:
            return hidden_states
        extra = self._after_layer(
            layer_idx=layer_idx,
            residual=residual_batch.snapshot(hidden_states, forward_batch),
            forward_batch=forward_batch,
        )
        if extra is None:
            return hidden_states
        return residual_batch.add_to_output(hidden_states, forward_batch, extra)

    def after_layer_with_residual(
        self,
        *,
        layer_idx: int,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """For models whose layers return ``(hidden_states, residual)``: the
        layer's output, and the residual before it. The next layer adds them.
        """
        if layer_idx not in self._layer_indices:
            return hidden_states, residual
        extra = self._after_layer(
            layer_idx=layer_idx,
            residual=hidden_states if residual is None else hidden_states + residual,
            forward_batch=forward_batch,
        )
        if extra is None:
            return hidden_states, residual
        return hidden_states + extra, residual

    def load_weights(self, named_tensors: list[tuple[str, torch.Tensor]]) -> None:
        """Copy updated weights, named as in the --franky-weights file, into this
        rank's parts; a part on another PP stage skips its weights.

        Raises:
            ValueError: a name is not a weight of this side's parts.
        """
        prefix, parts, num_parts = self._weight_prefix(), self._parts(), self._num_parts()
        with torch.no_grad():
            for name, tensor in named_tensors:
                side, index, weight = name.split(".", 2)
                if side != prefix or not 0 <= int(index) < num_parts:
                    raise ValueError(f"{name!r} is not a weight of this side's {prefix}")
                part = parts.get(int(index))
                if part is not None:
                    part.get_parameter(weight).copy_(tensor)

    def read_pp(self, *, pp_proxy_tensors: PPProxyTensors) -> None:
        """Take what an earlier PP stage carried."""

    def _attention_workspace(self) -> Optional[torch.Tensor]:
        """The buffer the readers' attention kernel needs, if any."""
        return None

    def write_pp(self, *, pp_proxy_tensors: PPProxyTensors) -> None:
        """Hand what the next PP stage needs."""

    @abstractmethod
    def _weight_prefix(self) -> str:
        """The first part of this side's weight names: encoders or readers."""
        raise NotImplementedError

    @abstractmethod
    def _parts(self) -> dict[int, nn.Module]:
        """This rank's encoders or readers, by index."""
        raise NotImplementedError

    @abstractmethod
    def _num_parts(self) -> int:
        """This side's encoders or readers on every PP stage."""
        raise NotImplementedError

    @abstractmethod
    def _tap_encoder_ids(self) -> tuple[int, ...]:
        """The encoders whose taps this PP stage holds."""
        raise NotImplementedError

    @abstractmethod
    def _after_layer(
        self, *, layer_idx: int, residual: torch.Tensor, forward_batch: ForwardBatch
    ) -> Optional[torch.Tensor]:
        """Run this layer's parts on the whole residual after it, without
        changing it. Return what to add to the layer's output, or None.
        """
        raise NotImplementedError


def _franky_module_of(model: nn.Module) -> FrankyMixin:
    modules = [module for module in model.modules() if isinstance(module, FrankyMixin)]
    if len(modules) != 1:
        raise ValueError(
            f"Franky needs one FrankyMixin module in {type(model).__name__}, found "
            f"{len(modules)}; python/sglang/srt/franky/parts/README.md shows how to "
            "add a model."
        )
    return modules[0]


def load_part(
    *,
    part: nn.Module,
    weights: dict[str, torch.Tensor],
    prefix: str,
    dtype: torch.dtype,
    device: str,
) -> nn.Module:
    """Load one encoder or reader from the weights under ``prefix``, frozen."""
    own = {
        name[len(prefix) :]: tensor
        for name, tensor in weights.items()
        if name.startswith(prefix)
    }
    try:
        part.load_state_dict(own)
    except RuntimeError as error:
        raise ValueError(f"Bad Franky weights under {prefix!r}: {error}") from error
    return part.to(device=device, dtype=dtype).requires_grad_(False)
