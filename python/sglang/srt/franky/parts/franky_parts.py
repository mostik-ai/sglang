from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import torch
from safetensors.torch import load_file

from sglang.srt.franky.cache.kv_allocator import FrankyKvAllocator
from sglang.srt.franky.cache.kv_options import FrankyKvOptions
from sglang.srt.franky.cache.swa_kv_allocator import FrankySwaKvAllocator
from sglang.srt.franky.cache.tap_cache import TAP_DTYPE, FrankyTapCache
from sglang.srt.franky.config import FrankyConfig
from sglang.srt.franky.parts.franky_mixin import FrankyMixin

if TYPE_CHECKING:
    from torch import nn

    from sglang.srt.franky.parts.injector.injector import Injector
    from sglang.srt.franky.parts.injector.layer_output import LayerOutput
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.model_executor.forward_batch_info import PPProxyTensors


class FrankyParts(ABC):
    """This rank's Franky parts inside its model. The model calls them through
    ``FrankyMixin``.
    """

    def __init__(
        self,
        *,
        config: FrankyConfig,
        injector: Injector,
        layer_indices: frozenset[int],
        device: str,
    ) -> None:
        self.config = config
        self.injector = injector
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
        config = FrankyConfig.load(config_path)
        is_sandy = side == "sandy"
        injector_config = config.injector.sandy if is_sandy else config.injector.ricky
        injector = injector_config.build()
        _check_output_form(language_model=language_model, injector=injector, side=side)
        parts_cls = SandyParts if is_sandy else RickyParts
        parts = parts_cls.build(
            model=model,
            language_model=language_model,
            config=config,
            injector=injector,
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
        injector: Injector,
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
            swa_allocator_cls=FrankySwaKvAllocator,
        )

    def init_tap_cache(
        self,
        *,
        num_tokens: int,
        kv_allocator: BaseTokenToKVPoolAllocator,
        req_to_token_pool: ReqToTokenPool,
    ) -> None:
        """Build the tap cache, once the KV budget has set ``num_tokens``."""
        if not isinstance(kv_allocator, (FrankyKvAllocator, FrankySwaKvAllocator)):
            raise ValueError(
                "Franky needs the paged KV allocator or the sliding-window one with "
                f"a full pool, got {type(kv_allocator).__name__}."
            )
        self.tap_cache = FrankyTapCache(
            encoder_ids=self._tap_encoder_ids(),
            output_size=self.config.encoder.output_size,
            page_size=self.config.tap_page_size,
            kv_page_size=kv_allocator.page_size,
            num_tokens=num_tokens,
            req_to_token=req_to_token_pool.req_to_token,
            device=self.device,
            own_pages=self._owns_tap_pages(),
            attention_workspace=self._attention_workspace(),
        )

    def runs_after(self, *, layer_idx: int) -> bool:
        """Whether a part of this PP stage runs after layer ``layer_idx``,
        counted from 0."""
        return layer_idx in self._layer_indices

    def after_layer(self, *, layer_idx: int, output: LayerOutput) -> LayerOutput:
        """Run the parts of layer ``layer_idx`` on ``output``.

        Returns:
            The residual the next layer reads.
        """
        if not self.runs_after(layer_idx=layer_idx):
            return output
        return self._after_layer(layer_idx=layer_idx, output=output)

    def read_pp(self, *, pp_proxy_tensors: PPProxyTensors) -> None:
        """Take what an earlier PP stage carried."""

    def _attention_workspace(self) -> Optional[torch.Tensor]:
        """The buffer the readers' attention kernel needs, if any."""
        return None

    def _owns_tap_pages(self) -> bool:
        """Whether this side's requests keep their taps in pages of their own."""
        return False

    def write_pp(self, *, pp_proxy_tensors: PPProxyTensors) -> None:
        """Hand what the next PP stage needs."""

    @abstractmethod
    def _tap_encoder_ids(self) -> tuple[int, ...]:
        """The encoders whose taps this PP stage holds."""
        raise NotImplementedError

    @abstractmethod
    def _after_layer(self, *, layer_idx: int, output: LayerOutput) -> LayerOutput:
        """Run this layer's parts, reading and changing the residual only through
        ``self.injector``.
        """
        raise NotImplementedError


def _check_output_form(
    *, language_model: FrankyMixin, injector: Injector, side: str
) -> None:
    output_cls = language_model.franky_output_cls
    if not issubclass(output_cls, injector.output_cls):
        raise ValueError(
            f"{type(language_model).__name__} hands Franky {output_cls.__name__}, "
            f"but the {side} injector {type(injector).__name__} takes "
            f"{injector.output_cls.__name__}; set injector.{side} in the Franky "
            "config to the injector of this model's residual form."
        )


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
