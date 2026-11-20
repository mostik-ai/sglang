from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import torch
from safetensors.torch import load_file

from sglang.srt.franky.cache.tap_buffers import TAP_DTYPE
from sglang.srt.franky.cache.tap_stripes import TapStripes
from sglang.srt.franky.config import FrankyConfig
from sglang.srt.franky.parts.franky_mixin import FrankyMixin

if TYPE_CHECKING:
    from torch import nn

    from sglang.srt.franky.parts.injector.injector import Injector
    from sglang.srt.franky.parts.injector.layer_output import LayerOutput
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
        stripes: TapStripes,
        tap_pool_ratio: float = 1.0,
    ) -> None:
        """
        Args:
            stripes: how the attention-TP ranks split the taps.
        """
        self.config = config
        self.injector = injector
        self.device = device
        self.stripes = stripes
        self._tap_pool_ratio = tap_pool_ratio
        self._layer_indices = layer_indices

    @staticmethod
    def attach(
        *,
        model: nn.Module,
        side: str,
        config_path: str,
        weights_path: Optional[str],
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
            weights=None if weights_path is None else load_file(weights_path),
            dtype=dtype,
            device=device,
        )
        # At load, before the KV pools: a part this stage cannot run fails here,
        # not at the first request.
        injector.check_stage(
            layer_indices=parts._layer_indices,
            end_layer=language_model.end_layer,
            num_layers=language_model.config.num_hidden_layers,
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
        weights: Optional[dict[str, torch.Tensor]],
        dtype: torch.dtype,
        device: str,
    ) -> FrankyParts:
        """Build this rank's parts from the config and the weights file, or at zero without one."""
        raise NotImplementedError

    def tap_bytes_per_kv_token(self) -> int:
        """This rank's tap bytes per KV token, which the KV budget sets aside."""
        tap_bytes = (
            self._tap_pool_ratio
            * len(self._tap_encoder_ids())
            * self.config.encoder.output_size
            * TAP_DTYPE.itemsize
        )
        return math.ceil(tap_bytes / self.stripes.num_ranks)

    @abstractmethod
    def init_taps(
        self,
        *,
        num_kv_tokens: int,
        kv_page_size: int,
        req_to_token_pool: ReqToTokenPool,
    ) -> None:
        """Build this rank's taps once the KV budget set ``num_kv_tokens``: the
        tap pages and, on Ricky, what the readers need to read them.
        """
        raise NotImplementedError

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

    def load_weights(self, named_tensors: list[tuple[str, torch.Tensor]]) -> None:
        """Copy updated weights, named as in the --franky-weights file, into this
        rank's parts; a part on another PP stage skips its weights.

        Raises:
            ValueError: a name is not a weight of this side's parts.
        """
        prefix = self._weight_prefix()
        parts = self._parts_by_index()
        num_parts = self._num_parts_on_all_stages()
        with torch.no_grad():
            for name, tensor in named_tensors:
                side, index, weight = name.split(".", 2)
                if side != prefix or not 0 <= int(index) < num_parts:
                    raise ValueError(
                        f"{name!r} is not a weight of this side's {prefix}"
                    )
                part = parts.get(int(index))
                if part is not None:
                    part.get_parameter(weight).copy_(tensor)

    def read_pp(self, *, pp_proxy_tensors: PPProxyTensors) -> None:
        """Take what an earlier PP stage carried."""

    def write_pp(self, *, pp_proxy_tensors: PPProxyTensors) -> None:
        """Hand what the next PP stage needs."""

    @abstractmethod
    def _weight_prefix(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def _parts_by_index(self) -> dict[int, nn.Module]:
        raise NotImplementedError

    @abstractmethod
    def _num_parts_on_all_stages(self) -> int:
        raise NotImplementedError

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
    weights: Optional[dict[str, torch.Tensor]],
    prefix: str,
    dtype: torch.dtype,
    device: str,
) -> nn.Module:
    """Load one encoder or reader from the weights under ``prefix``, frozen;
    without weights it starts at zero, until a weight update sends them."""
    if weights is None:
        with torch.no_grad():
            for parameter in part.parameters():
                parameter.zero_()
        return part.to(device=device, dtype=dtype).requires_grad_(False)
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
