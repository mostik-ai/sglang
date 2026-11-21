from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
from torch import nn
from typing_extensions import override

from sglang.srt.franky.cache.sandy_tap_cache import SandyTapCache
from sglang.srt.franky.cache.tap_stripes import TapStripes
from sglang.srt.franky.parts.franky_parts import FrankyParts, load_part

if TYPE_CHECKING:
    from sglang.srt.franky.config import FrankyConfig
    from sglang.srt.franky.parts.encoder.encoder import Encoder
    from sglang.srt.franky.parts.franky_mixin import FrankyMixin
    from sglang.srt.franky.parts.injector.injector import Injector
    from sglang.srt.franky.parts.injector.layer_output import LayerOutput
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.model_executor.forward_batch_info import PPProxyTensors


class SandyParts(FrankyParts):
    """Sandy's side on this rank: runs the encoders of this PP stage on every
    token its forwards run over, the tokens Sandy writes included.

    Layer numbers start at 1 and mean "after this layer". The stage runs layers
    ``start_layer + 1 .. end_layer``.
    """

    def __init__(
        self,
        *,
        config: FrankyConfig,
        injector: Injector,
        encoders: dict[int, Encoder],
        start_layer: int,
        end_layer: int,
        device: str,
        stripes: TapStripes,
    ) -> None:
        self._encoders = encoders
        self._start_layer = start_layer
        self._end_layer = end_layer
        self._encoders_of_layer: dict[int, list[int]] = {}
        for index, encoder in encoders.items():
            for layer in encoder.input_layers:
                if start_layer < layer <= end_layer:
                    self._encoders_of_layer.setdefault(layer, []).append(index)
        # Encoders whose running sum crosses a stage edge, even when this stage
        # has none of their layers.
        all_inputs = [
            config.encoder.input_layers(index=index)
            for index in range(len(config.encoder.layers))
        ]
        self._open_at_start = _open_at(all_inputs, edge=start_layer)
        self._open_at_end = _open_at(all_inputs, edge=end_layer)
        self._states: dict[int, torch.Tensor] = {}
        self.tap_cache: Optional[SandyTapCache] = None
        super().__init__(
            config=config,
            injector=injector,
            layer_indices=frozenset(layer - 1 for layer in self._encoders_of_layer),
            device=device,
            stripes=stripes,
        )

    @classmethod
    @override
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
    ) -> SandyParts:
        start_layer = language_model.start_layer
        end_layer = language_model.end_layer
        encoders = {}
        for index in range(len(config.encoder.layers)):
            inputs = config.encoder.input_layers(index=index)
            if any(start_layer < layer <= end_layer for layer in inputs):
                encoders[index] = load_part(
                    part=config.encoder.build(
                        index=index,
                        sandy_hidden_size=language_model.config.hidden_size,
                    ),
                    weights=weights,
                    prefix=f"encoders.{index}.",
                    dtype=dtype,
                    device=device,
                )
        return cls(
            config=config,
            injector=injector,
            encoders=encoders,
            start_layer=start_layer,
            end_layer=end_layer,
            device=device,
            stripes=TapStripes.of_attention_tp(),
        )

    @override
    def init_taps(
        self,
        *,
        num_kv_tokens: int,
        kv_page_size: int,
        req_to_token_pool: ReqToTokenPool,
    ) -> None:
        self.tap_cache = SandyTapCache.build(
            encoder_ids=self._tap_encoder_ids(),
            output_size=self.config.encoder.output_size,
            page_size=self.config.tap_page_size,
            num_kv_slots=num_kv_tokens + kv_page_size,
            stripes=self.stripes,
            device=self.device,
        )

    @override
    def _tap_encoder_ids(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, encoder in self._encoders.items()
            if self._start_layer < encoder.input_layers[-1] <= self._end_layer
        )

    @override
    def read_pp(self, *, pp_proxy_tensors: PPProxyTensors) -> None:
        for index in self._open_at_start:
            self._states[index] = pp_proxy_tensors[_state_key(index)]

    @override
    def write_pp(self, *, pp_proxy_tensors: PPProxyTensors) -> None:
        for index in self._open_at_end:
            pp_proxy_tensors[_state_key(index)] = self._states.pop(index)

    @override
    def _after_layer(self, *, layer_idx: int, output: LayerOutput) -> LayerOutput:
        residual = self.injector.read(output=output)
        layer = layer_idx + 1
        for index in self._encoders_of_layer[layer]:
            encoder = self._encoders[index]
            state = encoder.accumulate(
                state=None
                if layer == encoder.input_layers[0]
                else self._states.pop(index),
                layer=layer,
                hiddens=residual,
            )
            if layer == encoder.input_layers[-1]:
                self.tap_cache.write(
                    encoder_id=index,
                    taps=encoder.finish(state=state),
                    kv_slots=output.forward_batch.out_cache_loc,
                )
            else:
                self._states[index] = state
        return output


def _open_at(all_inputs: list[tuple[int, ...]], *, edge: int) -> tuple[int, ...]:
    """Encoders with input layers on both sides of the edge after layer ``edge``."""
    return tuple(
        index
        for index, inputs in enumerate(all_inputs)
        if inputs[0] <= edge < inputs[-1]
    )


def _state_key(index: int) -> str:
    return f"franky_encoder_state_{index}"
