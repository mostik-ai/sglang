from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
from torch import nn
from typing_extensions import override

from sglang.srt.franky.parts.franky_parts import FrankyParts, load_part
from sglang.srt.layers.logits_processor import LogitsProcessorOutput

if TYPE_CHECKING:
    from sglang.srt.franky.config import FrankyConfig
    from sglang.srt.franky.parts.encoder.encoder import Encoder
    from sglang.srt.franky.parts.franky_mixin import FrankyMixin
    from sglang.srt.model_executor.forward_batch_info import (
        ForwardBatch,
        PPProxyTensors,
    )


class SandyParts(FrankyParts):
    """Sandy's side on this rank: runs the encoders of this PP stage. It also drops
    the model's head, because Sandy never samples.

    Layer numbers start at 1 and mean "after this layer". The stage runs layers
    ``start_layer + 1 .. end_layer``.
    """

    def __init__(
        self,
        *,
        config: FrankyConfig,
        encoders: dict[int, Encoder],
        start_layer: int,
        end_layer: int,
        device: str,
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
        super().__init__(
            config=config,
            layer_indices=frozenset(layer - 1 for layer in self._encoders_of_layer),
            device=device,
        )

    @classmethod
    @override
    def build(
        cls,
        *,
        model: nn.Module,
        language_model: FrankyMixin,
        config: FrankyConfig,
        weights: Optional[dict[str, torch.Tensor]],
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
        _drop_head(model=model, language_model=language_model)
        return cls(
            config=config,
            encoders=encoders,
            start_layer=start_layer,
            end_layer=end_layer,
            device=device,
        )

    @override
    def _weight_prefix(self) -> str:
        return "encoders"

    @override
    def _parts(self) -> dict[int, nn.Module]:
        return dict(self._encoders)

    @override
    def _num_parts(self) -> int:
        return len(self.config.encoder.layers)

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
    def _after_layer(
        self, *, layer_idx: int, residual: torch.Tensor, forward_batch: ForwardBatch
    ) -> Optional[torch.Tensor]:
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
                    kv_slots=forward_batch.out_cache_loc,
                )
            else:
                self._states[index] = state
        return None


class _NoLogits(nn.Module):
    """Sandy's requests have ``max_new_tokens=0``, so the head never runs. The
    logits have rows and no columns: attention DP slices them after the forward,
    and a prefill-only batch samples nothing.
    """

    def forward(
        self, input_ids: torch.Tensor, hidden_states: torch.Tensor, *args, **kwargs
    ) -> LogitsProcessorOutput:
        return LogitsProcessorOutput(
            next_token_logits=hidden_states.new_empty((hidden_states.shape[0], 0))
        )


def _drop_head(*, model: nn.Module, language_model: FrankyMixin) -> None:
    model.logits_processor = _NoLogits()
    # A tied head is the language model's own embedding, so only a separate one goes.
    if all(module is not model.lm_head for module in language_model.modules()):
        model.lm_head = None


def _open_at(all_inputs: list[tuple[int, ...]], *, edge: int) -> tuple[int, ...]:
    """Encoders with input layers on both sides of the edge after layer ``edge``."""
    return tuple(
        index
        for index, inputs in enumerate(all_inputs)
        if inputs[0] <= edge < inputs[-1]
    )


def _state_key(index: int) -> str:
    return f"franky_encoder_state_{index}"
