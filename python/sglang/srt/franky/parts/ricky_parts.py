from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
from torch import nn
from typing_extensions import override

from sglang.srt.franky.parts.franky_parts import FrankyParts, load_part
from sglang.srt.franky.parts.reader.query.reader import attention_workspace

if TYPE_CHECKING:
    from sglang.srt.franky.config import FrankyConfig
    from sglang.srt.franky.parts.franky_mixin import FrankyMixin
    from sglang.srt.franky.parts.injector.injector import Injector
    from sglang.srt.franky.parts.reader.reader import Reader
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class RickyParts(FrankyParts):
    """Ricky's side on this rank: the readers of this PP stage, each with its
    injector. Reader ``i`` reads encoder ``i``'s taps.
    """

    def __init__(
        self,
        *,
        config: FrankyConfig,
        readers: dict[int, Reader],
        injectors: dict[int, Injector],
        device: str,
    ) -> None:
        self._readers = readers
        self._injectors = injectors
        # Reader layers count from 1, the layer loop from 0.
        self._index_of_layer = {
            config.reader.layers[index] - 1: index for index in readers
        }
        super().__init__(
            config=config,
            layer_indices=frozenset(self._index_of_layer),
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
    ) -> RickyParts:
        readers, injectors = {}, {}
        for index, layer in enumerate(config.reader.layers):
            if language_model.start_layer < layer <= language_model.end_layer:
                readers[index] = load_part(
                    part=config.reader.build(
                        input_size=config.encoder.output_size,
                        ricky_hidden_size=language_model.config.hidden_size,
                    ),
                    weights=weights,
                    prefix=f"readers.{index}.",
                    dtype=dtype,
                    device=device,
                )
                injectors[index] = config.injector.build()
        return cls(config=config, readers=readers, injectors=injectors, device=device)

    @override
    def _weight_prefix(self) -> str:
        return "readers"

    @override
    def _parts(self) -> dict[int, nn.Module]:
        return dict(self._readers)

    @override
    def _num_parts(self) -> int:
        return len(self.config.reader.layers)

    @override
    def _attention_workspace(self) -> Optional[torch.Tensor]:
        return attention_workspace(device=self.device)

    @override
    def _tap_encoder_ids(self) -> tuple[int, ...]:
        # Every stage holds all encoders' taps: PD's transfer pairs entries by id
        # and each Ricky stage reads every Sandy stage.
        return tuple(range(len(self.config.encoder.layers)))

    @override
    def _after_layer(
        self, *, layer_idx: int, residual: torch.Tensor, forward_batch: ForwardBatch
    ) -> Optional[torch.Tensor]:
        index = self._index_of_layer[layer_idx]
        reader_output = self._readers[index](
            tap_cache=self.tap_cache,
            encoder_id=index,
            residual=residual,
            forward_batch=forward_batch,
        )
        return self._injectors[index].inject(
            reader_output=reader_output, residual=residual
        )
