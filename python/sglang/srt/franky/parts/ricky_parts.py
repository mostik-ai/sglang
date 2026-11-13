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
    from sglang.srt.franky.parts.injector.layer_output import LayerOutput
    from sglang.srt.franky.parts.reader.reader import Reader


class RickyParts(FrankyParts):
    """Ricky's side on this rank: the readers of this PP stage. Reader ``i``
    reads encoder ``i``'s taps, and the injector adds its result.
    """

    def __init__(
        self,
        *,
        config: FrankyConfig,
        injector: Injector,
        readers: dict[int, Reader],
        device: str,
    ) -> None:
        self._readers = readers
        # Reader layers count from 1, the layer loop from 0.
        self._index_of_layer = {
            config.reader.layers[index] - 1: index for index in readers
        }
        super().__init__(
            config=config,
            injector=injector,
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
        injector: Injector,
        weights: dict[str, torch.Tensor],
        dtype: torch.dtype,
        device: str,
    ) -> RickyParts:
        readers = {}
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
        return cls(config=config, injector=injector, readers=readers, device=device)

    @override
    def _attention_workspace(self) -> Optional[torch.Tensor]:
        return attention_workspace(device=self.device)

    @override
    def _tap_encoder_ids(self) -> tuple[int, ...]:
        # Every stage holds all encoders' taps: PD's transfer pairs entries by id
        # and each Ricky stage reads every Sandy stage.
        return tuple(range(len(self.config.encoder.layers)))

    @override
    def _after_layer(self, *, layer_idx: int, output: LayerOutput) -> LayerOutput:
        index = self._index_of_layer[layer_idx]
        reader_output = self._readers[index](
            tap_cache=self.tap_cache,
            encoder_id=index,
            residual=self.injector.read(output=output),
            forward_batch=output.forward_batch,
            rows=output.rows,
        )
        return self.injector.inject(output=output, addition=reader_output)
