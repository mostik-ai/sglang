from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
from torch import nn
from typing_extensions import override

from sglang.srt.franky.parts.franky_parts import FrankyParts, load_part
from sglang.srt.franky.parts.reader.query.reader import attention_workspace
from sglang.srt.utils.common import async_h2d

if TYPE_CHECKING:
    from sglang.srt.franky.config import FrankyConfig
    from sglang.srt.franky.parts.franky_mixin import FrankyMixin
    from sglang.srt.franky.parts.injector.injector import Injector
    from sglang.srt.franky.parts.reader.reader import Reader
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
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
        # Row 0 serves padded requests and stays zero.
        self._outputs_past_the_taps: dict[int, torch.Tensor] = {}
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
    def init_tap_cache(
        self,
        *,
        num_tokens: int,
        kv_allocator: BaseTokenToKVPoolAllocator,
        req_to_token_pool: ReqToTokenPool,
    ) -> None:
        super().init_tap_cache(
            num_tokens=num_tokens,
            kv_allocator=kv_allocator,
            req_to_token_pool=req_to_token_pool,
        )
        num_rows = req_to_token_pool.req_to_token.shape[0]
        self._outputs_past_the_taps = {
            index: reader.mlp[-1].weight.new_zeros(
                (num_rows, reader.mlp[-1].out_features)
            )
            for index, reader in self._readers.items()
            if reader.same_past_the_taps
        }

    def keep_past_the_taps(self, *, req_pool_indices: list[int]) -> None:
        """Read once what the decoded tokens of these requests get, now that all
        their taps are in; a request without taps gets zero.
        """
        if not self._outputs_past_the_taps or not req_pool_indices:
            return
        rows = async_h2d(req_pool_indices, dtype=torch.int64, device=self.device)
        for index, outputs in self._outputs_past_the_taps.items():
            outputs[rows] = self._readers[index].read_past_the_taps(
                tap_cache=self.tap_cache, encoder_id=index, req_pool_indices=rows
            )

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
        if (
            index in self._outputs_past_the_taps
            and forward_batch.forward_mode.is_decode_or_idle()
        ):
            reader_output = self._outputs_past_the_taps[index][
                forward_batch.req_pool_indices
            ]
        else:
            reader_output = self._readers[index](
                tap_cache=self.tap_cache,
                encoder_id=index,
                residual=residual,
                forward_batch=forward_batch,
            )
        return self._injectors[index].inject(
            reader_output=reader_output, residual=residual
        )
