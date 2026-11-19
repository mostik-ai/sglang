from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
from torch import nn
from typing_extensions import override

from sglang.srt.distributed.parallel_state import get_attn_tp_group
from sglang.srt.franky.cache.ricky_tap_cache import RickyTapCache
from sglang.srt.franky.cache.tap_read_rows import TapReadRows
from sglang.srt.franky.parts.franky_parts import FrankyParts, load_part
from sglang.srt.franky.parts.reader.tap_attention import (
    TapAttention,
    TapReadings,
    TrtllmWorkspace,
)
from sglang.srt.runtime_context import get_franky
from sglang.srt.utils.common import ceil_div

if TYPE_CHECKING:
    from sglang.srt.franky.cache.tap_stripes import TapStripes
    from sglang.srt.franky.config import FrankyConfig
    from sglang.srt.franky.parts.franky_mixin import FrankyMixin
    from sglang.srt.franky.parts.injector.injector import Injector
    from sglang.srt.franky.parts.injector.layer_output import LayerOutput
    from sglang.srt.franky.parts.reader.reader import Reader
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool


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
        stripes: TapStripes,
        tap_pool_ratio: float = 1.0,
    ) -> None:
        self._readers = readers
        # Reader layers count from 1, the layer loop from 0.
        self._index_of_layer = {
            config.reader.layers[index] - 1: index for index in readers
        }
        self.tap_cache: Optional[RickyTapCache] = None
        self.read_rows: Optional[TapReadRows] = None
        self._attentions: dict[int, TapAttention] = {}
        super().__init__(
            config=config,
            injector=injector,
            layer_indices=frozenset(self._index_of_layer),
            device=device,
            stripes=stripes,
            tap_pool_ratio=tap_pool_ratio,
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
        return cls(
            config=config,
            injector=injector,
            readers=readers,
            device=device,
            stripes=cls.attention_tp_stripes(),
            tap_pool_ratio=get_franky().franky_tap_pool_ratio,
        )

    @override
    def init_tap_cache(
        self,
        *,
        num_kv_tokens: int,
        kv_page_size: int,
        req_to_token_pool: ReqToTokenPool,
    ) -> None:
        """Ricky's taps get a pool of their own: ``tap_pool_ratio`` taps per KV
        token, at least one request of the context length.
        """
        page_size = self.config.tap_page_size
        stripe_size = self.stripes.num_ranks * page_size
        context_len = req_to_token_pool.req_to_token.shape[1]
        num_taps = (
            int(self._tap_pool_ratio * num_kv_tokens) // stripe_size * stripe_size
        )
        if num_taps < ceil_div(context_len, stripe_size) * stripe_size:
            raise ValueError(
                f"Tap pool ratio {self._tap_pool_ratio} holds the taps of {num_taps} "
                f"tokens, fewer than one request at the context length {context_len}; "
                "raise --franky-tap-pool-ratio."
            )
        self.tap_cache = RickyTapCache.build(
            encoder_ids=self._tap_encoder_ids(),
            output_size=self.config.encoder.output_size,
            page_size=page_size,
            num_taps=num_taps,
            stripes=self.stripes,
            device=self.device,
        )
        self.read_rows = TapReadRows(
            num_rows=req_to_token_pool.req_to_token.shape[0],
            max_stripes=ceil_div(context_len, stripe_size),
            device=self.device,
        )
        self._attentions = self._build_attentions()

    @property
    def max_taps(self) -> int:
        """The most taps one request may read: Ricky's context length."""
        return self.read_rows.stripe_slots.shape[1] * self.tap_cache.stripe_size

    @override
    def _tap_encoder_ids(self) -> tuple[int, ...]:
        # Every stage holds all encoders' taps: PD's transfer pairs entries by id
        # and each Ricky stage reads every Sandy stage.
        return tuple(range(len(self.config.encoder.layers)))

    @override
    def _after_layer(self, *, layer_idx: int, output: LayerOutput) -> LayerOutput:
        index = self._index_of_layer[layer_idx]
        reader_output = self._readers[index](
            taps=self._attentions[index],
            residual=self.injector.read(output=output),
            forward_batch=output.forward_batch,
            rows=output.rows,
        )
        return self.injector.inject(output=output, addition=reader_output)

    def _build_attentions(self) -> dict[int, TapAttention]:
        readings = TapReadings(
            read_rows=self.read_rows,
            stripes=self.stripes,
            page_size=self.config.tap_page_size,
        )
        workspace = TrtllmWorkspace.for_device(self.device)
        group = get_attn_tp_group() if self.stripes.num_ranks > 1 else None
        first_index = min(self._readers, default=None)
        return {
            index: TapAttention(
                pages=self.tap_cache.buffers.pages(index),
                readings=readings,
                starts_forward=index == first_index,
                group=group,
                workspace=workspace,
            )
            for index in self._readers
        }
