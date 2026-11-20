from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Optional

import msgspec

from sglang.srt.franky.cache.generated_spans import (
    FrankyGeneratedSpans,
    GeneratedSpan,
    checked_earlier_answers,
)
from sglang.srt.franky.chains.chain import FrankyChain
from sglang.srt.franky.transfer.messages import FrankyRoom

if TYPE_CHECKING:
    from sglang.srt.managers.io_struct import GenerateReqInput
    from sglang.srt.managers.schedule_batch import Req


class FrankyRequest(msgspec.Struct, frozen=True, kw_only=True):
    """What a request asks of Franky. The tokenizer manager builds and checks it
    once from the request's API fields; SGLang's core carries it unopened.

    Attributes:
        room: where the request's taps meet; None for a request that reads none.
        chain: the chain the request is a block of, if any.
        earlier_answers: the request's earlier answers, in both sides' tokens.
        sandy_input_ids: Sandy's tokens of the input when Sandy tokenizes on its
            own; None when Sandy reads the request's own tokens.
    """

    room: Optional[FrankyRoom] = None
    chain: Optional[FrankyChain] = None
    earlier_answers: tuple[GeneratedSpan, ...] = ()
    sandy_input_ids: Optional[list[int]] = None

    @staticmethod
    def of_input(
        obj: GenerateReqInput, *, prompt_len: int, on_sandy: bool
    ) -> Optional[FrankyRequest]:
        """
        Args:
            prompt_len: the request's input tokens on this engine.
            on_sandy: whether this engine is Sandy, which reads its own input.
        Returns:
            The checked request, or None when the request names no Franky field.
        Raises:
            ValueError: a field contradicts the others or the input.
        """
        if (
            obj.franky_room is None
            and obj.franky_chain is None
            and not obj.franky_generated_spans
            and obj.franky_sandy_input_ids is None
            and obj.franky_sandy_spans is None
        ):
            return None
        sandy_input_len = (
            None
            if obj.franky_sandy_input_ids is None
            else len(obj.franky_sandy_input_ids)
        )
        chain = (
            None
            if obj.franky_chain is None
            else FrankyChain.from_request(
                franky_chain=obj.franky_chain,
                rid=obj.rid,
                prompt_len=prompt_len,
                sandy_input_len=sandy_input_len,
                on_sandy=on_sandy,
            )
        )
        return FrankyRequest(
            room=(
                None
                if obj.franky_room is None
                else msgspec.convert(obj.franky_room, FrankyRoom)
            ),
            chain=chain,
            earlier_answers=checked_earlier_answers(
                obj.franky_generated_spans or [],
                generation_start=prompt_len
                - (0 if chain is None else chain.num_generated_before),
                sandy_answers=obj.franky_sandy_spans,
                sandy_generation_start=(
                    chain.sandy_answer_start(sandy_input_len=sandy_input_len)
                    if chain is not None and sandy_input_len is not None
                    else sandy_input_len
                ),
            ),
            sandy_input_ids=obj.franky_sandy_input_ids,
        )

    @staticmethod
    def of(req: Req) -> FrankyRequest:
        """The Franky request ``req`` carries; an empty one when it carries none."""
        return _NO_REQUEST if req.franky is None else req.franky

    def spans(self, *, num_input_tokens: int) -> FrankyGeneratedSpans:
        """Which taps each token of the request reads: its earlier answers, then
        the answer it generates now, after its chain's earlier blocks.
        """
        return FrankyGeneratedSpans.of(
            earlier_answers=self.earlier_answers,
            num_input_tokens=num_input_tokens,
            chain=self.chain,
            sandy_input_len=(
                None if self.sandy_input_ids is None else len(self.sandy_input_ids)
            ),
        )

    def sandy_tokens(self, input_ids: Sequence[int]) -> Sequence[int]:
        """The tokens Sandy reads for a request with these input ids; Ricky's
        taps count them.
        """
        return input_ids if self.sandy_input_ids is None else self.sandy_input_ids


_NO_REQUEST = FrankyRequest()
