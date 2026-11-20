from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Optional

import msgspec

from sglang.srt.franky.cache.generated_spans import FrankyGeneratedSpans
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
        chain: the chain the request is part of, if any.
        tap_runs: on Ricky, which taps the forward of each token reads; None for
            a request that reads none.
        taps_only: on Sandy, the request runs its prompt for the taps a Ricky
            leg beside it reads, and writes nothing.
        sandy_input_ids: Sandy's tokens of the input when Sandy tokenizes on its
            own; None when Sandy reads the request's own tokens.
    """

    room: Optional[FrankyRoom] = None
    chain: Optional[FrankyChain] = None
    tap_runs: Optional[FrankyGeneratedSpans] = None
    taps_only: bool = False
    sandy_input_ids: Optional[list[int]] = None

    @staticmethod
    def of_input(
        obj: GenerateReqInput, *, prompt_len: int, on_sandy: bool, passes_taps: bool
    ) -> Optional[FrankyRequest]:
        """
        Args:
            prompt_len: the request's input tokens on this engine.
            on_sandy: whether this engine is Sandy, which reads no taps.
            passes_taps: whether this engine has Franky parts; one without only
                writes a chain's text.
        Returns:
            The checked request, or None when the request names no Franky field.
        Raises:
            ValueError: a field contradicts the others or the input.
        """
        if (
            obj.franky_room is None
            and obj.franky_chain is None
            and obj.franky_tap_runs is None
            and not obj.franky_taps_only
            and obj.franky_sandy_input_ids is None
        ):
            return None
        if obj.franky_taps_only and (obj.franky_room is None or not on_sandy):
            raise ValueError(
                f"Request {obj.rid} asks for taps only, which a Sandy request with "
                "a franky_room computes."
            )
        if obj.franky_room is not None and not passes_taps:
            raise ValueError(
                f"Request {obj.rid} names a franky_room, but this engine runs "
                "without --franky-config and passes no taps."
            )
        room = (
            None
            if obj.franky_room is None
            else msgspec.convert(obj.franky_room, FrankyRoom)
        )
        reads_taps = room is not None and not on_sandy
        if obj.franky_tap_runs is not None and not reads_taps:
            raise ValueError(
                f"Request {obj.rid} names franky_tap_runs, which only a Ricky "
                "request with a franky_room reads."
            )
        num_taps = (
            prompt_len
            if obj.franky_sandy_input_ids is None
            else len(obj.franky_sandy_input_ids)
        )
        return FrankyRequest(
            room=room,
            chain=(
                None
                if obj.franky_chain is None
                else FrankyChain.from_request(
                    franky_chain=obj.franky_chain, rid=obj.rid
                )
            ),
            tap_runs=(
                _tap_runs(obj.franky_tap_runs, prompt_len=prompt_len, num_taps=num_taps)
                if reads_taps
                else None
            ),
            taps_only=obj.franky_taps_only,
            sandy_input_ids=obj.franky_sandy_input_ids,
        )

    @staticmethod
    def of(req: Req) -> FrankyRequest:
        """The Franky request ``req`` carries; an empty one when it carries none."""
        return _NO_REQUEST if req.franky is None else req.franky

    def sandy_tokens(self, input_ids: Sequence[int]) -> Sequence[int]:
        """The tokens Sandy reads for a request with these input ids; Ricky's
        taps count them.
        """
        return input_ids if self.sandy_input_ids is None else self.sandy_input_ids


def _tap_runs(
    runs: Optional[list[list[int]]], *, prompt_len: int, num_taps: int
) -> FrankyGeneratedSpans:
    if runs is None:
        return FrankyGeneratedSpans.after_input(
            num_input_tokens=prompt_len, num_taps=num_taps
        )
    return FrankyGeneratedSpans.of_runs(
        runs, num_input_tokens=prompt_len, num_taps=num_taps
    )


_NO_REQUEST = FrankyRequest()
