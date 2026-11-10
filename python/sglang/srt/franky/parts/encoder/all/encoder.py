from __future__ import annotations

from typing_extensions import override

from sglang.srt.franky.parts.encoder.linear.encoder import LinearEncoder


class AllEncoder(LinearEncoder):
    """Every encoder takes all encoder layers."""

    @staticmethod
    @override
    def select_inputs(*, layers: tuple[int, ...], index: int) -> tuple[int, ...]:
        return layers
