from __future__ import annotations

from typing_extensions import override

from sglang.srt.franky.parts.encoder.linear.encoder import LinearEncoder


class MaskedEncoder(LinearEncoder):
    """Each encoder takes the layers up to and including its own."""

    @staticmethod
    @override
    def select_inputs(*, layers: tuple[int, ...], index: int) -> tuple[int, ...]:
        return layers[: index + 1]
