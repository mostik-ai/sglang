from __future__ import annotations

from typing_extensions import override

from sglang.srt.franky.parts.encoder.linear.encoder import LinearEncoder


class SingularEncoder(LinearEncoder):
    """Each encoder takes only its own layer."""

    @staticmethod
    @override
    def select_inputs(*, layers: tuple[int, ...], index: int) -> tuple[int, ...]:
        return (layers[index],)
