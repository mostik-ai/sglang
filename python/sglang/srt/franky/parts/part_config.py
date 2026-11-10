from __future__ import annotations

from typing import Annotated

import msgspec

PositiveInt = Annotated[int, msgspec.Meta(gt=0)]


class LayersPartConfig(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
    tag_field="type",
):
    """A part with one entry per layer. Layer numbers start at 1 and mean "after
    this layer".
    """

    layers: Annotated[tuple[PositiveInt, ...], msgspec.Meta(min_length=1)]

    def __post_init__(self) -> None:
        if any(lower >= upper for lower, upper in zip(self.layers, self.layers[1:])):
            raise ValueError(f"layers must go up, got {list(self.layers)}")
