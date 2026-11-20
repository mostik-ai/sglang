from __future__ import annotations

from enum import Enum
from typing import Annotated

import msgspec

from sglang.srt.franky.parts.encoder.config import EncoderConfig
from sglang.srt.franky.parts.injector.config import InjectorsConfig
from sglang.srt.franky.parts.reader.config import ReaderConfig


class FrankySide(str, Enum):
    NULL = "null"
    SANDY = "sandy"
    RICKY = "ricky"


class FrankyTapRole(Enum):
    """What an engine does with taps: a side with Franky parts sends them
    (Sandy) or reads them (Ricky); one without only writes a chain's text."""

    SENDS = "sends"
    READS = "reads"
    NONE = "none"

    @staticmethod
    def of_engine(*, side: str, has_config: bool) -> FrankyTapRole:
        """
        Args:
            side: ``--franky-side``.
            has_config: whether the engine runs a ``--franky-config``.
        """
        if not has_config:
            return FrankyTapRole.NONE
        return FrankyTapRole.SENDS if side == FrankySide.SANDY else FrankyTapRole.READS


class FrankyConfig(
    msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    """One file, read by both sides. Each part names its implementation in
    ``type``. Reader ``i`` reads encoder ``i``.
    """

    encoder: EncoderConfig
    reader: ReaderConfig
    injector: InjectorsConfig
    # Tokens of a tap page, the unit taps are sent and read in. Sandy's taps live
    # at its KV slots, so its --page-size is a multiple of it; Ricky's KV pages
    # are any size, since Ricky keeps its taps in pages of their own.
    tap_page_size: Annotated[int, msgspec.Meta(gt=1)]
    # Whether Sandy reads Ricky's tokens. When it tokenizes on its own, each
    # request sends Sandy's tokens of its input and names its answers in both
    # sides' tokens.
    shared_tokenizer: bool = True

    def __post_init__(self) -> None:
        num_encoders = len(self.encoder.layers)
        num_readers = len(self.reader.layers)
        if num_encoders != num_readers:
            raise ValueError(
                f"reader i reads encoder i, but there are {num_encoders} encoders "
                f"and {num_readers} readers"
            )

    @staticmethod
    def load(path: str) -> FrankyConfig:
        with open(path, "rb") as file:
            data = file.read()
        try:
            return msgspec.json.decode(data, type=FrankyConfig)
        except msgspec.DecodeError as error:
            raise ValueError(f"Bad Franky config {path}: {error}") from error
