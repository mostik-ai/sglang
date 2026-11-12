from __future__ import annotations

from enum import Enum
from typing import Annotated

import msgspec

from sglang.srt.franky.parts.encoder.config import EncoderConfig
from sglang.srt.franky.parts.injector.config import InjectorConfig
from sglang.srt.franky.parts.reader.config import ReaderConfig


class FrankySide(str, Enum):
    NULL = "null"
    SANDY = "sandy"
    RICKY = "ricky"


class FrankyConfig(
    msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    """One file, read by both sides. Each part names its implementation in
    ``type``. Reader ``i`` reads encoder ``i``.
    """

    encoder: EncoderConfig
    reader: ReaderConfig
    injector: InjectorConfig
    # Each KV page pairs with one tap page, so both sides run this --page-size.
    tap_page_size: Annotated[int, msgspec.Meta(gt=1)]

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
