"""Config fields of the ``franky`` namespace."""

from __future__ import annotations

from typing import (
    Literal,
    Optional,
)

import msgspec

from sglang.srt.arg_groups.arg_utils import A


class Franky(msgspec.Struct):
    """Namespace ``franky``."""

    _NS_PATH = "franky"

    franky_side: A[
        Literal["null", "sandy", "ricky"],
        "Side of a Franky model pair. 'sandy' is an engine regime that runs the "
        "frozen model and its encoders on input tokens only. 'ricky' runs the "
        "model with readers, with or without PD disaggregation. 'null' disables "
        "Franky.",
    ] = "null"
    franky_config: A[
        Optional[str],
        "Path to the Franky config JSON. Use the same file on both sides.",
    ] = None
    franky_weights: A[
        Optional[str],
        "Path to this side's Franky weights: encoders on sandy, readers on ricky.",
    ] = None
    franky_bootstrap_port: A[
        int,
        "Tap bootstrap server port. A sandy engine serves on it; a ricky PD "
        "prefill engine serves the relay to its decode on it; a ricky PD decode "
        "engine finds that relay on it. Default is 8996.",
    ] = 8996
    franky_chain_hold_seconds: A[
        float,
        "How long a sandy or ricky engine keeps a chain's cached prefix locked "
        "after one of its blocks, waiting for the next block. Sandy's must outlast "
        "a block's decode on ricky. Default is 60.",
    ] = 60.0
    franky_chain_hold_ratio: A[
        float,
        "Share of the KV pool that chain holds may lock; past it the oldest holds "
        "are released. Default is 0.25.",
    ] = 0.25
