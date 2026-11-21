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
        "Side of a Franky model pair. 'sandy' runs the frozen model: it writes "
        "what a chain asks and, with --franky-config, sends the taps of every "
        "token it ran. 'ricky' runs the model that writes the chain's blocks and, "
        "with --franky-config, reads the taps, with or without PD "
        "disaggregation. Without --franky-config the sides only write text. "
        "'null' disables Franky.",
    ] = "null"
    franky_config: A[
        Optional[str],
        "Path to the Franky config JSON of the pair's encoders and readers. Use "
        "the same file on both sides; omit it on both for text-only chains.",
    ] = None
    franky_weights: A[
        Optional[str],
        "Path to this side's Franky weights: encoders on sandy, readers on ricky; "
        "given with --franky-config.",
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
        "after one of the chain's requests, waiting for the next one, unless the "
        "router ends the chain first; a ricky that reads taps also keeps their "
        "pages. A held chain is in flight: other requests wait rather than take "
        "its memory. Sandy's must outlast ricky's blocks between two Sandy calls. "
        "Default is 60.",
    ] = 60.0
    franky_tap_pool_ratio: A[
        float,
        "Taps a ricky's tap pool holds per KV slot, split over its attention-TP "
        "ranks. Requests hold taps while they run and chains between two blocks; "
        "a prefix cache over sandy's tokens keeps the taps of finished requests "
        "until room is needed. A PD prefill engine fits far more KV with a lower "
        "ratio; with fewer taps, fewer requests read at once and the rest wait. "
        "Default is 1: taps for every KV slot.",
    ] = 1.0
