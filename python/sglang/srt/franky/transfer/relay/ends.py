from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Optional

from sglang.srt.disaggregation.common.conn import CommonKVBootstrapServer
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.franky.transfer.manager import FrankyTapManager
from sglang.srt.runtime_context import get_device, get_franky, get_parallel, get_serving

if TYPE_CHECKING:
    from sglang.srt.franky.cache.tap_buffers import TapBuffers
    from sglang.srt.franky.transfer.relay.manager import FrankyRelayManager
    from sglang.srt.franky.transfer.relay.target import FrankyRelayTarget
    from sglang.srt.server_args import ServerArgs


@dataclasses.dataclass(frozen=True)
class FrankySendingEnd:
    """An engine's sending end of tap transfers over PD's NIXL classes, at its
    ``--franky-bootstrap-port``: the bootstrap server on its first rank, and a
    tap manager on every rank, since a receiver waits for every rank and the
    server for every rank to register.

    Attributes:
        server: the bootstrap server; None off the first rank.
    """

    server: Optional[CommonKVBootstrapServer]
    manager: FrankyTapManager

    @staticmethod
    def build(*, buffers: TapBuffers, server_args: ServerArgs) -> FrankySendingEnd:
        parallel = get_parallel()
        port = get_franky().franky_bootstrap_port
        server = (
            CommonKVBootstrapServer(host=get_serving().host, port=port)
            if parallel.tp_rank == 0 and parallel.pp_rank == 0
            else None
        )
        manager = FrankyTapManager.from_buffers(
            buffers=buffers,
            side=DisaggregationMode.PREFILL,
            server_args=server_args,
            bootstrap_port=port,
            engine_rank=parallel.tp_rank,
            pp_rank=parallel.pp_rank,
            gpu_id=get_device().gpu_id,
        )
        return FrankySendingEnd(server=server, manager=manager)


def relay_from_prefill(
    kv_manager: FrankyRelayManager, *, buffers: TapBuffers, server_args: ServerArgs
) -> FrankySendingEnd:
    """PD prefill's end of the tap relay: a sending end, given to PD's KV
    manager so each KV room also carries the request's taps.
    """
    sending = FrankySendingEnd.build(buffers=buffers, server_args=server_args)
    kv_manager.attach_tap_relay(tap_relay=sending.manager)
    return sending


def relay_to_decode(
    kv_manager: FrankyRelayManager,
    *,
    buffers: TapBuffers,
    server_args: ServerArgs,
    target: FrankyRelayTarget,
) -> FrankyTapManager:
    """PD decode's end of the tap relay: this rank's receiving tap manager,
    given to PD's KV manager with ``target``, where each room's taps land.
    """
    parallel = get_parallel()
    manager = FrankyTapManager.from_buffers(
        buffers=buffers,
        side=DisaggregationMode.DECODE,
        server_args=server_args,
        bootstrap_port=None,
        engine_rank=parallel.attn_tp_rank,
        pp_rank=parallel.pp_rank,
        gpu_id=get_device().gpu_id,
        senders_split_alike=target.senders_split_alike,
    )
    kv_manager.attach_tap_relay(tap_relay=manager, relay_target=target)
    return manager
