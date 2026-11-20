from __future__ import annotations

from typing_extensions import override

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.franky.schedulers.sandy import SandyScheduler
from sglang.srt.franky.transfer.manager import FrankyTapManager
from sglang.srt.franky.transfer.relay.sandy_target import SandyRelayTarget
from sglang.srt.runtime_context import get_device, get_parallel


class SandyDecodeScheduler(SandyScheduler):
    """A Sandy PD decode engine: each call's trace KV and taps come from its
    Sandy prefill, landing at the call's KV slots here, and PD hands the call to
    decode once both landed. Decode writes the call and sends every tap to
    Ricky's room, as an aggregated Sandy does.
    """

    @override
    def init_disaggregation(self) -> None:
        super().init_disaggregation()
        self.init_franky_relay()

    def init_franky_relay(self) -> None:
        """PD decode's end of the tap relay from Sandy prefill: this rank's tap
        manager, given to PD's KV manager so each KV room also brings its taps.
        """
        parallel = get_parallel()
        tap_cache = self.tp_worker.model_runner.franky_parts.tap_cache
        self.franky_relay_manager = FrankyTapManager.from_buffers(
            buffers=tap_cache.buffers,
            side=DisaggregationMode.DECODE,
            server_args=self.server_args,
            bootstrap_port=None,
            engine_rank=parallel.attn_tp_rank,
            pp_rank=parallel.pp_rank,
            gpu_id=get_device().gpu_id,
        )
        self.disagg_decode_prealloc_queue.kv_manager.attach_tap_relay(
            tap_relay=self.franky_relay_manager,
            relay_target=SandyRelayTarget(
                tap_cache=tap_cache, kv_page_size=self.page_size
            ),
        )
