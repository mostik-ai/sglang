from __future__ import annotations

from typing_extensions import override

from sglang.srt.franky.schedulers.sandy import SandyScheduler
from sglang.srt.franky.transfer.relay.ends import relay_to_decode
from sglang.srt.franky.transfer.relay.sandy_target import SandyRelayTarget


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
        tap_cache = self.tp_worker.model_runner.franky_parts.tap_cache
        self.franky_relay_manager = relay_to_decode(
            self.disagg_decode_prealloc_queue.kv_manager,
            buffers=tap_cache.buffers,
            server_args=self.server_args,
            target=SandyRelayTarget(tap_cache=tap_cache, kv_page_size=self.page_size),
        )
