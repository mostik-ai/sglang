from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from typing_extensions import override

from sglang.srt.franky.schedulers.chain import FrankyChainScheduler
from sglang.srt.franky.schedulers.sandy import prepare_sandy_request
from sglang.srt.franky.transfer.relay.ends import relay_from_prefill
from sglang.srt.franky.transfer.relay.sender import FrankyRelaySender

if TYPE_CHECKING:
    from sglang.srt.franky.cache.sandy_tap_cache import SandyTapCache
    from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
    from sglang.srt.managers.schedule_batch import Req

_TAPS_ONLY_REFUSAL = (
    "A taps-only Sandy request prefills only, so it needs a Sandy outside PD."
)


class SandyPrefillScheduler(FrankyChainScheduler):
    """A Sandy PD prefill engine: runs each Sandy call's trace as any PD prefill
    does, and relays the trace's taps to the call's Sandy decode with its KV.
    The decode writes the call and sends every tap to Ricky's room, so this
    engine sends none to Ricky and serves no taps-only request.
    """

    @override
    def init_disaggregation(self) -> None:
        super().init_disaggregation()
        self.init_franky_relay()

    def init_franky_relay(self) -> None:
        """PD prefill's end of the tap relay to Sandy decode, given to PD's KV
        manager so each KV room also carries the trace's taps.
        """
        self.franky_relay = relay_from_prefill(
            self.disagg_prefill_bootstrap_queue.kv_manager,
            buffers=self._tap_cache.buffers,
            server_args=self.server_args,
        )

    @override
    def handle_generate_request(
        self,
        recv_req: TokenizedGenerateReqInput,
        *,
        mm_input_error: Optional[str] = None,
    ) -> None:
        taps_only = recv_req.franky is not None and recv_req.franky.taps_only
        refusal = _TAPS_ONLY_REFUSAL if taps_only else prepare_sandy_request(recv_req)
        super().handle_generate_request(
            recv_req, mm_input_error=refusal or mm_input_error
        )

    @override
    def send_kv_chunk(
        self, req: Req, last_chunk: bool = False, end_idx: Optional[int] = None
    ) -> None:
        if last_chunk:
            self._relay_taps(req)
        super().send_kv_chunk(req, last_chunk=last_chunk, end_idx=end_idx)

    def _relay_taps(self, req: Req) -> None:
        sender = req.disagg_kv_sender
        if not isinstance(sender, FrankyRelaySender) or sender.relays_taps:
            return
        slots = self.req_to_token_pool.req_to_token[
            req.kv.req_pool_idx, : len(req.origin_input_ids)
        ]
        kv_pages = (slots[:: self.page_size] // self.page_size).cpu().numpy()
        sender.relay_taps(
            tap_pages=self._tap_cache.pages_of_kv_pages(
                kv_pages=kv_pages, kv_page_size=self.page_size
            )
        )

    @property
    def _tap_cache(self) -> SandyTapCache:
        return self.tp_worker.model_runner.franky_parts.tap_cache
