from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
from typing_extensions import override

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.nixl.conn import NixlKVReceiver
from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.franky.transfer.manager import FrankyTapManager
    from sglang.srt.franky.transfer.messages import FrankyRoom


class FrankyReceiver(NixlKVReceiver):
    """One request's taps arriving on one Ricky rank over PD's NIXL transfer.
    It reaches the Sandy engine as its polls find it ready.
    """

    def __init__(
        self,
        *,
        manager: FrankyTapManager,
        franky_room: FrankyRoom,
        tap_pages: np.ndarray,
        num_tokens: int,
    ) -> None:
        """
        Args:
            tap_pages: int32 tap pages on this rank for the whole prompt.
        """
        super().__init__(manager, franky_room.bootstrap_address, franky_room.room)
        manager.track_room(franky_room.room)
        self.tap_pages = tap_pages
        self.num_tokens = num_tokens
        self.pages_registered = False

    @override
    def poll(self) -> int:
        if not self.pages_registered and self.conclude_state is None:
            self._register_pages()
        return super().poll()

    def landed(self) -> int:
        """
        Returns:
            The prompt tokens whose taps have landed on this rank.
        """
        landed_pages = self.kv_mgr.landed_pages(self.bootstrap_room)
        return min(landed_pages * self.kv_mgr.kv_args.page_size, self.num_tokens)

    def writes_stopped(self) -> bool:
        """
        Returns:
            Whether no sending rank can write into this rank's tap pages any more:
            the room ended, or every sender confirmed this rank's abort.
        """
        if not self.pages_registered:
            return True
        state = self.poll()
        if state == KVPoll.Success or not self.abort_notified:
            return state in (KVPoll.Success, KVPoll.Failed)
        if not self.kv_mgr.enable_deferred_decode_kv_release:
            return True
        return self.kv_mgr.is_abort_release_safe(
            self.bootstrap_room, self.required_prefill_response_num
        )

    def close(self) -> None:
        self.clear()
        self.kv_mgr.forget_room(self.bootstrap_room)

    def _register_pages(self) -> None:
        if not self.kv_mgr.try_ensure_parallel_info(self.bootstrap_addr):
            return
        sending_dp_rank = self._sending_dp_rank()
        if sending_dp_rank is None:
            return
        self.init(sending_dp_rank)
        if self.conclude_state == KVPoll.Failed:
            return
        self.send_metadata(kv_indices=self.tap_pages, aux_index=0)
        self.pages_registered = True

    def _sending_dp_rank(self) -> Optional[int]:
        info = self.kv_mgr.prefill_info_table[self.bootstrap_addr]
        if info.dp_size == 1:
            return 0
        if (
            info.follow_bootstrap_room
            and not envs.SGLANG_DISAGGREGATION_FORCE_QUERY_PREFILL_DP_RANK.get()
        ):
            return self.bootstrap_room % info.dp_size
        dp_ranks = self.query_prefill_dp_ranks(
            self.bootstrap_addr, [self.bootstrap_room]
        )
        return dp_ranks.get(str(self.bootstrap_room))
