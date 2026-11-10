from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch.distributed as dist

from sglang.srt.disaggregation.utils import all_reduce_flags

if TYPE_CHECKING:
    from sglang.srt.disaggregation.pp_consensus_store import PPConsensusStore


class FrankyRankReports:
    """Agreement on numbers, each under a key, across the GPUs of one DP rank: the
    lowest wins.

    Attention TP and CP ranks take the lowest with an all-reduce, as PD does. PP
    stages run different batches at one moment, so no all-reduce can join them:
    each stage also reports to stage 0 through ``PPConsensusStore``.
    """

    def __init__(
        self,
        *,
        attn_tp_cpu_group: dist.ProcessGroup,
        attn_cp_cpu_group: dist.ProcessGroup,
        pp_store: Optional[PPConsensusStore],
    ) -> None:
        self._attn_tp_cpu_group = attn_tp_cpu_group
        self._attn_cp_cpu_group = attn_cp_cpu_group
        self._pp_store = pp_store

    def report(self, *, numbers: dict[str, int]) -> dict[str, int]:
        """Return the lowest number per key over this stage's attention TP and
        CP ranks, and report it to stage 0 under PP. Every such rank must call
        this at the same step, with the same keys in the same order.
        """
        lowest = list(numbers.values())
        for group in (self._attn_tp_cpu_group, self._attn_cp_cpu_group):
            lowest = all_reduce_flags(flags=lowest, group=group, op=dist.ReduceOp.MIN)
        agreed = dict(zip(numbers, lowest, strict=True))
        if self._pp_store is not None:
            for key, number in agreed.items():
                self._pp_store[key] = number
        return agreed

    def lowest(self, *, key: str, agreed: int) -> Optional[int]:
        """On stage 0, ``agreed`` from ``report`` lowered by the other stages'
        reports; None until every stage has reported.
        """
        if self._pp_store is None:
            return agreed
        reports = self._pp_store.collect(key)
        if any(number is None for number in reports):
            return None
        return min(reports)

    def forget(self, *, key: str) -> None:
        if self._pp_store is not None:
            self._pp_store.pop(key, None)
