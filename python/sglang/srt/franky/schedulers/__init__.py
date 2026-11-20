from __future__ import annotations

from sglang.srt.franky.config import FrankySide
from sglang.srt.franky.schedulers.chain import FrankyChainScheduler
from sglang.srt.franky.schedulers.ricky import RickyScheduler
from sglang.srt.franky.schedulers.ricky_decode import RickyDecodeScheduler
from sglang.srt.franky.schedulers.sandy import SandyScheduler
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.runtime_context import get_disagg, get_franky


def franky_scheduler_class(side: str) -> type[Scheduler]:
    """The scheduler of a Franky side: one that passes taps with
    ``--franky-config``, and one that only writes a chain's text without it.
    """
    if get_franky().franky_config is None:
        return FrankyChainScheduler
    if FrankySide(side) == FrankySide.SANDY:
        return SandyScheduler
    if get_disagg().disaggregation_mode == "decode":
        return RickyDecodeScheduler
    return RickyScheduler
