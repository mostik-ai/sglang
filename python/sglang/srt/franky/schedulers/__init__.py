from __future__ import annotations

from sglang.srt.franky.config import FrankySide
from sglang.srt.franky.schedulers.ricky import RickyScheduler
from sglang.srt.franky.schedulers.ricky_decode import RickyDecodeScheduler
from sglang.srt.franky.schedulers.sandy import SandyScheduler
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.runtime_context import get_disagg


def franky_scheduler_class(side: str) -> type[Scheduler]:
    if FrankySide(side) == FrankySide.SANDY:
        return SandyScheduler
    if get_disagg().disaggregation_mode == "decode":
        return RickyDecodeScheduler
    return RickyScheduler
