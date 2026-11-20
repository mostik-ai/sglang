from __future__ import annotations

from sglang.srt.franky.config import FrankyTapRole
from sglang.srt.franky.schedulers.chain import FrankyChainScheduler
from sglang.srt.franky.schedulers.ricky import RickyScheduler
from sglang.srt.franky.schedulers.ricky_decode import RickyDecodeScheduler
from sglang.srt.franky.schedulers.sandy import SandyScheduler
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.runtime_context import get_disagg


def franky_scheduler_class() -> type[Scheduler]:
    """The scheduler of a Franky engine: one that sends taps, one that reads
    them, and one that only writes a chain's text.
    """
    tap_role = FrankyTapRole.of_running_engine()
    if tap_role == FrankyTapRole.NONE:
        return FrankyChainScheduler
    if tap_role == FrankyTapRole.SENDS:
        return SandyScheduler
    if get_disagg().disaggregation_mode == "decode":
        return RickyDecodeScheduler
    return RickyScheduler
