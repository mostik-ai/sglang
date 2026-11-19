from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


def sandy_input_ids(req: Req) -> Sequence[int]:
    """The tokens Sandy reads for a Ricky request: the ids the router sent in
    ``franky_sandy_input_ids`` when Sandy tokenizes on its own, else the
    request's own input ids. Ricky's taps count these tokens.
    """
    if req.franky_sandy_input_ids is not None:
        return req.franky_sandy_input_ids
    return req.origin_input_ids
