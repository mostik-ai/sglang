from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import msgspec

if TYPE_CHECKING:
    import torch

    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class LayerOutput(msgspec.Struct, frozen=True, kw_only=True):
    """A model's residual right after one layer, in the form the model keeps it.
    Each form has its own subclass and the injectors that read and change it.

    Attributes:
        rows: the rows of ``forward_batch`` this layer ran, as indices into them;
            None when it ran all of them.
    """

    forward_batch: ForwardBatch
    rows: Optional[torch.Tensor] = None
