"""Franky changes SGLang by subclassing it, so a renamed base method would leave
a Franky override silently dead. Every override must carry @override and match a
base method.
"""

import inspect
import unittest

from sglang.srt.disaggregation.nixl.conn import (
    NixlKVManager,
    NixlKVReceiver,
    NixlKVSender,
)
from sglang.srt.franky.cache.kv_allocator import FrankyKvAllocator
from sglang.srt.franky.parts.encoder.all.encoder import AllEncoder
from sglang.srt.franky.parts.encoder.linear.encoder import LinearEncoder
from sglang.srt.franky.parts.encoder.masked.encoder import MaskedEncoder
from sglang.srt.franky.parts.encoder.singular.encoder import SingularEncoder
from sglang.srt.franky.parts.injector.base_config import BaseInjectorConfig
from sglang.srt.franky.parts.injector.residual.config import ResidualInjectorConfig
from sglang.srt.franky.parts.injector.residual.injector import ResidualInjector
from sglang.srt.franky.parts.reader.base_config import BaseReaderConfig
from sglang.srt.franky.parts.reader.query.config import QueryReaderConfig
from sglang.srt.franky.parts.reader.query.reader import QueryReader
from sglang.srt.franky.parts.reader.residual_query.reader import ResidualQueryReader
from sglang.srt.franky.parts.ricky_parts import RickyParts
from sglang.srt.franky.parts.sandy_parts import SandyParts
from sglang.srt.franky.schedulers.prefill_adder import (
    FrankyPrefillAdder,
    FrankyPrefillBudget,
)
from sglang.srt.franky.schedulers.ricky import RickyScheduler
from sglang.srt.franky.schedulers.ricky_decode import RickyDecodeScheduler
from sglang.srt.franky.schedulers.sandy import SandyScheduler
from sglang.srt.franky.schedulers.sandy_batch_result_processor import (
    SandyBatchResultProcessor,
)
from sglang.srt.franky.transfer.manager import FrankyTapManager
from sglang.srt.franky.transfer.receiver import FrankyReceiver
from sglang.srt.franky.transfer.relay.manager import FrankyRelayManager
from sglang.srt.franky.transfer.relay.receiver import FrankyRelayReceiver
from sglang.srt.franky.transfer.relay.sender import FrankyRelaySender
from sglang.srt.franky.transfer.senders.room import FrankyRoomSender
from sglang.srt.managers.schedule_policy import PrefillAdder
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.prefill_budget import PrefillBudget
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

_SUBCLASSES = (
    (SandyScheduler, Scheduler),
    (SandyBatchResultProcessor, SchedulerBatchResultProcessor),
    (RickyScheduler, Scheduler),
    (FrankyPrefillAdder, PrefillAdder),
    (FrankyPrefillBudget, PrefillBudget),
    (FrankyKvAllocator, PagedTokenToKVPoolAllocator),
    (AllEncoder, LinearEncoder),
    (MaskedEncoder, LinearEncoder),
    (SingularEncoder, LinearEncoder),
    (ResidualQueryReader, QueryReader),
    (ResidualInjector, ResidualInjector.__mro__[1]),
    (SandyParts, SandyParts.__mro__[1]),
    (RickyParts, RickyParts.__mro__[1]),
    (QueryReaderConfig, BaseReaderConfig),
    (ResidualInjectorConfig, BaseInjectorConfig),
    (FrankyTapManager, NixlKVManager),
    (FrankyRoomSender, NixlKVSender),
    (FrankyReceiver, NixlKVReceiver),
    (FrankyRelayManager, NixlKVManager),
    (FrankyRelaySender, NixlKVSender),
    (FrankyRelayReceiver, NixlKVReceiver),
    (RickyDecodeScheduler, Scheduler),
)


def _function_of(member):
    return member.__func__ if isinstance(member, staticmethod) else member


def _own_methods(cls):
    """Methods written in the class; dunders such as a dataclass's are not."""
    return {
        name: _function_of(member)
        for name, member in vars(cls).items()
        if inspect.isfunction(_function_of(member)) and not name.startswith("__")
    }


class TestOverrideMarkers(CustomTestCase):
    def test_each_override_is_marked_and_matches_its_base(self):
        for cls, base in _SUBCLASSES:
            for name, method in _own_methods(cls).items():
                with self.subTest(cls=cls.__name__, method=name):
                    marked = getattr(method, "__override__", False)
                    self.assertEqual(marked, hasattr(base, name))
                    if marked:
                        self.assertEqual(
                            list(inspect.signature(method).parameters),
                            list(inspect.signature(getattr(base, name)).parameters),
                        )


if __name__ == "__main__":
    unittest.main()
