"""The Franky config file and ``handle_franky`` are the only guards between a bad
launch and a failure deep inside a running engine.
"""

import copy
import json
import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import msgspec

from sglang.srt.arg_groups import resolution_hooks as hooks_module
from sglang.srt.arg_groups.franky_hook import handle_franky
from sglang.srt.arg_groups.overrides import declare_resolution
from sglang.srt.arg_groups.resolution_hooks import register_resolution_hook
from sglang.srt.franky.config import FrankyConfig
from sglang.srt.franky.parts.encoder.all.encoder import AllEncoder
from sglang.srt.franky.parts.encoder.masked.encoder import MaskedEncoder
from sglang.srt.franky.parts.encoder.singular.encoder import SingularEncoder
from sglang.srt.franky.parts.injector.residual.config import ResidualInjectorConfig
from sglang.srt.franky.parts.reader.fixed_query.reader import FixedQueryReader
from sglang.srt.franky.parts.reader.residual_query.reader import ResidualQueryReader
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    Phase,
    with_phase,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

_CONFIG = {
    "encoder": {"type": "masked", "layers": [4, 8, 12], "output_size": 256},
    "reader": {
        "type": "fixed_query",
        "layers": [1, 4, 9],
        "num_queries": 1,
        "mlp_hidden_size": 1024,
    },
    "injector": {"type": "residual"},
    "tap_page_size": 64,
}


class _WithFiles(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.directory = tempfile.mkdtemp(prefix="franky_config_")
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)

    def _write_config(self, config) -> str:
        path = os.path.join(self.directory, "franky.json")
        with open(path, "w") as file:
            file.write(config if isinstance(config, str) else json.dumps(config))
        return path


class TestFrankyConfig(_WithFiles):
    def _load(self, *, encoder_type="masked", reader_type="fixed_query"):
        config = copy.deepcopy(_CONFIG)
        config["encoder"]["type"] = encoder_type
        config["reader"]["type"] = reader_type
        return FrankyConfig.load(self._write_config(config))

    def test_each_encoder_type_reads_its_layers(self):
        for encoder_type, expected in {
            "all": [(4, 8, 12), (4, 8, 12), (4, 8, 12)],
            "masked": [(4,), (4, 8), (4, 8, 12)],
            "singular": [(4,), (8,), (12,)],
        }.items():
            with self.subTest(encoder_type):
                encoder = self._load(encoder_type=encoder_type).encoder
                self.assertEqual(
                    [encoder.input_layers(index=index) for index in range(3)],
                    expected,
                )

    def test_type_names_pick_their_implementations(self):
        """The names are a contract with the config files people write."""
        for encoder_type, encoder_cls in {
            "all": AllEncoder,
            "masked": MaskedEncoder,
            "singular": SingularEncoder,
        }.items():
            with self.subTest(encoder_type):
                loaded = self._load(encoder_type=encoder_type)
                self.assertIs(loaded.encoder.encoder_cls, encoder_cls)
        for reader_type, reader_cls in {
            "fixed_query": FixedQueryReader,
            "residual_query": ResidualQueryReader,
        }.items():
            with self.subTest(reader_type):
                loaded = self._load(reader_type=reader_type)
                self.assertIs(loaded.reader.reader_cls, reader_cls)
        self.assertIsInstance(self._load().injector, ResidualInjectorConfig)

    def test_bad_configs_are_refused(self):
        for reason, edit in {
            "Invalid value 'linear'": lambda c: c["encoder"].update(type="linear"),
            "unknown field `output_sizes`": lambda c: c["encoder"].update(
                output_sizes=256
            ),
            "unknown field `scale`": lambda c: c["injector"].update(scale=2),
            "missing required field `tap_page_size`": lambda c: c.pop("tap_page_size"),
            "3 encoders and 2 readers": lambda c: c["reader"].update(layers=[1, 4]),
            "at `\\$.encoder.layers\\[0\\]`": lambda c: c["encoder"].update(
                layers=[0, 8, 12]
            ),
            "at `\\$.encoder.layers`": lambda c: c["encoder"].update(layers=[]),
            "got \\[8, 4, 12\\]": lambda c: c["encoder"].update(layers=[8, 4, 12]),
            "got \\[1, 1, 9\\]": lambda c: c["reader"].update(layers=[1, 1, 9]),
            "at `\\$.encoder.output_size`": lambda c: c["encoder"].update(
                output_size=0
            ),
            "at `\\$.reader.num_queries`": lambda c: c["reader"].update(num_queries=0),
            "at `\\$.reader.mlp_hidden_size`": lambda c: c["reader"].update(
                mlp_hidden_size=0
            ),
            "at `\\$.tap_page_size`": lambda c: c.update(tap_page_size=1),
        }.items():
            with self.subTest(reason):
                config = copy.deepcopy(_CONFIG)
                edit(config)
                path = self._write_config(config)
                with self.assertRaisesRegex(ValueError, reason):
                    FrankyConfig.load(path)

    def test_an_error_names_the_file(self):
        path = self._write_config("not json")
        with self.assertRaisesRegex(ValueError, path):
            FrankyConfig.load(path)


class TestHandleFranky(_WithFiles):
    def _server_args(
        self,
        *,
        side,
        config=_CONFIG,
        is_multimodal=False,
        is_hybrid_swa=False,
        num_hidden_layers=16,
        **fields,
    ):
        server_args = ServerArgs(model_path="dummy")
        for name, value in {
            "franky_side": side,
            "franky_config": self._write_config(config),
            "franky_weights": "franky.safetensors",
            "page_size": 64,
            "chunked_prefill_size": 8192,
            "cuda_graph_config": CudaGraphConfig(),
            **fields,
        }.items():
            msgspec.structs.force_setattr(server_args, name, value)
        server_args._model_config = SimpleNamespace(
            is_multimodal=is_multimodal,
            is_hybrid_swa=is_hybrid_swa,
            num_hidden_layers=num_hidden_layers,
        )
        return server_args

    def test_refused_on_both_sides(self):
        for reason, fields in {
            "speculative decoding": {"speculative_algorithm": "EAGLE"},
            "LoRA": {"enable_lora": True},
            "--dp-size": {"dp_size": 2},
            "--enable-attn-tp-input-scattered": {
                "enable_attn_tp_input_scattered": True
            },
            "--enable-layernorm-sp": {"enable_layernorm_sp": True},
            "--enable-two-batch-overlap": {"enable_two_batch_overlap": True},
            "--enable-pdmux": {"enable_pdmux": True},
            "page size 16": {"page_size": 16},
            "--enable-unified-memory": {"enable_unified_memory": True},
            "--enable-hisparse": {"enable_hisparse": True},
            "--enable-kv-cache-sharding": {"enable_kv_cache_sharding": True},
            "--dcp-size": {"dcp_size": 2},
            "hybrid SWA": {"is_hybrid_swa": True},
            "--enable-hierarchical-cache": {"enable_hierarchical_cache": True},
            "--enable-lmcache": {"enable_lmcache": True},
        }.items():
            for side in ("sandy", "ricky"):
                with self.subTest(reason, side=side):
                    server_args = self._server_args(side=side, **fields)
                    with self.assertRaisesRegex(ValueError, reason):
                        handle_franky(server_args)

    def test_refused_on_sandy_only(self):
        for reason, fields in {
            "--disaggregation-mode prefill": {
                "disaggregation_mode": "prefill",
                "disaggregation_transfer_backend": "nixl",
            },
            "encoder layer 12": {"num_hidden_layers": 10},
        }.items():
            with self.subTest(reason):
                server_args = self._server_args(side="sandy", **fields)
                with self.assertRaisesRegex(ValueError, reason):
                    handle_franky(server_args)
                handle_franky(self._server_args(side="ricky", **fields))

    def test_refused_on_ricky_only(self):
        shallow_encoders = copy.deepcopy(_CONFIG)
        shallow_encoders["encoder"]["layers"] = [1, 2, 3]
        wide_taps = copy.deepcopy(_CONFIG)
        wide_taps["encoder"]["output_size"] = 512
        odd_taps = copy.deepcopy(_CONFIG)
        odd_taps["encoder"]["output_size"] = 100
        for reason, fields in {
            "encoder output_size 512": {"config": wide_taps},
            "encoder output_size 100": {"config": odd_taps},
            "--chunked-prefill-size": {"chunked_prefill_size": -1},
            "--disable-radix-cache": {"disable_radix_cache": True},
            "reader layer 9": {"config": shallow_encoders, "num_hidden_layers": 8},
            "--cuda-graph-backend-prefill full": {
                "cuda_graph_config": with_phase(
                    CudaGraphConfig(), Phase.PREFILL, backend=Backend.FULL
                )
            },
        }.items():
            with self.subTest(reason):
                server_args = self._server_args(side="ricky", **fields)
                with self.assertRaisesRegex(ValueError, reason):
                    handle_franky(server_args)
                handle_franky(self._server_args(side="sandy", **fields))

    def test_ricky_pd_refusals(self):
        pd = {
            "disaggregation_mode": "decode",
            "disaggregation_transfer_backend": "nixl",
        }
        for reason, fields in {
            "--disaggregation-transfer-backend mooncake": {
                **pd,
                "disaggregation_transfer_backend": "mooncake",
            },
            "--pp-size above 1": {**pd, "pp_size": 2},
            "--disaggregation-decode-enable-radix-cache": {
                **pd,
                "disaggregation_decode_enable_radix_cache": True,
            },
            "--disaggregation-decode-retraction-backup cpu_tensor": {
                **pd,
                "disaggregation_decode_retraction_backup": "cpu_tensor",
            },
        }.items():
            with self.subTest(reason):
                server_args = self._server_args(side="ricky", **fields)
                with self.assertRaisesRegex(ValueError, reason):
                    handle_franky(server_args)

    def test_ricky_pd_decode_needs_no_chunks_and_drops_retraction_backups(self):
        server_args = self._server_args(
            side="ricky",
            disaggregation_mode="decode",
            disaggregation_transfer_backend="nixl",
            chunked_prefill_size=-1,
        )

        handle_franky(server_args)

        self.assertIn(
            ("handle_franky", {"disaggregation_decode_retraction_backup": "none"}),
            server_args._resolved_overrides,
        )

    def test_each_side_picks_its_own_attention_dp_size(self):
        for side, attn_dp_size in (("sandy", 2), ("ricky", 4)):
            with self.subTest(side):
                handle_franky(self._server_args(side=side, attn_dp_size=attn_dp_size))

    def test_multimodal_models_are_allowed(self):
        """Qwen3.5 checkpoints are multimodal, and --language-model-only does not
        support them, so only multimodal requests can be refused.
        """
        for side in ("sandy", "ricky"):
            with self.subTest(side):
                handle_franky(self._server_args(side=side, is_multimodal=True))

    def test_every_reason_comes_in_one_error(self):
        server_args = self._server_args(
            side="sandy", enable_lora=True, enable_hierarchical_cache=True
        )
        with self.assertRaisesRegex(ValueError, "LoRA.*--enable-hierarchical-cache"):
            handle_franky(server_args)

    def test_paths_are_required(self):
        for field in ("franky_config", "franky_weights"):
            with self.subTest(field):
                server_args = self._server_args(side="sandy", **{field: None})
                with self.assertRaisesRegex(ValueError, "--franky-config"):
                    handle_franky(server_args)

    def test_a_solo_server_is_not_checked(self):
        handle_franky(
            self._server_args(
                side="null",
                franky_config=None,
                page_size=1,
                speculative_algorithm="EAGLE",
            )
        )


class TestThroughThePipeline(_WithFiles):
    """A real ``resolve_once`` on a tiny model config on disk."""

    def setUp(self):
        super().setUp()
        patcher = patch.dict(hooks_module._HOOKS, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        with open(os.path.join(self.directory, "config.json"), "w") as file:
            json.dump(
                {
                    "architectures": ["LlamaForCausalLM"],
                    "model_type": "llama",
                    "hidden_size": 16,
                    "intermediate_size": 32,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 2,
                    "num_hidden_layers": 4,
                    "vocab_size": 128,
                    "max_position_embeddings": 2048,
                },
                file,
            )
        config = copy.deepcopy(_CONFIG)
        config["encoder"]["layers"] = [2, 4]
        config["reader"]["layers"] = [1, 3]
        self.config_path = self._write_config(config)

    def _server_args(self, *, side):
        return ServerArgs(
            model_path=self.directory,
            device="cuda",
            franky_side=side,
            franky_config=self.config_path,
            franky_weights="franky.safetensors",
            page_size=64,
        )

    def test_a_good_launch_resolves_on_each_side(self):
        for side in ("sandy", "ricky"):
            with self.subTest(side):
                self._server_args(side=side).resolve_once()

    def test_a_page_size_changed_by_a_late_step_is_refused(self):
        """Steps after the flags can still change it, so the check runs last."""

        @register_resolution_hook("handle_other_validations")
        def _change_page_size(server_args, previous):
            previous(server_args)
            declare_resolution(server_args, "test", page_size=16)

        with self.assertRaisesRegex(ValueError, "page size 16"):
            self._server_args(side="sandy").resolve_once()


if __name__ == "__main__":
    unittest.main()
