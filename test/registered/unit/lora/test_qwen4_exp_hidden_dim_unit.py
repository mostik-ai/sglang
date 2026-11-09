"""Unit tests for Qwen4-Exp's LoRA hidden-dim resolution.

Qwen4-Exp (`Qwen/Qwen3.8-Flash-Next`) gates its attention output, so `q_proj`
carries a gate row for every query row, and its shared expert is
`shared_expert_intermediate_size` wide. The language model's `get_hidden_dim`
(inherited from Qwen3.5) knows both. `Qwen4ExpForConditionalGeneration` wraps
that language model and must delegate to it: without the delegation the LoRA
memory pool uses `get_default_hidden_dim`, which sizes `qkv_proj` without the
gate rows and the shared expert from `intermediate_size`, and adapter loading
fails with `LoRA buffer shape ... does not match weight shape ...`.

These tests call the method on stand-ins built around a real
`Qwen4ExpTextConfig`: no CUDA, no server, no weights.

Usage:
    python -m pytest test/registered/unit/lora/test_qwen4_exp_hidden_dim_unit.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

# CPU-only unit test; no CUDA/distributed dependencies.
register_cpu_ci(est_time=11, suite="base-a-test-cpu")

import unittest

from sglang.srt.configs.qwen4_exp import Qwen4ExpTextConfig
from sglang.srt.lora.utils import get_default_hidden_dim, get_hidden_dim
from sglang.srt.models.qwen4_exp import (
    Qwen4ExpForConditionalGeneration,
    Qwen4ExpVLModel,
)

HIDDEN_SIZE = 2560
HEAD_DIM = 256
NUM_HEADS = 24
NUM_KV_HEADS = 2
SHARED_EXPERT_SIZE = 640


def _make_fake_qwen4_exp():
    """A `Qwen4ExpForConditionalGeneration` stand-in holding a language model
    stand-in, both built without `__init__`, with the released model's sizes."""
    config = Qwen4ExpTextConfig(
        hidden_size=HIDDEN_SIZE,
        head_dim=HEAD_DIM,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        shared_expert_intermediate_size=SHARED_EXPERT_SIZE,
        moe_intermediate_size=SHARED_EXPERT_SIZE,
    )
    language_model = Qwen4ExpVLModel.__new__(Qwen4ExpVLModel)
    language_model.config = config
    model = Qwen4ExpForConditionalGeneration.__new__(Qwen4ExpForConditionalGeneration)
    # nn.Module.__setattr__ refuses a child module before __init__ has run
    object.__setattr__(model, "model", language_model)
    return model, config


class TestQwen4ExpLoRAHiddenDims(unittest.TestCase):
    def setUp(self):
        self.model, self.config = _make_fake_qwen4_exp()

    def test_qkv_proj_counts_the_output_gate_rows_of_q_proj(self):
        gated_q_heads = 2 * NUM_HEADS
        expected = (HIDDEN_SIZE, HEAD_DIM * (gated_q_heads + 2 * NUM_KV_HEADS))
        self.assertEqual(self.model.get_hidden_dim("qkv_proj", layer_idx=3), expected)

    def test_o_proj_reads_the_ungated_heads(self):
        expected = (NUM_HEADS * HEAD_DIM, HIDDEN_SIZE)
        self.assertEqual(self.model.get_hidden_dim("o_proj", layer_idx=3), expected)

    def test_the_shared_expert_uses_its_own_width(self):
        self.assertEqual(
            self.model.get_hidden_dim("gate_up_proj", layer_idx=0),
            (HIDDEN_SIZE, 2 * SHARED_EXPERT_SIZE),
        )
        self.assertEqual(
            self.model.get_hidden_dim("down_proj", layer_idx=0),
            (SHARED_EXPERT_SIZE, HIDDEN_SIZE),
        )

    def test_the_lora_memory_pool_gets_the_model_dims_not_the_generic_fallback(self):
        """`lora.utils.get_hidden_dim` is what the memory pool calls."""
        for module_name in ("qkv_proj", "gate_up_proj", "down_proj"):
            resolved = get_hidden_dim(module_name, self.config, self.model, 0)
            self.assertEqual(resolved, self.model.get_hidden_dim(module_name, 0))
            self.assertNotEqual(
                resolved, get_default_hidden_dim(module_name, self.config, 0)
            )


if __name__ == "__main__":
    unittest.main()
