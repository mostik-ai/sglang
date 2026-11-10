"""Franky end to end on two GPUs: a Sandy server writes taps, NIXL moves them to a
Ricky server, and Ricky's readers read them, chunk by chunk as they land. Ricky's
greedy logprobs must match HF transformers running Sandy's model with the
encoders and Ricky's model with the readers; without the readers they must not.
"""

import json
import os
import shutil
import tempfile
import threading
import unittest

import requests
import torch
from safetensors.torch import save_file
from torch.nn import functional
from transformers import AutoTokenizer

from sglang.kernels.ops.attention.fla.chunk_delta_h import CHUNK_SIZE as _STATE_CHUNK
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    find_available_port,
    popen_launch_server,
)

register_cuda_ci(est_time=300, stage="base-b", runner_config="2-gpu-large")

_MODEL = "Qwen/Qwen3.5-0.8B"
_LAYERS = [6, 12, 18, 24]
_TAP_SIZE = 256
_MLP_SIZE = 256
_PAGE_SIZE = 16
_NEW_TOKENS = 8
_TEXT = (
    "The river ran through the old town, past the mill and the bakery, under the "
    "stone bridge where children threw pebbles into the water every evening. "
)


def _hf_model():
    from transformers import AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_pretrained(_MODEL, dtype=torch.bfloat16)
    model = model.cuda().eval()
    return model, model.model.language_model


def _layer_output(output):
    return output[0] if isinstance(output, tuple) else output


def _reader(weights, index, layer_output, taps):
    """The PR's fixed-query reader on one sequence: token ``p`` reads taps
    ``0 .. min(p, tap_len - 1)``."""
    prefix = f"readers.{index}."
    positions = torch.arange(layer_output.shape[0], device=taps.device)
    visible = (
        torch.arange(taps.shape[0], device=taps.device)[None] <= positions[:, None]
    )
    heads = weights[f"{prefix}queries"].expand(layer_output.shape[0], -1, -1)
    attended = functional.scaled_dot_product_attention(
        heads.transpose(0, 1), taps[None], taps[None], attn_mask=visible
    )
    joined = attended.transpose(0, 1).flatten(start_dim=1)
    hidden = functional.gelu(
        functional.linear(
            joined, weights[f"{prefix}mlp.0.weight"], weights[f"{prefix}mlp.0.bias"]
        )
    )
    return functional.linear(
        hidden, weights[f"{prefix}mlp.2.weight"], weights[f"{prefix}mlp.2.bias"]
    )


@unittest.skipUnless(torch.cuda.device_count() >= 2, "requires 2 GPUs")
class TestFrankyEndToEnd(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.mkdtemp(prefix="franky_e2e_")
        cls.sandy_weights, cls.ricky_weights = cls._write_files()
        cls.bootstrap_port = find_available_port(18996)
        cls.sandy_url = f"http://127.0.0.1:{find_available_port(31200)}"
        cls.ricky_url = f"http://127.0.0.1:{find_available_port(31300)}"
        cls.processes = [
            cls._launch("sandy", cls.sandy_url, gpu=0),
            cls._launch("ricky", cls.ricky_url, gpu=1),
        ]

    @classmethod
    def tearDownClass(cls):
        for process in cls.processes:
            kill_process_tree(process.pid)
        shutil.rmtree(cls.directory, ignore_errors=True)

    @classmethod
    def _launch(cls, side, url, *, gpu):
        args = [
            "--franky-side", side,
            "--franky-config", os.path.join(cls.directory, "franky.json"),
            "--franky-weights", os.path.join(cls.directory, f"{side}.safetensors"),
            "--page-size", str(_PAGE_SIZE),
            # Small chunks, so Ricky's chunks wait for taps as they land.
            "--chunked-prefill-size", "64",
            "--mem-fraction-static", "0.6",
            "--base-gpu-id", str(gpu),
        ]  # fmt: skip
        if torch.cuda.get_device_capability()[0] >= 10:
            # On Blackwell, a Qwen3.5 rule replaces --page-size unless the
            # attention backend is set.
            args += ["--attention-backend", "trtllm_mha"]
        if side == "sandy":
            args += ["--franky-bootstrap-port", str(cls.bootstrap_port)]
        return popen_launch_server(
            _MODEL, url, timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH, other_args=args
        )

    @classmethod
    def _write_files(cls):
        """Seeded weights whose taps have unit RMS, so readers attend softly as
        trained ones do, and whose readers add a fifth of the residual's RMS."""
        model, language_model = _hf_model()
        rms = {}
        for layer in _LAYERS:
            language_model.layers[layer - 1].register_forward_hook(
                lambda module, inputs, output, layer=layer: rms.__setitem__(
                    layer, _layer_output(output).float().pow(2).mean().sqrt().item()
                )
            )
        from transformers import AutoTokenizer

        sample = AutoTokenizer.from_pretrained(_MODEL).encode(_TEXT * 8)
        with torch.no_grad():
            language_model(input_ids=torch.tensor([sample], device="cuda"))
        hidden = language_model.config.hidden_size
        del model, language_model
        torch.cuda.empty_cache()

        generator = torch.Generator().manual_seed(0)

        def randn(*shape, std):
            return (torch.randn(*shape, generator=generator) * std).to(torch.bfloat16)

        sandy, ricky = {}, {}
        for index, layer in enumerate(_LAYERS):
            sandy[f"encoders.{index}.projection.weight"] = randn(
                _TAP_SIZE, hidden, std=1 / (rms[layer] * hidden**0.5)
            )
            ricky[f"readers.{index}.queries"] = randn(1, _TAP_SIZE, std=1.0)
            ricky[f"readers.{index}.mlp.0.weight"] = randn(
                _MLP_SIZE, _TAP_SIZE, std=0.05
            )
            ricky[f"readers.{index}.mlp.0.bias"] = randn(_MLP_SIZE, std=0.02)
            ricky[f"readers.{index}.mlp.2.weight"] = randn(
                hidden, _MLP_SIZE, std=0.2 * rms[layer] / (0.2 * _MLP_SIZE**0.5)
            )
            ricky[f"readers.{index}.mlp.2.bias"] = randn(hidden, std=0.0)
        save_file(sandy, os.path.join(cls.directory, "sandy.safetensors"))
        save_file(ricky, os.path.join(cls.directory, "ricky.safetensors"))
        config = {
            "encoder": {
                "type": "singular",
                "layers": _LAYERS,
                "output_size": _TAP_SIZE,
            },
            "reader": {
                "type": "fixed_query",
                "layers": _LAYERS,
                "num_queries": 1,
                "mlp_hidden_size": _MLP_SIZE,
            },
            "injector": {"type": "residual"},
            "tap_page_size": _PAGE_SIZE,
        }
        with open(os.path.join(cls.directory, "franky.json"), "w") as file:
            json.dump(config, file)
        return sandy, ricky

    def _generate(self, ids, room):
        """Send the prompt to Sandy and to Ricky in one room; return Ricky's
        greedy tokens and their logprobs."""
        answers = self._send(ids, room)
        pairs = answers[self.ricky_url]["meta_info"]["output_token_logprobs"]
        return [pair[1] for pair in pairs], torch.tensor([pair[0] for pair in pairs])

    def _send(self, ids, room, **block):
        franky_room = {
            "bootstrap_host": "127.0.0.1",
            "bootstrap_port": self.bootstrap_port,
            "room": room,
        }
        answers = {}

        def post(url, sampling_params):
            answers[url] = requests.post(
                f"{url}/generate",
                json={
                    "input_ids": ids,
                    "sampling_params": sampling_params,
                    "return_logprob": True,
                    "franky_room": franky_room,
                    **block,
                },
                timeout=600,
            ).json()

        threads = [
            threading.Thread(target=post, args=(self.ricky_url, {"temperature": 0, "max_new_tokens": _NEW_TOKENS})),
            threading.Thread(target=post, args=(self.sandy_url, {"max_new_tokens": 1})),
        ]  # fmt: skip
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return answers

    def _reference(self, prompts, outputs):
        """HF logprobs of Ricky's tokens, with the readers and without them."""
        model, language_model = _hf_model()
        weights = {name: tensor.cuda() for name, tensor in self.sandy_weights.items()}
        weights.update(
            {name: tensor.cuda() for name, tensor in self.ricky_weights.items()}
        )
        state = {}
        for index, layer in enumerate(_LAYERS):

            def hook(module, inputs, output, index=index):
                hidden = _layer_output(output)
                if state["mode"] == "taps":
                    state["taps"][index] = functional.linear(
                        hidden[0], weights[f"encoders.{index}.projection.weight"]
                    )
                    return output
                if state["mode"] == "plain":
                    return output
                extra = _reader(weights, index, hidden[0], state["taps"][index])
                hidden = hidden + extra[None]
                return (hidden, *output[1:]) if isinstance(output, tuple) else hidden

            language_model.layers[layer - 1].register_forward_hook(hook)
        results = []
        with torch.no_grad():
            for ids, (tokens, _) in zip(prompts, outputs):
                state.update(mode="taps", taps={})
                language_model(input_ids=torch.tensor([ids], device="cuda"))
                row = {}
                for mode in ("readers", "plain"):
                    state["mode"] = mode
                    sequence = torch.tensor([ids + tokens[:-1]], device="cuda")
                    logits = model(input_ids=sequence).logits[0, len(ids) - 1 :].float()
                    logprobs = torch.log_softmax(logits, dim=-1)
                    row[mode] = logprobs.gather(
                        1, torch.tensor(tokens, device="cuda")[:, None]
                    )[:, 0].cpu()
                    row[f"{mode}_argmax"] = logprobs.argmax(-1).tolist()
                results.append(row)
        return results

    def test_ricky_reads_the_taps_sandy_wrote(self):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(_MODEL)
        # One prompt inside one chunk, one across many chunks and pages.
        prompts = [tokenizer.encode(_TEXT), tokenizer.encode(_TEXT * 12)]
        outputs = [self._generate(ids, room=4100 + i) for i, ids in enumerate(prompts)]
        for (tokens, logprobs), ref in zip(outputs, self._reference(prompts, outputs)):
            error = (logprobs - ref["readers"]).abs().max().item()
            same_argmax = sum(a == b for a, b in zip(ref["readers_argmax"], tokens))
            readers_effect = (ref["plain"] - ref["readers"]).abs().max().item()
            self.assertLess(error, 0.2)
            self.assertGreaterEqual(same_argmax, _NEW_TOKENS - 1)
            self.assertGreater(readers_effect, 4 * error)

    def test_a_chains_next_block_reuses_its_held_prompt_and_reads_new_taps(self):
        prompt = AutoTokenizer.from_pretrained(_MODEL).encode(_TEXT * 6)
        chain = "chain4200"
        first = self._send(
            prompt,
            4200,
            rid=f"{chain}-1",
            franky_chain={"id": chain, "last_block": False},
        )
        pairs = first[self.ricky_url]["meta_info"]["output_token_logprobs"]
        context = prompt + [pair[1] for pair in pairs]
        second = self._send(
            context,
            4201,
            rid=f"{chain}-2",
            franky_chain={"id": chain, "block": 2, "last_block": True},
        )

        whole_chunks = len(prompt) - len(prompt) % _STATE_CHUNK
        for url in (self.sandy_url, self.ricky_url):
            self.assertGreaterEqual(
                second[url]["meta_info"]["cached_tokens"], whole_chunks, url
            )
        pairs = second[self.ricky_url]["meta_info"]["output_token_logprobs"]
        tokens = [pair[1] for pair in pairs]
        logprobs = torch.tensor([pair[0] for pair in pairs])
        [ref] = self._reference([context], [(tokens, logprobs)])
        self.assertLess((logprobs - ref["readers"]).abs().max().item(), 0.2)
        for url in (self.sandy_url, self.ricky_url):
            ended = requests.post(
                f"{url}/franky/end_chain", json={"chain_id": chain}, timeout=60
            )
            self.assertEqual(ended.status_code, 200, url)


if __name__ == "__main__":
    unittest.main()
