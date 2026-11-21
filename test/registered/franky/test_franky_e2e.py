"""Franky end to end: a Sandy server writes taps, NIXL moves them to a Ricky
server, and Ricky's readers read them, chunk by chunk as they land. Ricky's
greedy logprobs must match HF transformers running Sandy's model with the
encoders and Ricky's model with the readers; without the readers they must not.
A Sandy that writes part of the answer sends the taps of its own tokens too,
which Ricky reads as its tap runs say. With TP 2 on both sides, every
attention-TP rank keeps its stripe of the taps, each Sandy rank sends its pages
to the Ricky ranks that keep them, and Ricky's ranks merge their readers'
attention.
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

register_cuda_ci(est_time=900, stage="base-b", runner_config="4-gpu-b200")

_MODEL = "Qwen/Qwen3.5-0.8B"
_LAYERS = [6, 12, 18, 24]
_TAP_SIZE = 256
_MLP_SIZE = 256
_PAGE_SIZE = 16
_NEW_TOKENS = 8
# Tokens a Sandy call writes, more than a tap page so its taps span pages.
_WRITTEN = 24
# Sandy's own tokens keep every 4th of Ricky's, as the router's
# --franky-sandy-keep-every 4 sends them.
_SANDY_KEEP_EVERY = 4
_TEXT = (
    "The river ran through the old town, past the mill and the bakery, under the "
    "stone bridge where children threw pebbles into the water every evening. "
)


_HANDED_OUT_PORTS: set[int] = set()


def _free_port(base: int) -> int:
    """A free port that no earlier draw in this file handed out. An SGLang
    server binds its HTTP port with SO_REUSEPORT, so two servers that drew one
    port would share it without an error."""
    while True:
        port = find_available_port(base)
        if port not in _HANDED_OUT_PORTS:
            _HANDED_OUT_PORTS.add(port)
            return port


def _hf_model():
    from transformers import AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_pretrained(_MODEL, dtype=torch.bfloat16)
    model = model.cuda().eval()
    return model, model.model.language_model


def _layer_output(output):
    return output[0] if isinstance(output, tuple) else output


def _taps_read_by(token, runs):
    """The taps the forward that makes ``token`` reads, as ``franky_tap_runs``
    say: those of the last run that starts at or before it."""
    taps = 0
    for start, run_taps in runs:
        if start <= token:
            taps = run_taps
    return taps


def _block_runs(prompt_len, block_size, num_blocks):
    """The tap runs of a chain's leg after ``num_blocks - 1`` blocks, with Sandy
    writing nothing: each block reads the taps of all before it."""
    return [[prompt_len + block * block_size] * 2 for block in range(num_blocks)]


def _reader(weights, index, layer_output, taps, num_taps):
    prefix = f"readers.{index}."
    reads = num_taps > 0
    queries = functional.linear(
        layer_output[reads], weights[f"{prefix}query_projection.weight"]
    )
    heads = queries.view(queries.shape[0], -1, taps.shape[-1])
    visible = (
        torch.arange(taps.shape[0], device=taps.device)[None] < num_taps[reads, None]
    )
    attended = functional.scaled_dot_product_attention(
        heads.transpose(0, 1), taps[None], taps[None], attn_mask=visible
    )
    joined = attended.transpose(0, 1).flatten(start_dim=1)
    hidden = functional.gelu(
        functional.linear(
            joined, weights[f"{prefix}mlp.0.weight"], weights[f"{prefix}mlp.0.bias"]
        )
    )
    output = torch.zeros_like(layer_output)
    output[reads] = functional.linear(
        hidden, weights[f"{prefix}mlp.2.weight"], weights[f"{prefix}mlp.2.bias"]
    )
    return output


class _FrankyPair(CustomTestCase):
    """Franky servers on this node with seeded encoders and readers, and the HF
    reference of what Ricky's readers must read."""

    tp_size = 1
    # Whether Sandy reads Ricky's tokens; when not, it reads its own, which are
    # every _SANDY_KEEP_EVERY-th of Ricky's.
    shared_tokenizer = True

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.mkdtemp(prefix="franky_e2e_")
        cls.sandy_weights, cls.ricky_weights = cls._write_files()
        cls.bootstrap_port = _free_port(18996)
        cls.sandy_url = f"http://127.0.0.1:{_free_port(31200)}"
        cls.ricky_url = f"http://127.0.0.1:{_free_port(31300)}"
        cls.processes = cls._servers()

    @classmethod
    def tearDownClass(cls):
        for process in cls.processes:
            kill_process_tree(process.pid)
        shutil.rmtree(cls.directory, ignore_errors=True)

    @classmethod
    def _servers(cls):
        return [
            cls._launch("sandy", cls.sandy_url, gpu=0, tp_size=cls.tp_size),
            cls._launch("ricky", cls.ricky_url, gpu=cls.tp_size, tp_size=cls.tp_size),
        ]

    @classmethod
    def _launch(cls, side, url, *, gpu, tp_size, extra=()):
        args = [
            "--franky-side", side,
            "--franky-config", os.path.join(cls.directory, "franky.json"),
            "--franky-weights", os.path.join(cls.directory, f"{side}.safetensors"),
            "--page-size", str(_PAGE_SIZE),
            # Small chunks, so Ricky's chunks wait for taps as they land.
            "--chunked-prefill-size", "64",
            "--mem-fraction-static", "0.6",
            "--base-gpu-id", str(gpu),
            "--tp-size", str(tp_size),
            *extra,
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
            ricky[f"readers.{index}.query_projection.weight"] = randn(
                _TAP_SIZE, hidden, std=1 / (rms[layer] * hidden**0.5)
            )
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
                "type": "residual_query",
                "layers": _LAYERS,
                "num_queries": 1,
                "mlp_hidden_size": _MLP_SIZE,
            },
            "injector": {"sandy": {"type": "residual"}, "ricky": {"type": "residual"}},
            "tap_page_size": _PAGE_SIZE,
            "shared_tokenizer": cls.shared_tokenizer,
        }
        with open(os.path.join(cls.directory, "franky.json"), "w") as file:
            json.dump(config, file)
        return sandy, ricky

    def _sandy_ids(self, ids):
        """The tokens Sandy reads for Ricky's ``ids``."""
        return ids if self.shared_tokenizer else ids[::_SANDY_KEEP_EVERY]

    def _sandy_fields(self, ids):
        """What Ricky's request adds when Sandy reads its own tokens."""
        if self.shared_tokenizer:
            return {}
        return {"franky_sandy_input_ids": self._sandy_ids(ids)}

    def _assert_reads_the_taps(self, prompts, *, first_room):
        """Ricky's greedy logprobs follow HF with the readers, which move them."""
        outputs = [
            self._generate(ids, room=first_room + i) for i, ids in enumerate(prompts)
        ]
        for (tokens, logprobs), ref in zip(outputs, self._reference(prompts, outputs)):
            error = (logprobs - ref["readers"]).abs().max().item()
            same_argmax = sum(a == b for a, b in zip(ref["readers_argmax"], tokens))
            readers_effect = (ref["plain"] - ref["readers"]).abs().max().item()
            self.assertLess(error, 0.2)
            self.assertGreaterEqual(same_argmax, _NEW_TOKENS - 1)
            self.assertGreater(readers_effect, 4 * error)

    def _generate(self, ids, room):
        """Send the prompt to Sandy and to Ricky in one room; return Ricky's
        greedy tokens and their logprobs."""
        answers = self._send(ids, room)
        pairs = answers[self.ricky_url]["meta_info"]["output_token_logprobs"]
        return [pair[1] for pair in pairs], torch.tensor([pair[0] for pair in pairs])

    def _room(self, room):
        return {
            "bootstrap_host": "127.0.0.1",
            "bootstrap_port": self.bootstrap_port,
            "room": room,
        }

    def _send(self, ids, room, ricky_sampling=None, ricky_fields=None, chain=None):
        """Ricky's request beside a Sandy leg that sends the taps of ``ids``.
        ``chain``: the chain's id and block, and Ricky's blocks."""
        answers = {}

        def post(url, input_ids, sampling_params, fields):
            answers[url] = requests.post(
                f"{url}/generate",
                json={
                    "input_ids": input_ids,
                    "sampling_params": sampling_params,
                    "return_logprob": True,
                    "franky_room": self._room(room),
                    **fields,
                },
                timeout=600,
            ).json()

        ricky_fields = {**self._sandy_fields(ids), **(ricky_fields or {})}
        sandy_fields = {"franky_taps_only": True}
        if chain is not None:
            rid = f"{chain['id']}-{chain['block']}"
            ricky_fields.update(rid=rid, franky_chain=chain)
            sandy_chain = {key: chain[key] for key in ("id", "block")}
            sandy_fields.update(rid=rid, franky_chain=sandy_chain)
        threads = [
            threading.Thread(target=post, args=(self.ricky_url, ids, ricky_sampling or {"temperature": 0, "max_new_tokens": _NEW_TOKENS}, ricky_fields)),
            threading.Thread(target=post, args=(self.sandy_url, self._sandy_ids(ids), {}, sandy_fields)),
        ]  # fmt: skip
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return answers

    def _reference(self, prompts, outputs, runs=None):
        """HF logprobs of Ricky's tokens, with the readers and without them.
        ``runs``: per prompt, the tap runs Ricky read; by default its answer
        reads all of the prompt's taps."""
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
                extra = _reader(
                    weights, index, hidden[0], state["taps"][index], state["num_taps"]
                )
                hidden = hidden + extra[None]
                return (hidden, *output[1:]) if isinstance(output, tuple) else hidden

            language_model.layers[layer - 1].register_forward_hook(hook)
        results = []
        with torch.no_grad():
            for index, (ids, (tokens, _)) in enumerate(zip(prompts, outputs)):
                sandy_ids = self._sandy_ids(ids)
                prompt_runs = (
                    runs[index] if runs is not None else [[len(ids), len(sandy_ids)]]
                )
                num_taps = [
                    _taps_read_by(row + 1, prompt_runs)
                    for row in range(len(ids) + len(tokens) - 1)
                ]
                state.update(
                    mode="taps",
                    taps={},
                    num_taps=torch.tensor(num_taps, device="cuda"),
                )
                language_model(input_ids=torch.tensor([sandy_ids], device="cuda"))
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


@unittest.skipUnless(torch.cuda.device_count() >= 2, "requires 2 GPUs")
class TestFrankyEndToEnd(_FrankyPair):
    def test_ricky_reads_the_taps_sandy_wrote(self):
        tokenizer = AutoTokenizer.from_pretrained(_MODEL)
        # One prompt inside one chunk, one across many chunks and pages.
        prompts = [tokenizer.encode(_TEXT), tokenizer.encode(_TEXT * 12)]
        self._assert_reads_the_taps(prompts, first_room=4100)

    def test_a_chains_next_block_reuses_its_held_prompt_and_reads_new_taps(self):
        prompt = AutoTokenizer.from_pretrained(_MODEL).encode(_TEXT * 6)
        chain = "chain4200"
        blocks = {"block_size": _NEW_TOKENS}
        sampling = {"temperature": 0, "max_new_tokens": 4 * _NEW_TOKENS}
        first = self._send(
            prompt,
            4200,
            ricky_sampling=sampling,
            chain={"id": chain, "block": 1, "blocks": blocks},
        )
        self.assertEqual(
            first[self.ricky_url]["meta_info"]["finish_reason"],
            {"type": "franky_sandy_call", "next_block": 2},
        )
        pairs = first[self.ricky_url]["meta_info"]["output_token_logprobs"]
        self.assertEqual(len(pairs), _NEW_TOKENS)
        context = prompt + [pair[1] for pair in pairs]
        runs = _block_runs(len(prompt), _NEW_TOKENS, 2)
        second = self._send(
            context,
            4201,
            ricky_fields={"franky_tap_runs": runs},
            chain={"id": chain, "block": 2, "blocks": blocks},
        )

        whole_chunks = len(prompt) - len(prompt) % _STATE_CHUNK
        for url in (self.sandy_url, self.ricky_url):
            self.assertGreaterEqual(
                second[url]["meta_info"]["cached_tokens"], whole_chunks, url
            )
        pairs = second[self.ricky_url]["meta_info"]["output_token_logprobs"]
        tokens = [pair[1] for pair in pairs]
        logprobs = torch.tensor([pair[0] for pair in pairs])
        [ref] = self._reference([context], [(tokens, logprobs)], runs=[runs])
        self.assertLess((logprobs - ref["readers"]).abs().max().item(), 0.2)
        for url in (self.sandy_url, self.ricky_url):
            ended = requests.post(
                f"{url}/franky/end_chain", json={"chain_id": chain}, timeout=60
            )
            self.assertEqual(ended.status_code, 200, url)

    def test_a_next_block_recomputes_the_last_blocks_tokens_as_their_decode_did(self):
        # A short prompt and a long block make a wrong read show: block 1's last
        # token would read 47 taps more than the prompt's ones its decode read.
        prompt = AutoTokenizer.from_pretrained(_MODEL).encode(_TEXT)
        block_size = 48
        sampling = {"temperature": 0, "max_new_tokens": block_size, "ignore_eos": True}
        chain = "chain4300"
        blocks = {"block_size": block_size}
        first = self._send(
            prompt,
            4300,
            ricky_sampling=sampling,
            chain={"id": chain, "block": 1, "blocks": blocks},
        )
        decoded = first[self.ricky_url]["meta_info"]["output_token_logprobs"]
        self.assertEqual(len(decoded), block_size)
        second = self._send(
            prompt + [pair[1] for pair in decoded],
            4301,
            ricky_sampling=sampling,
            ricky_fields={
                "logprob_start_len": len(prompt) - 1,
                "franky_tap_runs": _block_runs(len(prompt), block_size, 2),
            },
            chain={"id": chain, "block": 2, "blocks": blocks},
        )

        recomputed = second[self.ricky_url]["meta_info"]["input_token_logprobs"]
        recomputed = recomputed[-block_size:]
        self.assertEqual(
            [pair[1] for pair in recomputed], [pair[1] for pair in decoded]
        )
        gap = (
            torch.tensor([pair[0] for pair in recomputed])
            - torch.tensor([pair[0] for pair in decoded])
        ).abs()
        # Prefill and decode kernels differ by about 0.01 on average here, a few
        # tokens up to 0.1; reading the block's own taps moves them about 0.07 on
        # average, some past 0.3.
        self.assertLess(gap.mean().item(), 0.03, gap.tolist())
        self.assertLess(gap.max().item(), 0.2, gap.tolist())
        for url in (self.sandy_url, self.ricky_url):
            requests.post(
                f"{url}/franky/end_chain", json={"chain_id": chain}, timeout=60
            )

    def test_ricky_reads_the_taps_of_the_tokens_sandy_wrote(self):
        # A short prompt whose continuation is new text, so the taps of
        # Sandy's tokens are most of what Ricky reads, not near copies of the
        # prompt's.
        prompt = AutoTokenizer.from_pretrained(_MODEL).encode(
            "Write a short story about a lighthouse keeper who finds a map."
        )
        room = self._room(4500)
        sandy = requests.post(
            f"{self.sandy_url}/generate",
            json={
                "input_ids": prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": _WRITTEN,
                    "ignore_eos": True,
                },
                "franky_room": room,
            },
            timeout=600,
        ).json()
        written = sandy["output_ids"]
        self.assertEqual(len(written), _WRITTEN)
        self.assertEqual(
            sandy["meta_info"]["finish_reason"], {"type": "length", "length": _WRITTEN}
        )
        trace = prompt + written
        runs = [[len(prompt), len(trace)]]
        ricky = requests.post(
            f"{self.ricky_url}/generate",
            json={
                "input_ids": trace,
                "sampling_params": {"temperature": 0, "max_new_tokens": _NEW_TOKENS},
                "return_logprob": True,
                "franky_room": room,
                "franky_tap_runs": runs,
            },
            timeout=600,
        ).json()

        pairs = ricky["meta_info"]["output_token_logprobs"]
        output = (
            [pair[1] for pair in pairs],
            torch.tensor([pair[0] for pair in pairs]),
        )
        without_written = [[len(prompt), len(prompt)]]
        ref, ref_without_written = self._reference(
            [trace, trace], [output, output], runs=[runs, without_written]
        )
        error = (output[1] - ref["readers"]).abs().max().item()
        written_effect = (ref["readers"] - ref_without_written["readers"]).abs().max()
        self.assertLess(error, 0.2)
        self.assertGreater(written_effect.item(), 4 * error)


@unittest.skipUnless(torch.cuda.device_count() >= 4, "requires 4 GPUs")
class TestFrankyEndToEndTp2(TestFrankyEndToEnd):
    tp_size = 2


@unittest.skipUnless(torch.cuda.device_count() >= 4, "requires 4 GPUs")
class TestFrankyPdEndToEnd(_FrankyPair):
    """Sandy, Ricky PD prefill on two ranks and Ricky PD decode on one: Sandy's
    rank sends its taps to both prefill ranks, each keeping its stripe pages,
    and prefill's ranks relay their pages to decode's rank with the KV."""

    prefill_tp_size = 2
    decode_tp_size = 1

    @classmethod
    def _servers(cls):
        cls.relay_port = _free_port(19096)
        cls.pd_port = _free_port(19196)
        cls.prefill_url = cls.ricky_url
        cls.decode_url = f"http://127.0.0.1:{_free_port(31400)}"
        pd = ["--disaggregation-transfer-backend", "nixl"]
        relay = ["--franky-bootstrap-port", str(cls.relay_port)]
        return [
            cls._launch("sandy", cls.sandy_url, gpu=0, tp_size=1),
            cls._launch(
                "ricky",
                cls.prefill_url,
                gpu=1,
                tp_size=cls.prefill_tp_size,
                extra=[
                    *pd,
                    *relay,
                    "--disaggregation-mode",
                    "prefill",
                    "--disaggregation-bootstrap-port",
                    str(cls.pd_port),
                ],
            ),
            cls._launch(
                "ricky",
                cls.decode_url,
                gpu=1 + cls.prefill_tp_size,
                tp_size=cls.decode_tp_size,
                extra=[*pd, *relay, "--disaggregation-mode", "decode"],
            ),
        ]

    def _send(self, ids, room, ricky_sampling=None, ricky_fields=None, chain=None):
        franky_room = self._room(room)
        pd = {
            "bootstrap_host": "127.0.0.1",
            "bootstrap_port": self.pd_port,
            "bootstrap_room": room,
        }
        sampling = ricky_sampling or {"temperature": 0, "max_new_tokens": _NEW_TOKENS}
        answers = {}

        def post(key, url, payload):
            answers[key] = requests.post(
                f"{url}/generate", json=payload, timeout=600
            ).json()

        ricky = {
            "input_ids": ids,
            "sampling_params": sampling,
            "return_logprob": True,
            "franky_room": franky_room,
            **pd,
            **self._sandy_fields(ids),
            **(ricky_fields or {}),
        }
        sandy = {
            "input_ids": self._sandy_ids(ids),
            "franky_room": franky_room,
            "franky_taps_only": True,
        }
        threads = [
            threading.Thread(target=post, args=("prefill", self.prefill_url, ricky)),
            threading.Thread(
                target=post, args=(self.ricky_url, self.decode_url, ricky)
            ),
            threading.Thread(target=post, args=(self.sandy_url, self.sandy_url, sandy)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return answers

    def test_decode_reads_the_taps_prefill_relayed(self):
        tokenizer = AutoTokenizer.from_pretrained(_MODEL)
        prompts = [tokenizer.encode(_TEXT), tokenizer.encode(_TEXT * 12)]
        self._assert_reads_the_taps(prompts, first_room=5100)

    def test_requests_at_once_each_read_their_own_taps(self):
        tokenizer = AutoTokenizer.from_pretrained(_MODEL)
        prompts = [
            tokenizer.encode(_TEXT * repeat) + [100 + repeat]
            for repeat in (1, 3, 6, 9, 12)
        ]
        outputs = [None] * len(prompts)

        def generate(index):
            outputs[index] = self._generate(prompts[index], room=5200 + index)

        threads = [
            threading.Thread(target=generate, args=(index,))
            for index in range(len(prompts))
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        for (tokens, logprobs), ref in zip(outputs, self._reference(prompts, outputs)):
            self.assertLess((logprobs - ref["readers"]).abs().max().item(), 0.2)


@unittest.skipUnless(torch.cuda.device_count() >= 5, "requires 5 GPUs")
class TestFrankyPdDecodeTp2EndToEnd(TestFrankyPdEndToEnd):
    """Ricky PD decode on two ranks too: each decode rank reads only the prefill
    rank that keeps its stripe."""

    decode_tp_size = 2


@unittest.skipUnless(torch.cuda.device_count() >= 7, "requires 7 GPUs")
class TestFrankyPdPrefillTp4DecodeTp2EndToEnd(TestFrankyPdEndToEnd):
    """Ricky PD prefill on four ranks and decode on two: each decode rank reads
    the two prefill ranks that keep its pages."""

    prefill_tp_size = 4
    decode_tp_size = 2


@unittest.skipUnless(torch.cuda.device_count() >= 2, "requires 2 GPUs")
class TestFrankyOwnSandyTokensEndToEnd(_FrankyPair):
    """Sandy reads its own tokens, so Ricky's taps count Sandy's tokens. The longer
    prompt starts with the shorter one, so with one rank it reads the shorter
    one's whole stripes from the tap prefix tree."""

    shared_tokenizer = False

    def test_ricky_reads_the_taps_of_sandys_own_tokens(self):
        tokenizer = AutoTokenizer.from_pretrained(_MODEL)
        prompts = [tokenizer.encode(_TEXT * 4), tokenizer.encode(_TEXT * 12)]
        self._assert_reads_the_taps(prompts, first_room=4600)


@unittest.skipUnless(torch.cuda.device_count() >= 4, "requires 4 GPUs")
class TestFrankyOwnSandyTokensEndToEndTp2(TestFrankyOwnSandyTokensEndToEnd):
    tp_size = 2


@unittest.skipUnless(torch.cuda.device_count() >= 4, "requires 4 GPUs")
class TestFrankyOwnSandyTokensPdEndToEnd(TestFrankyPdEndToEnd):
    shared_tokenizer = False


@unittest.skipUnless(torch.cuda.device_count() >= 2, "requires 2 GPUs")
class TestFrankyTextOnlyPdEndToEnd(CustomTestCase):
    """A Ricky without Franky parts under PD, as a chain without taps runs it:
    its legs go from prefill to decode over NIXL's own classes, and decode ends
    a leg where its blocks call Sandy."""

    @classmethod
    def setUpClass(cls):
        cls.pd_port = _free_port(19296)
        cls.prefill_url = f"http://127.0.0.1:{_free_port(31500)}"
        cls.decode_url = f"http://127.0.0.1:{_free_port(31600)}"
        ricky = [
            "--franky-side", "ricky",
            "--disaggregation-transfer-backend", "nixl",
            "--mem-fraction-static", "0.6",
        ]  # fmt: skip
        if torch.cuda.get_device_capability()[0] >= 10:
            ricky += ["--attention-backend", "trtllm_mha"]
        prefill = [
            "--base-gpu-id", "0",
            "--disaggregation-mode", "prefill",
            "--disaggregation-bootstrap-port", str(cls.pd_port),
        ]  # fmt: skip
        decode = ["--base-gpu-id", "1", "--disaggregation-mode", "decode"]
        cls.processes = [
            popen_launch_server(
                _MODEL,
                url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[*ricky, *stage],
            )
            for url, stage in ((cls.prefill_url, prefill), (cls.decode_url, decode))
        ]

    @classmethod
    def tearDownClass(cls):
        for process in cls.processes:
            kill_process_tree(process.pid)

    def test_a_leg_ends_on_decode_where_its_blocks_call_sandy(self):
        leg = {
            "rid": "text-only-r1",
            "input_ids": AutoTokenizer.from_pretrained(_MODEL).encode(_TEXT),
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 64,
                "ignore_eos": True,
            },
            "franky_chain": {
                "id": "text-only",
                "block": 1,
                "blocks": {
                    "block_size": 8,
                    "decider": {
                        "type": "call_token_or_every_blocks",
                        "every_blocks": 2,
                    },
                },
            },
            "bootstrap_host": "127.0.0.1",
            "bootstrap_port": self.pd_port,
            "bootstrap_room": 6100,
        }
        answers = {}

        def post(url):
            answers[url] = requests.post(
                f"{url}/generate", json=leg, timeout=600
            ).json()

        threads = [
            threading.Thread(target=post, args=(url,))
            for url in (self.prefill_url, self.decode_url)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        meta_info = answers[self.decode_url]["meta_info"]
        self.assertEqual(meta_info["finish_reason"]["type"], "franky_sandy_call")
        self.assertEqual(meta_info["finish_reason"]["next_block"], 3)
        self.assertEqual(meta_info["completion_tokens"], 16)


if __name__ == "__main__":
    unittest.main()
