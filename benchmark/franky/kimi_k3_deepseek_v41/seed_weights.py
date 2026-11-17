"""Seeded Franky weights for a pair without trained ones, for benchmarks.

Writes sandy.safetensors (encoders) and ricky.safetensors (readers) for a Franky
config. Encoders and query projections keep their outputs near unit scale; a
reader's last layer is small (``--output-std``), so Ricky answers as it would
alone while every reader still runs and adds to the residual.

    python seed_weights.py --config franky-config.json \
        --sandy-model /models/Kimi-K3 --ricky-model /models/DeepSeek-V4.1-Flash \
        --output-dir /models/franky-k3-dsv41
"""

import argparse
import json
import os

import torch
from safetensors.torch import save_file


def hidden_size(model_dir: str) -> int:
    with open(os.path.join(model_dir, "config.json")) as file:
        config = json.load(file)
    return config.get("hidden_size") or config["text_config"]["hidden_size"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--sandy-model", required=True)
    parser.add_argument("--ricky-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-std", type=float, default=1e-4)
    args = parser.parse_args()

    with open(args.config) as file:
        config = json.load(file)
    sandy_hidden = hidden_size(args.sandy_model)
    ricky_hidden = hidden_size(args.ricky_model)
    tap_size = config["encoder"]["output_size"]
    num_queries = config["reader"]["num_queries"]
    mlp_size = config["reader"]["mlp_hidden_size"]
    generator = torch.Generator().manual_seed(args.seed)

    def randn(*shape: int, std: float) -> torch.Tensor:
        return (torch.randn(*shape, generator=generator) * std).to(torch.bfloat16)

    sandy, ricky = {}, {}
    for index in range(len(config["encoder"]["layers"])):
        sandy[f"encoders.{index}.projection.weight"] = randn(
            tap_size, sandy_hidden, std=sandy_hidden**-0.5
        )
        ricky[f"readers.{index}.query_projection.weight"] = randn(
            num_queries * tap_size, ricky_hidden, std=ricky_hidden**-0.5
        )
        ricky[f"readers.{index}.mlp.0.weight"] = randn(
            mlp_size, num_queries * tap_size, std=(num_queries * tap_size) ** -0.5
        )
        ricky[f"readers.{index}.mlp.0.bias"] = randn(mlp_size, std=0.0)
        ricky[f"readers.{index}.mlp.2.weight"] = randn(
            ricky_hidden, mlp_size, std=args.output_std
        )
        ricky[f"readers.{index}.mlp.2.bias"] = randn(ricky_hidden, std=0.0)
    os.makedirs(args.output_dir, exist_ok=True)
    save_file(sandy, os.path.join(args.output_dir, "sandy.safetensors"))
    save_file(ricky, os.path.join(args.output_dir, "ricky.safetensors"))
    print(
        f"wrote {len(sandy)} Sandy and {len(ricky)} Ricky tensors to {args.output_dir}: "
        f"Sandy hidden {sandy_hidden}, Ricky hidden {ricky_hidden}"
    )


if __name__ == "__main__":
    main()
