#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""최소 실행/학습/스윕 스크립트."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.bpe import BPETokenizer
from src.dataset import create_dataloader
from src.model import GPTModel, generate_text_simple
from src.train import calc_loss_batch, calc_loss_loader


def build_loaders(
    text: str,
    vocab_size: int,
    context_length: int,
    batch_size: int,
    tokenizer_chars: int,
    train_ratio: float,
):
    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.train(text[:tokenizer_chars])
    token_ids = tokenizer.encode(text)

    split_idx = max(int(len(token_ids) * train_ratio), context_length + 2)
    train_ids = token_ids[:split_idx]
    val_ids = token_ids[split_idx:]
    if len(val_ids) < context_length + 2:
        val_ids = train_ids[-(context_length + 2) :]

    train_loader = create_dataloader(
        train_ids,
        context_length=context_length,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = create_dataloader(
        val_ids,
        context_length=context_length,
        batch_size=batch_size,
        shuffle=False,
    )
    return tokenizer, train_loader, val_loader


def run_training_loop(
    model: GPTModel,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
):
    train_losses: list[float] = []
    val_losses: list[float] = []

    model.to(device)
    for _ in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for input_batch, target_batch in train_loader:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        train_losses.append(total_loss / max(num_batches, 1))
        val_losses.append(calc_loss_loader(val_loader, model, device))

    return train_losses, val_losses


def save_loss_plot(train_losses: list[float], val_losses: list[float], output_path: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def count_parameters(model: GPTModel) -> int:
    return sum(param.numel() for param in model.parameters())


def make_model_config(args, overrides: dict | None = None) -> dict:
    config = {
        "vocab_size": args.vocab_size,
        "context_length": args.context_length,
        "emb_dim": args.emb_dim,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "drop_rate": args.drop_rate,
        "qkv_bias": args.qkv_bias,
        "pre_norm": not args.post_norm,
    }
    if overrides:
        config.update(overrides)
    return config


def train_once(args, text: str, overrides: dict | None = None) -> dict:
    config = make_model_config(args, overrides)
    tokenizer, train_loader, val_loader = build_loaders(
        text=text,
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        batch_size=args.batch_size,
        tokenizer_chars=args.tokenizer_chars,
        train_ratio=args.train_ratio,
    )

    model = GPTModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=overrides.get("lr", args.lr) if overrides else args.lr)
    device = torch.device(args.device)

    start_time = time.time()
    train_losses, val_losses = run_training_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=args.num_epochs,
    )
    elapsed = time.time() - start_time

    sample_ids = tokenizer.encode(args.prompt, add_bos_eos=False)
    sample_tensor = torch.tensor([sample_ids], dtype=torch.long, device=device)
    generated = generate_text_simple(
        model=model,
        idx=sample_tensor,
        max_new_tokens=args.max_new_tokens,
        context_size=config["context_length"],
    )
    generated_text = tokenizer.decode(generated[0].tolist(), skip_special=True)

    final_val_loss = val_losses[-1]
    perplexity = math.exp(min(final_val_loss, 20.0))

    return {
        "config": config,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "final_train_loss": train_losses[-1],
        "final_val_loss": final_val_loss,
        "perplexity": perplexity,
        "num_params": count_parameters(model),
        "elapsed_sec": elapsed,
        "generated_text": generated_text,
    }


def run_train_command(args) -> None:
    text = Path(args.text_file).read_text(encoding="utf-8")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = train_once(args, text)
    save_loss_plot(result["train_losses"], result["val_losses"], output_dir / "loss_curve.png")
    (output_dir / "train_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("final_train_loss:", result["final_train_loss"])
    print("final_val_loss:", result["final_val_loss"])
    print("perplexity:", result["perplexity"])
    print("generated_text:")
    print(result["generated_text"])


def load_sweep_configs(args) -> list[dict]:
    if args.sweep_file:
        return json.loads(Path(args.sweep_file).read_text(encoding="utf-8"))
    return [
        {"emb_dim": 32, "n_layers": 1, "drop_rate": 0.0, "lr": 1e-3},
        {"emb_dim": 64, "n_layers": 1, "drop_rate": 0.0, "lr": 1e-3},
        {"emb_dim": 64, "n_layers": 2, "drop_rate": 0.1, "lr": 1e-3},
        {"emb_dim": 64, "n_layers": 2, "drop_rate": 0.1, "lr": 5e-4},
    ]


def run_sweep_command(args) -> None:
    text = Path(args.text_file).read_text(encoding="utf-8")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sweep_results = []
    for sweep_config in load_sweep_configs(args):
        result = train_once(args, text, overrides=sweep_config)
        sweep_results.append(
            {
                "config": result["config"] | {"lr": sweep_config.get("lr", args.lr)},
                "final_train_loss": result["final_train_loss"],
                "final_val_loss": result["final_val_loss"],
                "perplexity": result["perplexity"],
                "num_params": result["num_params"],
                "elapsed_sec": result["elapsed_sec"],
            }
        )

    sweep_results.sort(key=lambda item: item["final_val_loss"])
    (output_dir / "sweep_results.json").write_text(
        json.dumps(sweep_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    labels = [f"run{i+1}" for i in range(len(sweep_results))]
    val_losses = [item["final_val_loss"] for item in sweep_results]
    plt.figure(figsize=(7, 4))
    plt.bar(labels, val_losses)
    plt.xlabel("Run")
    plt.ylabel("Validation Loss")
    plt.title("Hyperparameter Sweep")
    plt.tight_layout()
    plt.savefig(output_dir / "sweep_val_loss.png")
    plt.close()

    print("best_config:")
    print(json.dumps(sweep_results[0], ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="최소 GPT 실험 스크립트")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_arguments(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--text-file", default="data/nsmc_lm_train.txt")
        subparser.add_argument("--output-dir", default="artifacts")
        subparser.add_argument("--device", default="cpu")
        subparser.add_argument("--vocab-size", type=int, default=300)
        subparser.add_argument("--context-length", type=int, default=32)
        subparser.add_argument("--emb-dim", type=int, default=32)
        subparser.add_argument("--n-heads", type=int, default=4)
        subparser.add_argument("--n-layers", type=int, default=1)
        subparser.add_argument("--drop-rate", type=float, default=0.0)
        subparser.add_argument("--lr", type=float, default=1e-3)
        subparser.add_argument("--batch-size", type=int, default=8)
        subparser.add_argument("--num-epochs", type=int, default=3)
        subparser.add_argument("--train-ratio", type=float, default=0.9)
        subparser.add_argument("--tokenizer-chars", type=int, default=5000)
        subparser.add_argument("--prompt", default="안녕하세요")
        subparser.add_argument("--max-new-tokens", type=int, default=30)
        subparser.add_argument("--qkv-bias", action="store_true")
        subparser.add_argument("--post-norm", action="store_true")

    train_parser = subparsers.add_parser("train")
    add_common_arguments(train_parser)

    sweep_parser = subparsers.add_parser("sweep")
    add_common_arguments(sweep_parser)
    sweep_parser.add_argument("--sweep-file")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "train":
        run_train_command(args)
    elif args.command == "sweep":
        run_sweep_command(args)


if __name__ == "__main__":
    main()
