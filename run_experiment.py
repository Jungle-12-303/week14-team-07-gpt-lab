#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""최소 실행/학습/스윕 스크립트."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch

from src.bpe import BPETokenizer
from src.dataset import create_dataloader
from src.model import GPTModel, generate_text_simple
from src.train import calc_loss_batch, calc_loss_loader, save_checkpoint


PRESETS = {
    "tiny": {
        "vocab_size": 300,
        "context_length": 32,
        "emb_dim": 32,
        "n_heads": 4,
        "n_layers": 1,
        "drop_rate": 0.0,
        "lr": 1e-3,
        "batch_size": 8,
        "num_epochs": 3,
        "tokenizer_chars": 5000,
        "max_new_tokens": 30,
        "stride": 16,
        "train_shards": 1,
        "patience": 0,
        "min_delta": 1e-4,
        "activation_name": "gelu",
    },
    "better_cpu": {
        "vocab_size": 800,
        "context_length": 64,
        "emb_dim": 96,
        "n_heads": 4,
        "n_layers": 3,
        "drop_rate": 0.1,
        "lr": 3e-4,
        "batch_size": 8,
        "num_epochs": 10,
        "tokenizer_chars": 200000,
        "max_new_tokens": 40,
        "stride": 8,
        "train_shards": 4,
        "patience": 3,
        "min_delta": 1e-4,
        "activation_name": "gelu",
    },
    "better_gpu": {
        "vocab_size": 1500,
        "context_length": 128,
        "emb_dim": 192,
        "n_heads": 6,
        "n_layers": 6,
        "drop_rate": 0.1,
        "lr": 3e-4,
        "batch_size": 16,
        "num_epochs": 15,
        "tokenizer_chars": 500000,
        "max_new_tokens": 60,
        "stride": 16,
        "train_shards": 8,
        "patience": 3,
        "min_delta": 1e-4,
        "activation_name": "gelu",
    },
}


def apply_preset(args) -> None:
    preset = PRESETS.get(args.preset)
    if preset is None:
        return
    for key, value in preset.items():
        setattr(args, key, value)


def compute_perplexity(loss_value: float) -> float:
    return math.exp(min(loss_value, 20.0))


def build_tokenizer_and_ids(text: str, vocab_size: int, tokenizer_chars: int):
    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.train(text[:tokenizer_chars])
    token_ids = tokenizer.encode(text)
    return tokenizer, token_ids


def split_train_val_ids(token_ids: list[int], context_length: int, train_ratio: float):
    split_idx = max(int(len(token_ids) * train_ratio), context_length + 2)
    train_ids = token_ids[:split_idx]
    val_ids = token_ids[split_idx:]
    if len(val_ids) < context_length + 2:
        val_ids = train_ids[-(context_length + 2) :]
    return train_ids, val_ids


def make_loader_from_ids(
    token_ids: list[int],
    context_length: int,
    batch_size: int,
    stride: int,
    shuffle: bool,
):
    return create_dataloader(
        token_ids,
        context_length=context_length,
        batch_size=batch_size,
        stride=stride,
        shuffle=shuffle,
    )


def shard_token_ids(token_ids: list[int], shard_count: int, context_length: int) -> list[list[int]]:
    if shard_count <= 1:
        return [token_ids]

    min_shard_len = context_length + 2
    shard_size = max(len(token_ids) // shard_count, min_shard_len)
    shards: list[list[int]] = []
    start = 0
    while start < len(token_ids):
        end = min(start + shard_size, len(token_ids))
        shard = token_ids[start:end]
        if len(shard) >= min_shard_len:
            shards.append(shard)
        start = end
    return shards or [token_ids]


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
        "activation_name": args.activation_name,
    }
    if overrides:
        config.update(overrides)
    return config


def make_epoch_metrics(train_losses: list[float], val_losses: list[float]) -> list[dict]:
    metrics: list[dict] = []
    for epoch_idx, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses), start=1):
        metrics.append(
            {
                "epoch": epoch_idx,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_perplexity": compute_perplexity(train_loss),
                "val_perplexity": compute_perplexity(val_loss),
                "generalization_gap": val_loss - train_loss,
            }
        )
    return metrics


def save_metrics_csv(metrics: list[dict], output_path: Path) -> None:
    if not metrics:
        return
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)


def save_curve_plot(
    values_a: list[float],
    values_b: list[float] | None,
    label_a: str,
    label_b: str | None,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcfg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    plt.plot(values_a, label=label_a)
    if values_b is not None and label_b is not None:
        plt.plot(values_b, label=label_b)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_gap_plot(gaps: list[float], output_path: Path) -> None:
    save_curve_plot(
        values_a=gaps,
        values_b=None,
        label_a="val - train",
        label_b=None,
        ylabel="Gap",
        title="Generalization Gap",
        output_path=output_path,
    )


def make_version_dir(output_root: Path, run_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"{run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def append_run_index(output_root: Path, summary: dict) -> None:
    index_path = output_root / "run_index.jsonl"
    with index_path.open("a", encoding="utf-8") as index_file:
        index_file.write(json.dumps(summary, ensure_ascii=False) + "\n")


def run_training_loop(
    model: GPTModel,
    train_shards,
    val_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    context_length: int,
    batch_size: int,
    stride: int,
    patience: int,
    min_delta: float,
):
    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_model_state = None
    best_optimizer_state = None
    epochs_without_improvement = 0

    model.to(device)
    for epoch_idx in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for shard_ids in train_shards:
            train_loader = make_loader_from_ids(
                shard_ids,
                context_length=context_length,
                batch_size=batch_size,
                stride=stride,
                shuffle=True,
            )
            for input_batch, target_batch in train_loader:
                loss = calc_loss_batch(input_batch, target_batch, model, device)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        avg_train_loss = total_loss / max(num_batches, 1)
        current_val_loss = calc_loss_loader(val_loader, model, device)
        train_losses.append(avg_train_loss)
        val_losses.append(current_val_loss)

        if current_val_loss < (best_val_loss - min_delta):
            best_val_loss = current_val_loss
            best_epoch = epoch_idx
            best_model_state = deepcopy(model.state_dict())
            best_optimizer_state = deepcopy(optimizer.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if patience > 0 and epochs_without_improvement >= patience:
            break

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "best_model_state": best_model_state,
        "best_optimizer_state": best_optimizer_state,
        "stopped_early": len(train_losses) < num_epochs,
    }


def train_once(args, text: str, overrides: dict | None = None, return_state: bool = False):
    config = make_model_config(args, overrides)
    tokenizer, token_ids = build_tokenizer_and_ids(
        text=text,
        vocab_size=config["vocab_size"],
        tokenizer_chars=args.tokenizer_chars,
    )
    train_ids, val_ids = split_train_val_ids(token_ids, config["context_length"], args.train_ratio)
    train_shards = shard_token_ids(train_ids, args.train_shards, config["context_length"])
    val_loader = make_loader_from_ids(
        val_ids,
        context_length=config["context_length"],
        batch_size=args.batch_size,
        stride=args.stride,
        shuffle=False,
    )

    model = GPTModel(config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=overrides.get("lr", args.lr) if overrides else args.lr,
    )
    device = torch.device(args.device)

    start_time = time.time()
    train_output = run_training_loop(
        model=model,
        train_shards=train_shards,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=args.num_epochs,
        context_length=config["context_length"],
        batch_size=args.batch_size,
        stride=args.stride,
        patience=args.patience,
        min_delta=args.min_delta,
    )
    elapsed = time.time() - start_time

    train_losses = train_output["train_losses"]
    val_losses = train_output["val_losses"]
    epoch_metrics = make_epoch_metrics(train_losses, val_losses)

    sample_ids = tokenizer.encode(args.prompt, add_bos_eos=False)
    sample_tensor = torch.tensor([sample_ids], dtype=torch.long, device=device)
    generated = generate_text_simple(
        model=model,
        idx=sample_tensor,
        max_new_tokens=args.max_new_tokens,
        context_size=config["context_length"],
    )
    generated_text = tokenizer.decode(generated[0].tolist(), skip_special=True)

    final_train_loss = train_losses[-1]
    final_val_loss = val_losses[-1]

    result = {
        "config": config,
        "epoch_metrics": epoch_metrics,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "final_train_perplexity": compute_perplexity(final_train_loss),
        "final_val_perplexity": compute_perplexity(final_val_loss),
        "generalization_gap": final_val_loss - final_train_loss,
        "best_val_loss": train_output["best_val_loss"],
        "best_epoch": train_output["best_epoch"],
        "stopped_early": train_output["stopped_early"],
        "epochs_ran": len(train_losses),
        "num_params": count_parameters(model),
        "elapsed_sec": elapsed,
        "generated_text": generated_text,
        "num_train_shards": len(train_shards),
        "stride": args.stride,
    }
    if return_state:
        return result, model, optimizer, tokenizer, train_output
    return result


def run_train_command(args) -> None:
    text = Path(args.text_file).read_text(encoding="utf-8")
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = make_version_dir(output_root, args.run_name)

    result, model, optimizer, tokenizer, train_output = train_once(args, text, return_state=True)

    save_curve_plot(
        result["train_losses"],
        result["val_losses"],
        "train",
        "val",
        "Loss",
        "Training / Validation Loss",
        run_dir / "loss_curve.png",
    )
    save_curve_plot(
        [item["train_perplexity"] for item in result["epoch_metrics"]],
        [item["val_perplexity"] for item in result["epoch_metrics"]],
        "train perplexity",
        "val perplexity",
        "Perplexity",
        "Training / Validation Perplexity",
        run_dir / "perplexity_curve.png",
    )
    save_gap_plot(
        [item["generalization_gap"] for item in result["epoch_metrics"]],
        run_dir / "generalization_gap.png",
    )
    save_metrics_csv(result["epoch_metrics"], run_dir / "epoch_metrics.csv")
    (run_dir / "train_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "config.json").write_text(
        json.dumps(result["config"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=result["epochs_ran"],
        global_step=result["epochs_ran"],
        path=str(run_dir / "final_checkpoint.pt"),
    )
    final_checkpoint = torch.load(run_dir / "final_checkpoint.pt", map_location="cpu")
    final_checkpoint["config"] = result["config"]
    final_checkpoint["tokenizer_chars"] = args.tokenizer_chars
    final_checkpoint["vocab_size"] = args.vocab_size
    torch.save(final_checkpoint, run_dir / "final_checkpoint.pt")

    if train_output["best_model_state"] is not None:
        model.load_state_dict(train_output["best_model_state"])
        optimizer.load_state_dict(train_output["best_optimizer_state"])
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=result["best_epoch"],
            global_step=result["best_epoch"],
            path=str(run_dir / "best_checkpoint.pt"),
        )
        best_checkpoint = torch.load(run_dir / "best_checkpoint.pt", map_location="cpu")
        best_checkpoint["config"] = result["config"]
        best_checkpoint["tokenizer_chars"] = args.tokenizer_chars
        best_checkpoint["vocab_size"] = args.vocab_size
        torch.save(best_checkpoint, run_dir / "best_checkpoint.pt")

    summary = {
        "run_name": run_dir.name,
        "final_train_loss": result["final_train_loss"],
        "final_val_loss": result["final_val_loss"],
        "final_train_perplexity": result["final_train_perplexity"],
        "final_val_perplexity": result["final_val_perplexity"],
        "generalization_gap": result["generalization_gap"],
        "best_epoch": result["best_epoch"],
        "best_val_loss": result["best_val_loss"],
        "stopped_early": result["stopped_early"],
        "epochs_ran": result["epochs_ran"],
        "num_params": result["num_params"],
        "elapsed_sec": result["elapsed_sec"],
        "config": result["config"],
    }
    append_run_index(output_root, summary)

    print("final_train_loss:", result["final_train_loss"])
    print("final_val_loss:", result["final_val_loss"])
    print("final_train_perplexity:", result["final_train_perplexity"])
    print("final_val_perplexity:", result["final_val_perplexity"])
    print("generalization_gap:", result["generalization_gap"])
    print("best_epoch:", result["best_epoch"])
    print("best_val_loss:", result["best_val_loss"])
    print("stopped_early:", result["stopped_early"])
    print("run_dir:", run_dir)
    print("best_checkpoint:", run_dir / "best_checkpoint.pt")
    print("final_checkpoint:", run_dir / "final_checkpoint.pt")
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
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    sweep_results = []
    for sweep_config in load_sweep_configs(args):
        result = train_once(args, text, overrides=sweep_config)
        sweep_results.append(
            {
                "config": result["config"] | {"lr": sweep_config.get("lr", args.lr)},
                "final_train_loss": result["final_train_loss"],
                "final_val_loss": result["final_val_loss"],
                "final_val_perplexity": result["final_val_perplexity"],
                "generalization_gap": result["generalization_gap"],
                "best_epoch": result["best_epoch"],
                "stopped_early": result["stopped_early"],
                "num_params": result["num_params"],
                "elapsed_sec": result["elapsed_sec"],
            }
        )

    sweep_results.sort(key=lambda item: item["final_val_loss"])
    (output_root / "sweep_results.json").write_text(
        json.dumps(sweep_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    save_curve_plot(
        [item["final_val_loss"] for item in sweep_results],
        None,
        "validation loss",
        None,
        "Validation Loss",
        "Hyperparameter Sweep",
        output_root / "sweep_val_loss.png",
    )

    print("best_config:")
    print(json.dumps(sweep_results[0], ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="최소 GPT 실험 스크립트")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_arguments(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--text-file", default="data/nsmc_lm_train.txt")
        subparser.add_argument("--output-dir", default="artifacts")
        subparser.add_argument("--run-name", default="gpt_run")
        subparser.add_argument("--device", default="cpu")
        subparser.add_argument("--preset", choices=sorted(PRESETS.keys()), default="tiny")
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
        subparser.add_argument("--stride", type=int, default=16)
        subparser.add_argument("--train-shards", type=int, default=1)
        subparser.add_argument("--patience", type=int, default=0)
        subparser.add_argument("--min-delta", type=float, default=1e-4)
        subparser.add_argument("--prompt", default="안녕하세요")
        subparser.add_argument("--max-new-tokens", type=int, default=30)
        subparser.add_argument("--qkv-bias", action="store_true")
        subparser.add_argument("--post-norm", action="store_true")
        subparser.add_argument("--activation-name", choices=["gelu", "gelu_exact", "quick_gelu"], default="gelu")

    train_parser = subparsers.add_parser("train")
    add_common_arguments(train_parser)

    sweep_parser = subparsers.add_parser("sweep")
    add_common_arguments(sweep_parser)
    sweep_parser.add_argument("--sweep-file")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    apply_preset(args)
    if args.command == "train":
        run_train_command(args)
    elif args.command == "sweep":
        run_sweep_command(args)


if __name__ == "__main__":
    main()
