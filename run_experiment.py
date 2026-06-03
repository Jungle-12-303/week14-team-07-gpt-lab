#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""최소 실행/학습/스윕 스크립트."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import torch

from src.bpe import BPETokenizer
from src.dataset import create_dataloader
from src.historical_replay import save_dense_replay_artifacts
from src.model import GPTModel, generate_text_simple
from src.train import (
    calc_loss_batch,
    calc_loss_loader,
    compute_grad_norm,
    compute_historical_fit_metrics,
    save_checkpoint,
    run_pytest_before_training,
)


PRESETS = {
    "tiny": {
        "vocab_size": 300,
        "min_frequency": 1,
        "context_length": 32,
        "emb_dim": 32,
        "n_heads": 4,
        "n_layers": 1,
        "drop_rate": 0.0,
        "lr": 1e-3,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "batch_size": 8,
        "num_epochs": 3,
        "eval_batches": 4,
        "eval_every_steps": 0,
        "tokenizer_chars": 5000,
        "max_new_tokens": 30,
        "stride": 16,
        "train_shards": 1,
        "patience": 0,
        "min_delta": 1e-4,
        "activation_name": "gelu",
        "ffn_mult": 4,
        "ffn_dropout_position": "after_output",
        "attention_impl": "manual",
        "tie_embeddings": False,
        "init_std": 0.02,
        "norm_first": True,
        "norm_eps": 1e-5,
    },
    "better_cpu": {
        "vocab_size": 800,
        "min_frequency": 1,
        "context_length": 64,
        "emb_dim": 96,
        "n_heads": 4,
        "n_layers": 3,
        "drop_rate": 0.1,
        "lr": 3e-4,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "batch_size": 8,
        "num_epochs": 10,
        "eval_batches": 4,
        "eval_every_steps": 0,
        "tokenizer_chars": 200000,
        "max_new_tokens": 40,
        "stride": 8,
        "train_shards": 4,
        "patience": 3,
        "min_delta": 1e-4,
        "activation_name": "gelu",
        "ffn_mult": 4,
        "ffn_dropout_position": "after_output",
        "attention_impl": "manual",
        "tie_embeddings": False,
        "init_std": 0.02,
        "norm_first": True,
        "norm_eps": 1e-5,
    },
    "better_gpu": {
        "vocab_size": 1500,
        "min_frequency": 1,
        "context_length": 128,
        "emb_dim": 192,
        "n_heads": 6,
        "n_layers": 6,
        "drop_rate": 0.1,
        "lr": 3e-4,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "batch_size": 16,
        "num_epochs": 15,
        "eval_batches": 4,
        "eval_every_steps": 0,
        "tokenizer_chars": 500000,
        "max_new_tokens": 60,
        "stride": 16,
        "train_shards": 8,
        "patience": 3,
        "min_delta": 1e-4,
        "activation_name": "gelu",
        "ffn_mult": 4,
        "ffn_dropout_position": "after_output",
        "attention_impl": "manual",
        "tie_embeddings": False,
        "init_std": 0.02,
        "norm_first": True,
        "norm_eps": 1e-5,
    },
}

COMMON_DEFAULTS = {
    "text_file": "data/nsmc_lm_train.txt",
    "output_dir": "train/train2/runs",
    "run_name": "train2_run",
    "device": "cpu",
    "vocab_size": 300,
    "min_frequency": 1,
    "context_length": 32,
    "emb_dim": 32,
    "n_heads": 4,
    "n_layers": 1,
    "drop_rate": 0.0,
    "lr": 1e-3,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "batch_size": 8,
    "num_epochs": 3,
    "eval_batches": 4,
    "eval_every_steps": 0,
    "train_ratio": 0.9,
    "tokenizer_chars": 5000,
    "stride": 16,
    "train_shards": 1,
    "patience": 0,
    "min_delta": 1e-4,
    "prompt": "안녕하세요",
    "max_new_tokens": 30,
    "qkv_bias": False,
    "post_norm": False,
    "norm_first": None,
    "activation_name": "gelu",
    "ffn_mult": 4,
    "ffn_dropout_position": "after_output",
    "attention_impl": "manual",
    "tie_embeddings": False,
    "init_std": 0.02,
    "norm_eps": 1e-5,
    "seed": 123,
}


def apply_preset(args) -> None:
    preset = PRESETS.get(args.preset)
    if preset is not None:
        for key, value in preset.items():
            if getattr(args, key, None) is None:
                setattr(args, key, value)

    for key, value in COMMON_DEFAULTS.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)

    if args.post_norm:
        args.norm_first = False
    elif args.norm_first is None:
        args.norm_first = True


def compute_perplexity(loss_value: float) -> float:
    return math.exp(min(loss_value, 20.0))


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_tokenizer_and_ids(text: str, vocab_size: int, tokenizer_chars: int, min_frequency: int):
    tokenizer = BPETokenizer(vocab_size=vocab_size, min_frequency=min_frequency)
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
    effective_stride = stride if stride is not None else context_length
    available = len(token_ids) - context_length - 1
    has_samples = available >= 0 and (available // effective_stride + 1) > 0
    return create_dataloader(
        token_ids,
        context_length=context_length,
        batch_size=batch_size,
        stride=stride,
        shuffle=shuffle if has_samples else False,
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


def make_model_config(args, actual_vocab_size: int | None = None, overrides: dict | None = None) -> dict:
    overrides = overrides or {}

    def resolve(name: str):
        return overrides.get(name, getattr(args, name))

    norm_first = overrides.get("norm_first", args.norm_first)
    config = {
        "vocab_size": actual_vocab_size if actual_vocab_size is not None else resolve("vocab_size"),
        "context_length": resolve("context_length"),
        "emb_dim": resolve("emb_dim"),
        "n_heads": resolve("n_heads"),
        "n_layers": resolve("n_layers"),
        "drop_rate": resolve("drop_rate"),
        "qkv_bias": resolve("qkv_bias"),
        "pre_norm": bool(norm_first),
        "norm_first": bool(norm_first),
        "norm_eps": resolve("norm_eps"),
        "activation_name": resolve("activation_name"),
        "ffn_mult": resolve("ffn_mult"),
        "ffn_dropout_position": resolve("ffn_dropout_position"),
        "attention_impl": resolve("attention_impl"),
        "tie_embeddings": resolve("tie_embeddings"),
        "init_std": resolve("init_std"),
    }
    return config


def make_run_plan(
    args,
    config: dict,
    actual_vocab_size: int,
    train_token_count: int,
    val_token_count: int,
    overrides: dict | None = None,
) -> dict:
    overrides = overrides or {}

    def resolve(name: str):
        return overrides.get(name, getattr(args, name))

    return {
        "seed": resolve("seed"),
        "vocab_size": args.vocab_size,
        "actual_vocab_size": actual_vocab_size,
        "min_frequency": resolve("min_frequency"),
        "context_length": config["context_length"],
        "stride": resolve("stride"),
        "batch_size": resolve("batch_size"),
        "num_epochs": resolve("num_epochs"),
        "eval_batches": resolve("eval_batches"),
        "eval_every_steps": resolve("eval_every_steps"),
        "train_ratio": resolve("train_ratio"),
        "learning_rate": resolve("lr"),
        "weight_decay": resolve("weight_decay"),
        "grad_clip": resolve("grad_clip"),
        "emb_dim": config["emb_dim"],
        "dff": config["emb_dim"] * config["ffn_mult"],
        "n_heads": config["n_heads"],
        "n_layers": config["n_layers"],
        "drop_rate": config["drop_rate"],
        "qkv_bias": config["qkv_bias"],
        "ffn_mult": config["ffn_mult"],
        "norm_first": config["norm_first"],
        "norm_eps": config["norm_eps"],
        "activation_name": config["activation_name"],
        "ffn_dropout_position": config["ffn_dropout_position"],
        "attention_impl": config["attention_impl"],
        "tie_embeddings": config["tie_embeddings"],
        "init_std": config["init_std"],
        "tokenizer_chars": resolve("tokenizer_chars"),
        "train_shards": resolve("train_shards"),
        "prompt": resolve("prompt"),
        "max_new_tokens": resolve("max_new_tokens"),
        "train_token_count": train_token_count,
        "val_token_count": val_token_count,
    }


def make_log_context(plan: dict, parameter_count: int) -> dict:
    return {
        "learning_rate": float(plan["learning_rate"]),
        "batch_size": int(plan["batch_size"]),
        "emb_dim": int(plan["emb_dim"]),
        "dff": int(plan["dff"]),
        "n_heads": int(plan["n_heads"]),
        "n_layers": int(plan["n_layers"]),
        "ffn_mult": int(plan["ffn_mult"]),
        "activation_name": plan["activation_name"],
        "ffn_dropout_position": plan["ffn_dropout_position"],
        "attention_impl": plan["attention_impl"],
        "tie_embeddings": bool(plan["tie_embeddings"]),
        "init_std": float(plan["init_std"]),
        "norm_first": bool(plan["norm_first"]),
        "parameter_count": int(parameter_count),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as jsonl_file:
        for row in rows:
            jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def build_experiment_args(
    *,
    preset: str | None = None,
    output_dir: str | Path | None = None,
    run_name: str | None = None,
    text_file: str | Path | None = None,
    device: str | None = None,
    **overrides,
):
    values = deepcopy(COMMON_DEFAULTS)
    if preset is not None:
        values.update(deepcopy(PRESETS.get(preset, {})))
    values["preset"] = preset

    if output_dir is not None:
        values["output_dir"] = str(output_dir)
    if run_name is not None:
        values["run_name"] = run_name
    if text_file is not None:
        values["text_file"] = str(text_file)
    if device is not None:
        values["device"] = device

    if "activation" in overrides and "activation_name" not in overrides:
        overrides["activation_name"] = overrides.pop("activation")
    if "eval_iter" in overrides and "eval_batches" not in overrides:
        overrides["eval_batches"] = overrides.pop("eval_iter")
    if "post_norm" in overrides and overrides["post_norm"]:
        overrides["norm_first"] = False

    for key, value in overrides.items():
        if value is not None:
            values[key] = value

    if values.get("norm_first") is None:
        values["norm_first"] = not bool(values.get("post_norm", False))
    elif values.get("post_norm"):
        values["norm_first"] = False

    return SimpleNamespace(**values)


def run_training_loop(
    model: GPTModel,
    train_shards,
    train_eval_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    plan: dict,
    num_epochs: int,
    eval_batches: int,
    eval_every_steps: int,
    grad_clip: float,
    patience: int,
    min_delta: float,
    parameter_count: int,
    total_planned_steps: int,
):
    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_step = 0
    best_model_state = None
    best_optimizer_state = None
    epochs_without_improvement = 0
    log_context = make_log_context(plan, parameter_count)

    initial_train_loss = calc_loss_loader(train_eval_loader, model, device, num_batches=eval_batches)
    initial_val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_batches)
    initial_fit = compute_historical_fit_metrics(
        initial_train_loss=initial_train_loss,
        initial_val_loss=initial_val_loss,
        final_train_loss=initial_train_loss,
        final_val_loss=initial_val_loss,
        max_steps=max(total_planned_steps, 1),
    )

    step_history: list[dict] = []
    eval_history: list[dict] = [
        {
            **log_context,
            "step": 0,
            "epoch": 0,
            "train_eval_loss": initial_train_loss,
            "val_loss": initial_val_loss,
            "train_perplexity": compute_perplexity(initial_train_loss),
            "val_perplexity": compute_perplexity(initial_val_loss),
            "generalization_gap": initial_fit["final_generalization_gap"],
            "generalization_gap_delta": initial_fit["generalization_gap_delta"],
            "train_val_improvement_gap": initial_fit["train_val_improvement_gap"],
            "overfit_score": initial_fit["overfit_score"],
            "fit_status": initial_fit["fit_status"],
            "elapsed_sec": 0.0,
            "tokens_seen": 0,
        }
    ]
    epoch_history: list[dict] = []
    global_step = 0
    tokens_seen = 0
    run_start = time.perf_counter()

    model.to(device)
    for epoch_idx in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0
        epoch_losses: list[float] = []

        for shard_ids in train_shards:
            train_loader = shard_ids
            for input_batch, target_batch in train_loader:
                loss = calc_loss_batch(input_batch, target_batch, model, device)
                optimizer.zero_grad()
                loss.backward()
                grad_norm = compute_grad_norm(model.parameters())
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

                loss_value = loss.item()
                total_loss += loss_value
                num_batches += 1
                global_step += 1
                epoch_losses.append(loss_value)
                tokens_seen += int(input_batch.numel())

                step_history.append(
                    {
                        **log_context,
                        "step": global_step,
                        "epoch": epoch_idx,
                        "train_step_loss": loss_value,
                        "grad_norm": grad_norm,
                        "lr": float(optimizer.param_groups[0]["lr"]),
                        "tokens_seen": tokens_seen,
                        "elapsed_sec": time.perf_counter() - run_start,
                    }
                )

                if eval_every_steps > 0 and global_step % eval_every_steps == 0:
                    train_eval_loss = calc_loss_loader(
                        train_eval_loader,
                        model,
                        device,
                        num_batches=eval_batches,
                    )
                    current_val_loss = calc_loss_loader(
                        val_loader,
                        model,
                        device,
                        num_batches=eval_batches,
                    )
                    fit_metrics = compute_historical_fit_metrics(
                        initial_train_loss=initial_train_loss,
                        initial_val_loss=initial_val_loss,
                        final_train_loss=train_eval_loss,
                        final_val_loss=current_val_loss,
                        max_steps=max(total_planned_steps, 1),
                    )
                    eval_history.append(
                        {
                            **log_context,
                            "step": global_step,
                            "epoch": epoch_idx,
                            "train_eval_loss": train_eval_loss,
                            "val_loss": current_val_loss,
                            "train_perplexity": compute_perplexity(train_eval_loss),
                            "val_perplexity": compute_perplexity(current_val_loss),
                            "generalization_gap": fit_metrics["final_generalization_gap"],
                            "generalization_gap_delta": fit_metrics["generalization_gap_delta"],
                            "train_val_improvement_gap": fit_metrics["train_val_improvement_gap"],
                            "overfit_score": fit_metrics["overfit_score"],
                            "fit_status": fit_metrics["fit_status"],
                            "elapsed_sec": time.perf_counter() - run_start,
                            "tokens_seen": tokens_seen,
                        }
                    )

        avg_train_loss = total_loss / max(num_batches, 1)
        if not epoch_losses:
            break

        if eval_history[-1]["step"] != global_step:
            train_eval_loss = calc_loss_loader(
                train_eval_loader,
                model,
                device,
                num_batches=eval_batches,
            )
            current_val_loss = calc_loss_loader(
                val_loader,
                model,
                device,
                num_batches=eval_batches,
            )
            fit_metrics = compute_historical_fit_metrics(
                initial_train_loss=initial_train_loss,
                initial_val_loss=initial_val_loss,
                final_train_loss=train_eval_loss,
                final_val_loss=current_val_loss,
                max_steps=max(total_planned_steps, 1),
            )
            eval_history.append(
                {
                    **log_context,
                    "step": global_step,
                    "epoch": epoch_idx,
                    "train_eval_loss": train_eval_loss,
                    "val_loss": current_val_loss,
                    "train_perplexity": compute_perplexity(train_eval_loss),
                    "val_perplexity": compute_perplexity(current_val_loss),
                    "generalization_gap": fit_metrics["final_generalization_gap"],
                    "generalization_gap_delta": fit_metrics["generalization_gap_delta"],
                    "train_val_improvement_gap": fit_metrics["train_val_improvement_gap"],
                    "overfit_score": fit_metrics["overfit_score"],
                    "fit_status": fit_metrics["fit_status"],
                    "elapsed_sec": time.perf_counter() - run_start,
                    "tokens_seen": tokens_seen,
                }
            )

        last_eval = eval_history[-1]
        train_losses.append(avg_train_loss)
        val_losses.append(float(last_eval["val_loss"]))
        epoch_history.append(
            {
                **log_context,
                "epoch": epoch_idx,
                "steps_completed": global_step,
                "avg_train_step_loss": avg_train_loss,
                "train_eval_loss": float(last_eval["train_eval_loss"]),
                "val_loss": float(last_eval["val_loss"]),
                "train_perplexity": float(last_eval["train_perplexity"]),
                "val_perplexity": float(last_eval["val_perplexity"]),
                "generalization_gap": float(last_eval["generalization_gap"]),
                "overfit_score": float(last_eval["overfit_score"]),
                "fit_status": last_eval["fit_status"],
                "elapsed_sec": float(last_eval["elapsed_sec"]),
                "tokens_seen": int(last_eval["tokens_seen"]),
            }
        )

        current_val_loss = float(last_eval["val_loss"])

        if current_val_loss < (best_val_loss - min_delta):
            best_val_loss = current_val_loss
            best_epoch = epoch_idx
            best_step = global_step
            best_model_state = deepcopy(model.state_dict())
            best_optimizer_state = deepcopy(optimizer.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if patience > 0 and epochs_without_improvement >= patience:
            break

    return {
        "initial_train_loss": initial_train_loss,
        "initial_val_loss": initial_val_loss,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "step_history": step_history,
        "eval_history": eval_history,
        "epoch_history": epoch_history,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "best_step": best_step,
        "best_model_state": best_model_state,
        "best_optimizer_state": best_optimizer_state,
        "stopped_early": len(train_losses) < num_epochs,
        "global_step": global_step,
        "tokens_seen": tokens_seen,
        "elapsed_sec": time.perf_counter() - run_start,
    }


def train_once(args, text: str, overrides: dict | None = None, return_state: bool = False):
    set_global_seed(args.seed)
    tokenizer, token_ids = build_tokenizer_and_ids(
        text=text,
        vocab_size=overrides.get("vocab_size", args.vocab_size) if overrides else args.vocab_size,
        tokenizer_chars=args.tokenizer_chars,
        min_frequency=overrides.get("min_frequency", args.min_frequency) if overrides else args.min_frequency,
    )
    actual_vocab_size = len(tokenizer.id_to_token)
    config = make_model_config(args, actual_vocab_size=actual_vocab_size, overrides=overrides)
    train_ratio = overrides.get("train_ratio", args.train_ratio) if overrides else args.train_ratio
    train_ids, val_ids = split_train_val_ids(token_ids, config["context_length"], train_ratio)
    train_shard_ids = shard_token_ids(train_ids, args.train_shards, config["context_length"])
    batch_size = overrides.get("batch_size", args.batch_size) if overrides else args.batch_size
    stride = overrides.get("stride", args.stride) if overrides else args.stride
    eval_batches = overrides.get("eval_batches", args.eval_batches) if overrides else args.eval_batches
    eval_every_steps = overrides.get("eval_every_steps", args.eval_every_steps) if overrides else args.eval_every_steps
    weight_decay = overrides.get("weight_decay", args.weight_decay) if overrides else args.weight_decay
    grad_clip = overrides.get("grad_clip", args.grad_clip) if overrides else args.grad_clip
    num_epochs = overrides.get("num_epochs", args.num_epochs) if overrides else args.num_epochs

    train_eval_loader = make_loader_from_ids(
        train_ids,
        context_length=config["context_length"],
        batch_size=batch_size,
        stride=stride,
        shuffle=False,
    )
    val_loader = make_loader_from_ids(
        val_ids,
        context_length=config["context_length"],
        batch_size=batch_size,
        stride=stride,
        shuffle=False,
    )
    train_shards = []
    for shard_ids in train_shard_ids:
        shard_loader = make_loader_from_ids(
            shard_ids,
            context_length=config["context_length"],
            batch_size=batch_size,
            stride=stride,
            shuffle=True,
        )
        if len(shard_loader) > 0:
            train_shards.append(shard_loader)
    if len(train_eval_loader) == 0 or len(val_loader) == 0 or not train_shards:
        raise ValueError("dataset is too small for the requested context_length/batch_size/stride setup")
    total_planned_steps = max(sum(len(loader) for loader in train_shards) * num_epochs, 1)

    model = GPTModel(config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=overrides.get("lr", args.lr) if overrides else args.lr,
        weight_decay=weight_decay,
    )
    device = torch.device(args.device)
    parameter_count = count_parameters(model)
    plan = make_run_plan(
        args=args,
        config=config,
        actual_vocab_size=actual_vocab_size,
        train_token_count=len(train_ids),
        val_token_count=len(val_ids),
        overrides=overrides,
    )

    start_time = time.time()
    train_output = run_training_loop(
        model=model,
        train_shards=train_shards,
        train_eval_loader=train_eval_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        plan=plan,
        num_epochs=num_epochs,
        eval_batches=eval_batches,
        eval_every_steps=eval_every_steps,
        grad_clip=grad_clip,
        patience=args.patience,
        min_delta=args.min_delta,
        parameter_count=parameter_count,
        total_planned_steps=total_planned_steps,
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
    fit_metrics = compute_historical_fit_metrics(
        initial_train_loss=train_output["initial_train_loss"],
        initial_val_loss=train_output["initial_val_loss"],
        final_train_loss=final_train_loss,
        final_val_loss=final_val_loss,
        max_steps=total_planned_steps,
    )

    result = {
        "plan": plan,
        "config": config,
        "epoch_metrics": epoch_metrics,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "step_history": train_output["step_history"],
        "eval_history": train_output["eval_history"],
        "epoch_history": train_output["epoch_history"],
        "initial_train_loss": train_output["initial_train_loss"],
        "initial_val_loss": train_output["initial_val_loss"],
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "final_train_perplexity": compute_perplexity(final_train_loss),
        "final_val_perplexity": compute_perplexity(final_val_loss),
        "generalization_gap": fit_metrics["final_generalization_gap"],
        "initial_generalization_gap": fit_metrics["initial_generalization_gap"],
        "generalization_gap_delta": fit_metrics["generalization_gap_delta"],
        "train_loss_delta": fit_metrics["train_loss_delta"],
        "val_loss_delta": fit_metrics["val_loss_delta"],
        "train_val_improvement_gap": fit_metrics["train_val_improvement_gap"],
        "overfit_score": fit_metrics["overfit_score"],
        "fit_status": fit_metrics["fit_status"],
        "best_val_loss": train_output["best_val_loss"],
        "best_epoch": train_output["best_epoch"],
        "best_step": train_output["best_step"],
        "stopped_early": train_output["stopped_early"],
        "epochs_ran": len(train_losses),
        "num_params": parameter_count,
        "elapsed_sec": elapsed,
        "generated_text": generated_text,
        "num_train_shards": len(train_shards),
        "stride": stride,
        "actual_vocab_size": actual_vocab_size,
        "train_token_count": len(train_ids),
        "val_token_count": len(val_ids),
        "tokens_seen": train_output["tokens_seen"],
        "tokens_per_sec": train_output["tokens_seen"] / elapsed if elapsed > 0 else 0.0,
        "device": str(device),
    }
    if return_state:
        return result, model, optimizer, tokenizer, train_output
    return result


def persist_train_artifacts(
    *,
    args,
    output_root: Path,
    run_dir: Path,
    result: dict,
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    tokenizer,
    train_output: dict,
) -> dict:
    results_row = result["plan"] | {
        "initial_train_loss": result["initial_train_loss"],
        "initial_val_loss": result["initial_val_loss"],
        "final_train_loss": result["final_train_loss"],
        "final_val_loss": result["final_val_loss"],
        "final_train_perplexity": result["final_train_perplexity"],
        "final_val_perplexity": result["final_val_perplexity"],
        "initial_generalization_gap": result["initial_generalization_gap"],
        "final_generalization_gap": result["generalization_gap"],
        "generalization_gap_delta": result["generalization_gap_delta"],
        "train_loss_delta": result["train_loss_delta"],
        "val_loss_delta": result["val_loss_delta"],
        "train_val_improvement_gap": result["train_val_improvement_gap"],
        "overfit_score": result["overfit_score"],
        "fit_status": result["fit_status"],
        "best_val_loss": result["best_val_loss"],
        "best_epoch": result["best_epoch"],
        "best_step": result["best_step"],
        "stopped_early": result["stopped_early"],
        "epochs_ran": result["epochs_ran"],
        "elapsed_sec": result["elapsed_sec"],
        "tokens_seen": result["tokens_seen"],
        "tokens_per_sec": result["tokens_per_sec"],
        "parameter_count": result["num_params"],
        "device": result["device"],
    }
    write_jsonl(run_dir / "results.jsonl", [results_row])
    write_csv(run_dir / "results.csv", [results_row])
    write_csv(run_dir / "plan.csv", [result["plan"]])
    (run_dir / "plan.json").write_text(
        json.dumps([result["plan"]], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tokenizer.save(run_dir / "tokenizer.json")
    save_dense_replay_artifacts(
        artifact_dir=run_dir,
        step_history=result["step_history"],
        eval_history=result["eval_history"],
        epoch_history=result["epoch_history"],
    )

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
    result_for_json = {
        key: value
        for key, value in result.items()
        if key not in {"step_history", "eval_history", "epoch_history"}
    }
    (run_dir / "train_result.json").write_text(
        json.dumps(result_for_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "config.json").write_text(
        json.dumps(result["config"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    final_checkpoint_path = run_dir / "final_checkpoint.pt"
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=result["epochs_ran"],
        global_step=result["epochs_ran"],
        path=str(final_checkpoint_path),
    )
    final_checkpoint = torch.load(final_checkpoint_path, map_location="cpu")
    final_checkpoint["config"] = result["config"]
    final_checkpoint["tokenizer_chars"] = args.tokenizer_chars
    final_checkpoint["vocab_size"] = args.vocab_size
    torch.save(final_checkpoint, final_checkpoint_path)

    best_checkpoint_path = run_dir / "best_checkpoint.pt"
    if train_output["best_model_state"] is not None:
        model.load_state_dict(train_output["best_model_state"])
        optimizer.load_state_dict(train_output["best_optimizer_state"])
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=result["best_epoch"],
            global_step=result["best_epoch"],
            path=str(best_checkpoint_path),
        )
        best_checkpoint = torch.load(best_checkpoint_path, map_location="cpu")
        best_checkpoint["config"] = result["config"]
        best_checkpoint["tokenizer_chars"] = args.tokenizer_chars
        best_checkpoint["vocab_size"] = args.vocab_size
        torch.save(best_checkpoint, best_checkpoint_path)

    summary = {
        "run_name": run_dir.name,
        "final_train_loss": result["final_train_loss"],
        "final_val_loss": result["final_val_loss"],
        "final_train_perplexity": result["final_train_perplexity"],
        "final_val_perplexity": result["final_val_perplexity"],
        "generalization_gap": result["generalization_gap"],
        "overfit_score": result["overfit_score"],
        "fit_status": result["fit_status"],
        "best_epoch": result["best_epoch"],
        "best_step": result["best_step"],
        "best_val_loss": result["best_val_loss"],
        "stopped_early": result["stopped_early"],
        "epochs_ran": result["epochs_ran"],
        "num_params": result["num_params"],
        "elapsed_sec": result["elapsed_sec"],
        "config": result["config"],
    }
    append_run_index(output_root, summary)
    return {
        "summary": summary,
        "best_checkpoint_path": best_checkpoint_path if best_checkpoint_path.exists() else None,
        "final_checkpoint_path": final_checkpoint_path,
    }


def run_train_experiment(
    *,
    corpus: str | None = None,
    text_file: str | Path | None = None,
    output_dir: str | Path | None = None,
    run_name: str = "notebook_run",
    preset: str | None = None,
    device: str = "cpu",
    corpus_len: int | None = None,
    checkpoint_path: str | Path | None = None,
    align_tokenizer_to_corpus: bool = True,
    run_tests: bool = False,
    test_paths: list[str] | tuple[str, ...] | None = None,
    repo_root: str | Path | None = None,
    **overrides,
):
    if run_tests:
        run_pytest_before_training(repo_root=repo_root, test_paths=test_paths)

    if corpus is None:
        text_path = Path(text_file or COMMON_DEFAULTS["text_file"])
        corpus = text_path.read_text(encoding="utf-8")
    if corpus_len is not None:
        corpus = corpus[:corpus_len]

    # In notebook-style runs that pass an in-memory corpus directly, keep the
    # tokenizer training window aligned with the actual corpus slice used for
    # model training so the baseline is not accidentally built on mismatched
    # text ranges.
    if align_tokenizer_to_corpus and text_file is None:
        effective_corpus_chars = len(corpus)
        requested_tokenizer_chars = overrides.get("tokenizer_chars")
        if (
            requested_tokenizer_chars is not None
            and int(requested_tokenizer_chars) != effective_corpus_chars
        ):
            raise ValueError(
                "When passing corpus directly, tokenizer_chars must match the "
                "actual corpus slice used for training. "
                f"Got tokenizer_chars={requested_tokenizer_chars}, "
                f"effective_corpus_chars={effective_corpus_chars}."
            )
        overrides["tokenizer_chars"] = effective_corpus_chars

    output_root = Path(output_dir or COMMON_DEFAULTS["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)

    args = build_experiment_args(
        preset=preset,
        output_dir=output_root,
        run_name=run_name,
        text_file=text_file,
        device=device,
        **overrides,
    )
    run_dir = make_version_dir(output_root, args.run_name)

    result, model, optimizer, tokenizer, train_output = train_once(args, corpus, return_state=True)
    persisted = persist_train_artifacts(
        args=args,
        output_root=output_root,
        run_dir=run_dir,
        result=result,
        model=model,
        optimizer=optimizer,
        tokenizer=tokenizer,
        train_output=train_output,
    )

    external_checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    if external_checkpoint is not None and persisted["best_checkpoint_path"] is not None:
        external_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(persisted["best_checkpoint_path"], external_checkpoint)

    history = [
        {
            "epoch": item["epoch"],
            "train_loss": item["train_loss"],
            "val_loss": item["val_loss"],
            "gap": item["generalization_gap"],
        }
        for item in result["epoch_metrics"]
    ]

    stopped_reason = "early stopping: no validation improvement" if result["stopped_early"] else "max_epochs completed"
    return {
        "model": model,
        "tokenizer": tokenizer,
        "config": result["config"],
        "optimizer": optimizer,
        "history": history,
        "epoch_metrics": result["epoch_metrics"],
        "step_history": result["step_history"],
        "eval_history": result["eval_history"],
        "epoch_history": result["epoch_history"],
        "plan": result["plan"],
        "train_result": result,
        "run_dir": run_dir,
        "best_checkpoint_path": external_checkpoint if external_checkpoint is not None else persisted["best_checkpoint_path"],
        "final_checkpoint_path": persisted["final_checkpoint_path"],
        "best_val_loss": result["best_val_loss"],
        "best_epoch": result["best_epoch"],
        "stopped_reason": stopped_reason,
    }


def run_train_command(args) -> None:
    experiment = run_train_experiment(
        text_file=args.text_file,
        output_dir=args.output_dir,
        run_name=args.run_name,
        preset=args.preset,
        device=args.device,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        context_length=args.context_length,
        emb_dim=args.emb_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        drop_rate=args.drop_rate,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        eval_batches=args.eval_batches,
        eval_every_steps=args.eval_every_steps,
        train_ratio=args.train_ratio,
        tokenizer_chars=args.tokenizer_chars,
        stride=args.stride,
        train_shards=args.train_shards,
        patience=args.patience,
        min_delta=args.min_delta,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        qkv_bias=args.qkv_bias,
        post_norm=args.post_norm,
        norm_first=args.norm_first,
        activation_name=args.activation_name,
        ffn_mult=args.ffn_mult,
        ffn_dropout_position=args.ffn_dropout_position,
        attention_impl=args.attention_impl,
        tie_embeddings=args.tie_embeddings,
        init_std=args.init_std,
        norm_eps=args.norm_eps,
        seed=args.seed,
    )

    result = experiment["train_result"]
    run_dir = experiment["run_dir"]
    print("final_train_loss:", result["final_train_loss"])
    print("final_val_loss:", result["final_val_loss"])
    print("final_train_perplexity:", result["final_train_perplexity"])
    print("final_val_perplexity:", result["final_val_perplexity"])
    print("generalization_gap:", result["generalization_gap"])
    print("overfit_score:", result["overfit_score"])
    print("fit_status:", result["fit_status"])
    print("best_epoch:", result["best_epoch"])
    print("best_step:", result["best_step"])
    print("best_val_loss:", result["best_val_loss"])
    print("stopped_early:", result["stopped_early"])
    print("run_dir:", run_dir)
    print("best_checkpoint:", experiment["best_checkpoint_path"])
    print("final_checkpoint:", experiment["final_checkpoint_path"])
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
                "overfit_score": result["overfit_score"],
                "fit_status": result["fit_status"],
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
        subparser.add_argument("--text-file", default=None)
        subparser.add_argument("--output-dir", default=None)
        subparser.add_argument("--run-name", default=None)
        subparser.add_argument("--device", default=None)
        subparser.add_argument("--preset", choices=sorted(PRESETS.keys()), default="tiny")
        subparser.add_argument("--vocab-size", type=int, default=None)
        subparser.add_argument("--min-frequency", type=int, default=None)
        subparser.add_argument("--context-length", type=int, default=None)
        subparser.add_argument("--emb-dim", type=int, default=None)
        subparser.add_argument("--n-heads", type=int, default=None)
        subparser.add_argument("--n-layers", type=int, default=None)
        subparser.add_argument("--drop-rate", type=float, default=None)
        subparser.add_argument("--lr", type=float, default=None)
        subparser.add_argument("--weight-decay", type=float, default=None)
        subparser.add_argument("--grad-clip", type=float, default=None)
        subparser.add_argument("--batch-size", type=int, default=None)
        subparser.add_argument("--num-epochs", type=int, default=None)
        subparser.add_argument("--eval-batches", type=int, default=None)
        subparser.add_argument("--eval-every-steps", type=int, default=None)
        subparser.add_argument("--train-ratio", type=float, default=None)
        subparser.add_argument("--tokenizer-chars", type=int, default=None)
        subparser.add_argument("--stride", type=int, default=None)
        subparser.add_argument("--train-shards", type=int, default=None)
        subparser.add_argument("--patience", type=int, default=None)
        subparser.add_argument("--min-delta", type=float, default=None)
        subparser.add_argument("--prompt", default=None)
        subparser.add_argument("--max-new-tokens", type=int, default=None)
        subparser.add_argument("--qkv-bias", action="store_true")
        subparser.add_argument("--post-norm", action="store_true")
        subparser.add_argument("--norm-first", dest="norm_first", action="store_true", default=None)
        subparser.add_argument(
            "--activation-name",
            choices=["gelu", "gelu_exact", "quick_gelu", "silu", "mish", "squared_relu", "swiglu"],
            default=None,
        )
        subparser.add_argument("--ffn-mult", type=int, default=None)
        subparser.add_argument(
            "--ffn-dropout-position",
            choices=["after_output", "after_activation", "none"],
            default=None,
        )
        subparser.add_argument("--attention-impl", choices=["manual", "sdpa"], default=None)
        subparser.add_argument("--tie-embeddings", action="store_true")
        subparser.add_argument("--init-std", type=float, default=None)
        subparser.add_argument("--norm-eps", type=float, default=None)
        subparser.add_argument("--seed", type=int, default=None)

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
