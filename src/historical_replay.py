# -*- coding: utf-8 -*-
"""Historical experiment replay with dense logging."""

from __future__ import annotations

import csv
import json
import os
import random
import time
from pathlib import Path

import torch

from .bpe import BPETokenizer
from .dataset import create_dataloader
from .model import GPTModel
from .train import (
    calc_loss_batch,
    calc_loss_loader,
    compute_grad_norm,
    compute_historical_fit_metrics,
    compute_perplexity,
    save_checkpoint,
)


def load_historical_plan(plan_path: str | Path) -> dict:
    """Load a historical plan.json file, accepting either dict or [dict]."""

    data = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        if not data:
            raise ValueError(f"Plan file is empty: {plan_path}")
        return data[0]
    if not isinstance(data, dict):
        raise TypeError(f"Unsupported plan payload type: {type(data)!r}")
    return data


def set_replay_seed(seed: int) -> None:
    """Set Python and PyTorch seeds for deterministic replays."""

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(device_name: str = "auto") -> torch.device:
    """Resolve the best available torch device."""

    if device_name != "auto":
        return torch.device(device_name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def split_train_val_ids(
    token_ids: list[int],
    context_length: int,
    train_ratio: float,
) -> tuple[list[int], list[int]]:
    """Split token ids into train/val segments with a safe minimum val length."""

    split_idx = max(int(len(token_ids) * train_ratio), context_length + 2)
    train_ids = token_ids[:split_idx]
    val_ids = token_ids[split_idx:]
    if len(val_ids) < context_length + 2:
        val_ids = train_ids[-(context_length + 2) :]
    return train_ids, val_ids


def infer_text_prefix_by_token_count(
    *,
    text: str,
    vocab_size: int,
    min_frequency: int,
    target_total_tokens: int,
) -> tuple[str, int, int]:
    """Find a text prefix whose trained+encoded token count is closest to target_total_tokens."""

    if target_total_tokens <= 0:
        return text, len(text), 0

    def token_count_for(chars: int) -> int:
        prefix = text[:chars]
        tokenizer = BPETokenizer(vocab_size=vocab_size, min_frequency=min_frequency)
        tokenizer.train(prefix)
        return len(tokenizer.encode(prefix))

    lo = 1
    hi = len(text)
    best_chars = hi
    best_count = token_count_for(hi)
    best_diff = abs(best_count - target_total_tokens)

    while lo <= hi:
        mid = (lo + hi) // 2
        count = token_count_for(mid)
        diff = abs(count - target_total_tokens)
        if diff < best_diff:
            best_chars = mid
            best_count = count
            best_diff = diff
        if count < target_total_tokens:
            lo = mid + 1
        elif count > target_total_tokens:
            hi = mid - 1
        else:
            return text[:mid], mid, count

    window_start = max(1, best_chars - 200)
    window_end = min(len(text), best_chars + 200)
    for chars in range(window_start, window_end + 1):
        count = token_count_for(chars)
        diff = abs(count - target_total_tokens)
        if diff < best_diff:
            best_chars = chars
            best_count = count
            best_diff = diff
            if diff == 0:
                break

    return text[:best_chars], best_chars, best_count


def build_model_config(plan: dict, actual_vocab_size: int) -> dict:
    """Translate historical plan keys into the current GPTModel config."""

    return {
        "vocab_size": actual_vocab_size,
        "context_length": plan["context_length"],
        "emb_dim": plan["emb_dim"],
        "n_heads": plan["n_heads"],
        "n_layers": plan["n_layers"],
        "drop_rate": plan["drop_rate"],
        "qkv_bias": plan["qkv_bias"],
        "ffn_mult": plan["ffn_mult"],
        "pre_norm": plan.get("norm_first", False),
        "norm_first": plan.get("norm_first", False),
        "norm_eps": plan.get("norm_eps", 1e-5),
        "activation_name": plan["activation_name"],
        "ffn_dropout_position": plan.get("ffn_dropout_position", "after_output"),
        "attention_impl": plan.get("attention_impl", "manual"),
        "tie_embeddings": plan.get("tie_embeddings", False),
        "init_std": plan.get("init_std", 0.02),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as jsonl_file:
        for row in rows:
            jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _save_line_plot(
    output_path: Path,
    series: list[tuple[list[float], list[float], str]],
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcfg")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4))
    for x_values, y_values, label in series:
        plt.plot(x_values, y_values, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _save_run_metrics_plot(
    output_path: Path,
    step_history: list[dict],
    eval_history: list[dict],
) -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcfg")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    step_x = [row["step"] for row in step_history]
    step_loss = [row["train_step_loss"] for row in step_history]
    eval_x = [row["step"] for row in eval_history]
    train_eval = [row["train_eval_loss"] for row in eval_history]
    val_eval = [row["val_loss"] for row in eval_history]
    gap_eval = [row["generalization_gap"] for row in eval_history]
    overfit_eval = [row["overfit_score"] for row in eval_history]

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    axes[0].plot(step_x, step_loss, label="train step loss", alpha=0.5)
    axes[0].plot(eval_x, train_eval, label="train eval loss")
    axes[0].plot(eval_x, val_eval, label="val loss")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(eval_x, [compute_perplexity(v) for v in train_eval], label="train ppl")
    axes[1].plot(eval_x, [compute_perplexity(v) for v in val_eval], label="val ppl")
    axes[1].set_ylabel("Perplexity")
    axes[1].legend()

    axes[2].plot(eval_x, gap_eval, label="generalization gap")
    axes[2].plot(eval_x, overfit_eval, label="overfit score")
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("Gap / Score")
    axes[2].legend()

    fig.suptitle("Replay Metrics")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_dense_replay_artifacts(
    artifact_dir: Path,
    step_history: list[dict],
    eval_history: list[dict],
    epoch_history: list[dict],
) -> None:
    """Persist dense logs and common metric plots."""

    _write_csv(artifact_dir / "step_history.csv", step_history)
    _write_jsonl(artifact_dir / "step_history.jsonl", step_history)
    _write_csv(artifact_dir / "eval_history.csv", eval_history)
    _write_jsonl(artifact_dir / "eval_history.jsonl", eval_history)
    _write_csv(artifact_dir / "epoch_history.csv", epoch_history)
    _write_jsonl(artifact_dir / "epoch_history.jsonl", epoch_history)

    step_x = [row["step"] for row in step_history]
    step_loss = [row["train_step_loss"] for row in step_history]
    eval_x = [row["step"] for row in eval_history]
    train_eval = [row["train_eval_loss"] for row in eval_history]
    val_eval = [row["val_loss"] for row in eval_history]
    gap_eval = [row["generalization_gap"] for row in eval_history]
    train_ppl = [row["train_perplexity"] for row in eval_history]
    val_ppl = [row["val_perplexity"] for row in eval_history]

    _save_line_plot(
        artifact_dir / "step_loss_curve.svg",
        [
            (step_x, step_loss, "train step loss"),
            (eval_x, train_eval, "train eval loss"),
            (eval_x, val_eval, "val loss"),
        ],
        xlabel="Step",
        ylabel="Loss",
        title="Step-level Loss Curves",
    )
    _save_line_plot(
        artifact_dir / "step_perplexity_curve.svg",
        [
            (eval_x, train_ppl, "train perplexity"),
            (eval_x, val_ppl, "val perplexity"),
        ],
        xlabel="Step",
        ylabel="Perplexity",
        title="Step-level Perplexity Curves",
    )
    _save_line_plot(
        artifact_dir / "step_generalization_gap.svg",
        [(eval_x, gap_eval, "generalization gap")],
        xlabel="Step",
        ylabel="Gap",
        title="Step-level Generalization Gap",
    )

    epoch_x = [row["epoch"] for row in epoch_history]
    epoch_train = [row["avg_train_step_loss"] for row in epoch_history]
    epoch_train_eval = [row["train_eval_loss"] for row in epoch_history]
    epoch_val = [row["val_loss"] for row in epoch_history]
    epoch_train_ppl = [row["train_perplexity"] for row in epoch_history]
    epoch_val_ppl = [row["val_perplexity"] for row in epoch_history]
    epoch_gap = [row["generalization_gap"] for row in epoch_history]

    _save_line_plot(
        artifact_dir / "epoch_loss_curve.svg",
        [
            (epoch_x, epoch_train, "avg train step loss"),
            (epoch_x, epoch_train_eval, "train eval loss"),
            (epoch_x, epoch_val, "val loss"),
        ],
        xlabel="Epoch",
        ylabel="Loss",
        title="Epoch-level Loss Curves",
    )
    _save_line_plot(
        artifact_dir / "epoch_perplexity_curve.svg",
        [
            (epoch_x, epoch_train_ppl, "train perplexity"),
            (epoch_x, epoch_val_ppl, "val perplexity"),
        ],
        xlabel="Epoch",
        ylabel="Perplexity",
        title="Epoch-level Perplexity Curves",
    )
    _save_line_plot(
        artifact_dir / "epoch_generalization_gap.svg",
        [(epoch_x, epoch_gap, "generalization gap")],
        xlabel="Epoch",
        ylabel="Gap",
        title="Epoch-level Generalization Gap",
    )

    _save_run_metrics_plot(artifact_dir / "run_metrics.svg", step_history, eval_history)


def replay_historical_plan(
    *,
    plan: dict,
    text: str,
    artifact_dir: str | Path,
    device: torch.device,
    eval_every: int = 1,
    text_char_limit: int | None = None,
    expected_total_tokens: int | None = None,
) -> dict:
    """Replay one archived experiment and write dense logs to artifact_dir."""

    artifact_path = Path(artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)

    set_replay_seed(int(plan["seed"]))

    replay_text = text
    inferred_text_chars = None
    inferred_total_tokens = None
    if text_char_limit is not None:
        replay_text = text[:text_char_limit]
        inferred_text_chars = text_char_limit
    elif expected_total_tokens is not None:
        replay_text, inferred_text_chars, inferred_total_tokens = infer_text_prefix_by_token_count(
            text=text,
            vocab_size=int(plan["vocab_size"]),
            min_frequency=int(plan.get("min_frequency", 1)),
            target_total_tokens=expected_total_tokens,
        )

    tokenizer = BPETokenizer(
        vocab_size=int(plan["vocab_size"]),
        min_frequency=int(plan.get("min_frequency", 1)),
    )
    tokenizer.train(replay_text)
    token_ids = tokenizer.encode(replay_text)
    train_ids, val_ids = split_train_val_ids(
        token_ids=token_ids,
        context_length=int(plan["context_length"]),
        train_ratio=float(plan["train_ratio"]),
    )
    actual_vocab_size = len(tokenizer.id_to_token)

    model = GPTModel(build_model_config(plan, actual_vocab_size))
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(plan["learning_rate"]),
        weight_decay=float(plan["weight_decay"]),
    )

    context_length = int(plan["context_length"])
    batch_size = int(plan["batch_size"])
    stride = plan.get("stride")
    eval_batches = int(plan.get("eval_batches", 4))
    max_steps = int(plan["max_steps"])
    grad_clip = float(plan.get("grad_clip", 0.0))

    train_eval_loader = create_dataloader(
        train_ids,
        context_length=context_length,
        batch_size=batch_size,
        stride=stride,
        shuffle=False,
    )
    val_loader = create_dataloader(
        val_ids,
        context_length=context_length,
        batch_size=batch_size,
        stride=stride,
        shuffle=False,
    )

    initial_train_loss = calc_loss_loader(train_eval_loader, model, device, num_batches=eval_batches)
    initial_val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_batches)
    initial_fit = compute_historical_fit_metrics(
        initial_train_loss=initial_train_loss,
        initial_val_loss=initial_val_loss,
        final_train_loss=initial_train_loss,
        final_val_loss=initial_val_loss,
        max_steps=max_steps,
    )

    step_history: list[dict] = []
    eval_history: list[dict] = [
        {
            "step": 0,
            "epoch": 0,
            "learning_rate": float(plan["learning_rate"]),
            "batch_size": batch_size,
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

    run_start = time.perf_counter()
    tokens_seen = 0
    global_step = 0
    epoch = 0
    last_step_loss = initial_train_loss

    while global_step < max_steps:
        epoch += 1
        epoch_losses: list[float] = []
        train_loader = create_dataloader(
            train_ids,
            context_length=context_length,
            batch_size=batch_size,
            stride=stride,
            shuffle=True,
        )

        for input_batch, target_batch in train_loader:
            if global_step >= max_steps:
                break

            model.train()
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()

            grad_norm = compute_grad_norm(model.parameters())
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            global_step += 1
            last_step_loss = loss.item()
            epoch_losses.append(last_step_loss)
            tokens_seen += int(input_batch.numel())

            step_history.append(
                {
                    "step": global_step,
                    "epoch": epoch,
                    "train_step_loss": last_step_loss,
                    "grad_norm": grad_norm,
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "tokens_seen": tokens_seen,
                    "elapsed_sec": time.perf_counter() - run_start,
                }
            )

            if global_step % eval_every == 0 or global_step == max_steps:
                train_eval_loss = calc_loss_loader(
                    train_eval_loader,
                    model,
                    device,
                    num_batches=eval_batches,
                )
                val_loss = calc_loss_loader(
                    val_loader,
                    model,
                    device,
                    num_batches=eval_batches,
                )
                fit_metrics = compute_historical_fit_metrics(
                    initial_train_loss=initial_train_loss,
                    initial_val_loss=initial_val_loss,
                    final_train_loss=train_eval_loss,
                    final_val_loss=val_loss,
                    max_steps=max_steps,
                )
                eval_history.append(
                    {
                        "step": global_step,
                        "epoch": epoch,
                        "learning_rate": float(plan["learning_rate"]),
                        "batch_size": batch_size,
                        "train_eval_loss": train_eval_loss,
                        "val_loss": val_loss,
                        "train_perplexity": compute_perplexity(train_eval_loss),
                        "val_perplexity": compute_perplexity(val_loss),
                        "generalization_gap": fit_metrics["final_generalization_gap"],
                        "generalization_gap_delta": fit_metrics["generalization_gap_delta"],
                        "train_val_improvement_gap": fit_metrics["train_val_improvement_gap"],
                        "overfit_score": fit_metrics["overfit_score"],
                        "fit_status": fit_metrics["fit_status"],
                        "elapsed_sec": time.perf_counter() - run_start,
                        "tokens_seen": tokens_seen,
                    }
                )

        if not epoch_losses:
            break

        last_eval = eval_history[-1]
        epoch_history.append(
            {
                "epoch": epoch,
                "steps_completed": global_step,
                "learning_rate": float(plan["learning_rate"]),
                "batch_size": batch_size,
                "avg_train_step_loss": sum(epoch_losses) / len(epoch_losses),
                "train_eval_loss": last_eval["train_eval_loss"],
                "val_loss": last_eval["val_loss"],
                "train_perplexity": last_eval["train_perplexity"],
                "val_perplexity": last_eval["val_perplexity"],
                "generalization_gap": last_eval["generalization_gap"],
                "overfit_score": last_eval["overfit_score"],
                "fit_status": last_eval["fit_status"],
                "elapsed_sec": last_eval["elapsed_sec"],
                "tokens_seen": last_eval["tokens_seen"],
            }
        )

    final_elapsed = time.perf_counter() - run_start
    final_eval = eval_history[-1]
    summary_metrics = compute_historical_fit_metrics(
        initial_train_loss=initial_train_loss,
        initial_val_loss=initial_val_loss,
        final_train_loss=float(final_eval["train_eval_loss"]),
        final_val_loss=float(final_eval["val_loss"]),
        max_steps=max_steps,
    )

    summary = dict(plan)
    summary.update(
        {
            "actual_vocab_size": actual_vocab_size,
            "train_token_count": len(train_ids),
            "val_token_count": len(val_ids),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "initial_train_loss": initial_train_loss,
            "initial_val_loss": initial_val_loss,
            "last_step_loss": last_step_loss,
            "final_train_loss": float(final_eval["train_eval_loss"]),
            "final_val_loss": float(final_eval["val_loss"]),
            "elapsed_sec": final_elapsed,
            "tokens_seen": tokens_seen,
            "tokens_per_sec": tokens_seen / final_elapsed if final_elapsed > 0 else 0.0,
            "device": str(device),
            "text_chars_used": inferred_text_chars if inferred_text_chars is not None else len(replay_text),
            "replay_total_tokens": len(token_ids),
        }
    )
    if inferred_total_tokens is not None:
        summary["inferred_total_tokens"] = inferred_total_tokens
    summary.update(summary_metrics)

    plan_payload = [dict(plan)]
    (artifact_path / "plan.json").write_text(
        json.dumps(plan_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_csv(artifact_path / "plan.csv", [dict(plan)])
    _write_jsonl(artifact_path / "results.jsonl", [summary])
    _write_csv(artifact_path / "results.csv", [summary])
    tokenizer.save(artifact_path / "tokenizer.json")
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        global_step=global_step,
        path=str(artifact_path / "checkpoint.pt"),
    )
    save_dense_replay_artifacts(
        artifact_dir=artifact_path,
        step_history=step_history,
        eval_history=eval_history,
        epoch_history=epoch_history,
    )
    return summary
