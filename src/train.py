# -*- coding: utf-8 -*-
"""Training utilities for mini GPT."""

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn

try:
    from .model import GPTModel
except ImportError:
    from model import GPTModel


def calc_loss_batch(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: GPTModel,
    device: torch.device,
) -> torch.Tensor:
    """Compute next-token cross-entropy for one batch."""

    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    loss, _ = model(input_batch, targets=target_batch)
    return loss


def calc_loss_loader(
    data_loader,
    model: GPTModel,
    device: torch.device,
    num_batches: int | None = None,
) -> float:
    """Compute average loss over a loader, optionally truncated to num_batches."""

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    if num_batches == 0:
        return 0.0

    total_loss = 0.0
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for batch_idx, (input_batch, target_batch) in enumerate(data_loader):
            if batch_idx >= num_batches:
                break
            total_loss += calc_loss_batch(input_batch, target_batch, model, device).item()

    if was_training:
        model.train()

    return total_loss / num_batches


def compute_perplexity(loss_value: float) -> float:
    """Convert loss to perplexity with a numerical safety cap."""

    return math.exp(min(loss_value, 20.0))


def compute_grad_norm(parameters) -> float:
    """Return the global L2 gradient norm for a parameter iterable."""

    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        total += float(torch.sum(grad * grad).item())
    return math.sqrt(total)


def compute_historical_fit_metrics(
    *,
    initial_train_loss: float,
    initial_val_loss: float,
    final_train_loss: float,
    final_val_loss: float,
    max_steps: int,
) -> dict[str, float | str]:
    """Reconstruct the historical mini-GPT fit metrics used in train/ archives."""

    train_loss_delta = initial_train_loss - final_train_loss
    val_loss_delta = initial_val_loss - final_val_loss
    initial_gap = initial_val_loss - initial_train_loss
    final_gap = final_val_loss - final_train_loss
    gap_delta = final_gap - initial_gap
    train_val_improvement_gap = train_loss_delta - val_loss_delta
    overfit_score = max(0.0, final_gap) + 2.0 * max(0.0, gap_delta)

    if max_steps <= 1 or (train_loss_delta < 0.05 and val_loss_delta < 0.05):
        fit_status = "underfit_or_too_short"
    elif overfit_score >= 0.148:
        fit_status = "overfit_risk"
    else:
        fit_status = "generalizing"

    return {
        "initial_generalization_gap": initial_gap,
        "final_generalization_gap": final_gap,
        "generalization_gap_delta": gap_delta,
        "train_loss_delta": train_loss_delta,
        "val_loss_delta": val_loss_delta,
        "train_val_improvement_gap": train_val_improvement_gap,
        "overfit_score": overfit_score,
        "fit_status": fit_status,
    }


def save_checkpoint(
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    path: str,
) -> None:
    """Save model/optimizer state with epoch and global_step."""

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    model: GPTModel,
    optimizer: torch.optim.Optimizer | None,
    path: str,
    device: torch.device,
) -> tuple[int, int]:
    """Restore a checkpoint created by save_checkpoint()."""

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint.get("epoch", 0), checkpoint.get("global_step", 0)


def generate(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_id: int | None = None,
) -> torch.Tensor:
    """Temperature + top-k sampling helper."""

    was_training = model.training
    model.eval()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            logits = model(idx_cond)
            next_token_logits = logits[:, -1, :]

            if temperature <= 0:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                next_token_logits = next_token_logits / temperature
                if top_k is not None and top_k > 0:
                    top_logits, _ = torch.topk(
                        next_token_logits,
                        min(top_k, next_token_logits.size(-1)),
                    )
                    cutoff = top_logits[:, [-1]]
                    next_token_logits = next_token_logits.masked_fill(
                        next_token_logits < cutoff,
                        float("-inf"),
                    )
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, next_token), dim=1)
            if eos_id is not None and torch.all(next_token == eos_id):
                break

    if was_training:
        model.train()

    return idx


def generate_and_print_sample(
    model: GPTModel,
    tokenizer,
    device: torch.device,
    start_context: str,
    max_new_tokens: int = 50,
    context_size: int = 256,
    temperature: float = 0.8,
    top_k: int | None = 40,
) -> None:
    """Encode prompt, sample continuation, and print decoded text."""

    model.to(device)
    start_ids = tokenizer.encode(start_context, add_bos_eos=False)
    idx = torch.tensor([start_ids], dtype=torch.long, device=device)
    out = generate(
        model=model,
        idx=idx,
        max_new_tokens=max_new_tokens,
        context_size=context_size,
        temperature=temperature,
        top_k=top_k,
        eos_id=getattr(tokenizer, "get_eos_id", lambda: None)(),
    )
    text = tokenizer.decode(out[0].tolist(), skip_special=True)
    print(text)


def train_model(
    model: GPTModel,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int | None,
    start_context: str,
    tokenizer,
    ckpt_freq: int | None = None,
    start_epoch: int = 0,
    global_step: int = 0,
) -> dict[str, list[float]]:
    """Simple epoch-based pretraining loop kept for the existing interface."""

    del start_context, tokenizer, ckpt_freq, global_step

    train_losses = []
    test_losses = []
    model.to(device)

    for epoch in range(start_epoch, start_epoch + num_epochs):
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

        avg_train_loss = total_loss / max(num_batches, 1)
        train_losses.append(avg_train_loss)

        if val_loader is not None and eval_freq > 0 and ((epoch - start_epoch + 1) % eval_freq == 0):
            test_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
            test_losses.append(test_loss)
            print(
                f"Epoch {epoch - start_epoch + 1:02d}: "
                f"train_loss={avg_train_loss:.4f}, test_loss={test_loss:.4f}"
            )

    return {
        "train_losses": train_losses,
        "test_losses": test_losses,
        "val_losses": test_losses,
    }


def run_pytest_before_training(
    repo_root: str | Path | None = None,
    test_paths: list[str] | tuple[str, ...] | None = None,
) -> None:
    """Run project tests before a notebook training experiment."""

    repo_path = Path(repo_root) if repo_root is not None else Path.cwd()
    selected_tests = list(
        test_paths
        if test_paths is not None
        else (
            "tests/test_bpe.py",
            "tests/test_dataset.py",
            "tests/test_attention.py",
            "tests/test_model.py",
            "tests/test_train.py",
        )
    )
    command = [sys.executable, "-m", "pytest", *selected_tests, "-v"]

    print("Running tests before training:")
    print(" ".join(command))
    result = subprocess.run(
        command,
        cwd=str(repo_path),
        text=True,
        capture_output=True,
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if result.returncode != 0:
        raise RuntimeError("pytest failed. Fix tests before training.")

    print("pytest passed. Starting training.")


def get_performance_hypotheses() -> list[dict[str, str]]:
    """Return short experiment ideas for notebook reporting."""

    return [
        {
            "experiment": "vocab_size 3000 -> 1000/2000",
            "hypothesis": "A smaller corpus may train more stably with fewer BPE tokens.",
            "expected": "Fewer broken-looking generations and faster token learning.",
        },
        {
            "experiment": "n_layers 1 -> 2",
            "hypothesis": "One transformer block may be too weak for language patterns.",
            "expected": "Lower validation loss if overfitting is controlled.",
        },
        {
            "experiment": "n_heads 1 -> 2/4",
            "hypothesis": "More heads can split attention into different relation types.",
            "expected": "Better context use when emb_dim is large enough.",
        },
        {
            "experiment": "dropout 0.0 -> 0.1",
            "hypothesis": "No dropout can overfit small corpora quickly.",
            "expected": "Validation loss may become more stable.",
        },
        {
            "experiment": "corpus_len 10000 -> 50000+",
            "hypothesis": "The model needs more examples to learn Korean byte/BPE patterns.",
            "expected": "Better generation quality and lower validation loss.",
        },
        {
            "experiment": "learning_rate 3e-4 -> 1e-4/5e-4",
            "hypothesis": "The chosen learning rate may be too slow or too unstable.",
            "expected": "Find a setting where train and validation loss decrease together.",
        },
    ]


def print_loss_history(history: list[dict[str, float | int]]) -> None:
    """Print epoch-level train/validation losses as a simple table."""

    print("epoch | train_loss | val_loss | gap | overfit_count")
    print("----- | ---------- | -------- | --- | -------------")
    for row in history:
        print(
            f"{int(row['epoch']):5d} | "
            f"{float(row['train_loss']):.4f} | "
            f"{float(row['val_loss']):.4f} | "
            f"{float(row['gap']):.4f} | "
            f"{int(row['overfit_count'])}"
        )


def print_performance_hypotheses(
    hypotheses: list[dict[str, str]] | None = None,
) -> None:
    """Print experiment ideas as a simple Markdown-style table."""

    rows = hypotheses if hypotheses is not None else get_performance_hypotheses()
    print("experiment | hypothesis | expected")
    print("---------- | ---------- | --------")
    for row in rows:
        print(f"{row['experiment']} | {row['hypothesis']} | {row['expected']}")


def run_auto_pretraining_experiment(
    corpus: str,
    device: str | torch.device = "cpu",
    corpus_len: int | None = 10000,
    vocab_size: int = 3000,
    context_length: int = 64,
    emb_dim: int = 64,
    n_heads: int = 1,
    n_layers: int = 1,
    drop_rate: float = 0.0,
    activation: str = "relu",
    num_epochs: int = 20,
    batch_size: int = 32,
    train_ratio: float = 0.9,
    stride: int | None = None,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    patience: int = 3,
    min_delta: float = 0.0,
    eval_iter: int | None = None,
    checkpoint_path: str | Path | None = "best_checkpoint.pt",
    seed: int = 123,
    run_tests: bool = False,
    test_paths: list[str] | tuple[str, ...] | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, object]:
    """Run a small notebook-friendly GPT pretraining experiment.

    The helper trains a BPE tokenizer, splits token ids into train/validation
    loaders, records epoch-level losses, saves the best checkpoint, and stops
    early when validation loss stops improving while train loss keeps falling.
    """

    if run_tests:
        run_pytest_before_training(repo_root=repo_root, test_paths=test_paths)

    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")
    if context_length <= 0:
        raise ValueError("context_length must be positive.")
    if num_epochs <= 0:
        raise ValueError("num_epochs must be positive.")

    torch.manual_seed(seed)
    device_obj = torch.device(device)

    try:
        from .bpe import BPETokenizer
        from .dataset import create_dataloader
    except ImportError:
        from bpe import BPETokenizer
        from dataset import create_dataloader

    train_corpus = corpus if corpus_len is None else corpus[:corpus_len]
    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.train(train_corpus)

    token_ids = tokenizer.encode(train_corpus)
    split_idx = int(len(token_ids) * train_ratio)
    train_token_ids = token_ids[:split_idx]
    val_token_ids = token_ids[split_idx:]

    min_tokens = context_length + 1
    if len(train_token_ids) < min_tokens or len(val_token_ids) < min_tokens:
        raise ValueError(
            "Not enough tokens for train/validation split. "
            "Increase corpus_len or reduce context_length. "
            f"Got train_tokens={len(train_token_ids)}, "
            f"val_tokens={len(val_token_ids)}, required_each={min_tokens}."
        )

    train_loader = create_dataloader(
        train_token_ids,
        context_length=context_length,
        batch_size=batch_size,
        stride=stride,
        shuffle=True,
        drop_last=False,
    )
    val_loader = create_dataloader(
        val_token_ids,
        context_length=context_length,
        batch_size=batch_size,
        stride=stride,
        shuffle=False,
        drop_last=False,
    )

    config = {
        "vocab_size": len(tokenizer.id_to_token),
        "context_length": context_length,
        "emb_dim": emb_dim,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "drop_rate": drop_rate,
        "qkv_bias": False,
        "activation": activation,
    }
    model = GPTModel(config).to(device_obj)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    best_val_loss = float("inf")
    best_epoch = 0
    overfit_count = 0
    previous_train_loss: float | None = None
    global_step = 0
    history: list[dict[str, float | int]] = []
    stopped_reason = "max_epochs completed"

    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    if checkpoint is not None and checkpoint.parent != Path(""):
        checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device_obj)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            global_step += 1

        train_loss = total_loss / max(num_batches, 1)
        val_loss = calc_loss_loader(
            val_loader,
            model,
            device_obj,
            num_batches=eval_iter,
        )
        gap = val_loss - train_loss

        improved = val_loss < best_val_loss - min_delta
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            overfit_count = 0
            if checkpoint is not None:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    global_step=global_step,
                    path=str(checkpoint),
                )
        else:
            train_loss_decreased = (
                previous_train_loss is not None
                and train_loss < previous_train_loss
            )
            if train_loss_decreased:
                overfit_count += 1
            else:
                overfit_count = 0

        previous_train_loss = train_loss
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "gap": gap,
                "overfit_count": overfit_count,
            }
        )

        print(
            f"Epoch {epoch:02d}: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"gap={gap:.4f}, "
            f"overfit_count={overfit_count}"
        )

        if overfit_count >= patience:
            stopped_reason = "early stopping: possible overfitting"
            break

    print_loss_history(history)
    print_performance_hypotheses()
    print(f"Stopped reason: {stopped_reason}")
    print(f"Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")

    if checkpoint is not None and checkpoint.exists():
        load_checkpoint(model, optimizer, str(checkpoint), device_obj)

    return {
        "model": model,
        "tokenizer": tokenizer,
        "config": config,
        "optimizer": optimizer,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "history": history,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "stopped_reason": stopped_reason,
        "checkpoint_path": checkpoint,
    }


def plot_losses(train_losses: list[float], val_losses: list[float] | None = None) -> None:
    """Plot train/val losses using a non-interactive backend."""

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcfg")
    import matplotlib.pyplot as plt

    plt.plot(train_losses, label="Train")
    if val_losses is not None:
        plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training / Validation Loss")
    plt.show()
