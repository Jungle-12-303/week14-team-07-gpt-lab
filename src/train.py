# -*- coding: utf-8 -*-
"""Training utilities for mini GPT."""

from __future__ import annotations

import math
import os
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
    eval_iter: int,
    start_context: str,
    tokenizer,
    ckpt_freq: int | None = None,
    start_epoch: int = 0,
    global_step: int = 0,
) -> list[float]:
    """Simple epoch-based pretraining loop kept for the existing interface."""

    del start_context, tokenizer, ckpt_freq, global_step

    train_losses = []
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
            calc_loss_loader(val_loader, model, device, num_batches=eval_iter)

    return train_losses


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
