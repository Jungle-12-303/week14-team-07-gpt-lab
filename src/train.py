# -*- coding: utf-8 -*-
"""GPT pretraining utilities."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

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
    """Move one batch to device and compute next-token prediction loss."""
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
    """Compute average loss over a DataLoader or over the first num_batches."""
    if len(data_loader) == 0:
        return float("nan")

    model.eval()
    total_loss = 0.0
    batches_seen = 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    with torch.no_grad():
        for input_batch, target_batch in data_loader:
            if batches_seen >= num_batches:
                break
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
            batches_seen += 1

    model.train()

    if batches_seen == 0:
        return float("nan")
    return total_loss / batches_seen


def save_checkpoint(
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    path: str,
) -> None:
    """Save model/optimizer state and training position."""
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
    """Load checkpoint and restore model/optimizer state."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    global_step = checkpoint.get("global_step", 0)
    return epoch, global_step


def generate(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_id: int | None = None,
) -> torch.Tensor:
    """Generate tokens with greedy decoding or temperature/top-k sampling."""
    model.eval()

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]

        if top_k is not None:
            top_values, _ = torch.topk(logits, top_k)
            min_top_value = top_values[:, -1].unsqueeze(-1)
            logits = torch.where(
                logits < min_top_value,
                torch.full_like(logits, float("-inf")),
                logits,
            )

        if temperature == 0:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, idx_next), dim=1)

        if eos_id is not None and torch.all(idx_next == eos_id):
            break

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
    """Encode a prompt, generate tokens, decode them, and print the sample."""
    model.eval()

    encoded = tokenizer.encode(start_context, add_bos_eos=False)
    idx = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)
    generated = generate(
        model,
        idx,
        max_new_tokens=max_new_tokens,
        context_size=context_size,
        temperature=temperature,
        top_k=top_k,
        eos_id=tokenizer.get_eos_id(),
    )
    text = tokenizer.decode(generated[0].tolist(), skip_special=True)
    print(text)

    model.train()


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
    """Run the pretraining loop and return epoch-level train losses."""
    model.to(device)
    model.train()

    train_losses = []

    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0.0
        batches_seen = 0

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batches_seen += 1
            global_step += 1

            if eval_freq > 0 and global_step % eval_freq == 0:
                train_loss = calc_loss_loader(train_loader, model, device, eval_iter)
                val_loss = calc_loss_loader(val_loader, model, device, eval_iter)
                print(
                    f"Ep {epoch + 1}, Step {global_step}: "
                    f"train loss {train_loss:.3f}, val loss {val_loss:.3f}"
                )
                generate_and_print_sample(
                    model,
                    tokenizer,
                    device,
                    start_context,
                    context_size=model.config["context_length"],
                )

            if ckpt_freq is not None and ckpt_freq > 0 and global_step % ckpt_freq == 0:
                ckpt_path = Path(f"checkpoint_step_{global_step}.pt")
                save_checkpoint(model, optimizer, epoch, global_step, str(ckpt_path))

        if batches_seen > 0:
            train_losses.append(epoch_loss / batches_seen)

    return train_losses


def plot_losses(train_losses: list[float], val_losses: list[float] | None = None) -> None:
    """Plot training/validation loss curves."""
    plt.plot(train_losses, label="Train")
    if val_losses is not None:
        plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training / Validation Loss")
    plt.show()
