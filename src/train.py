# -*- coding: utf-8 -*-
"""GPT pretraining utilities."""

import subprocess
import sys
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


def run_pytest_before_training(
    test_paths: list[str] | None = None,
    repo_root: str | Path | None = None,
) -> None:
    """Run pytest before an automated training experiment."""
    if test_paths is None:
        test_paths = [
            "tests/test_bpe.py",
            "tests/test_dataset.py",
            "tests/test_attention.py",
            "tests/test_model.py",
            "tests/test_train.py",
        ]

    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent
    else:
        repo_root = Path(repo_root)

    command = [sys.executable, "-m", "pytest", *test_paths, "-v"]
    print("Running tests before training:")
    print(" ".join(command))

    result = subprocess.run(command, cwd=repo_root)
    if result.returncode != 0:
        raise RuntimeError("pytest failed, so training was stopped.")

    print("pytest passed. Starting training.")


def get_performance_hypotheses() -> list[dict[str, str]]:
    """Return simple experiment ideas for improving small GPT pretraining."""
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
    """Print epoch-level train/validation losses as a compact table."""
    if not history:
        print("No loss history.")
        return

    print("epoch | train_loss | val_loss | gap | overfit_count")
    print("----- | ---------- | -------- | --- | -------------")
    for row in history:
        print(
            f"{row['epoch']:>5} | "
            f"{row['train_loss']:.4f} | "
            f"{row['val_loss']:.4f} | "
            f"{row['gap']:.4f} | "
            f"{row['overfit_count']}"
        )


def print_performance_hypotheses(
    hypotheses: list[dict[str, str]] | None = None,
) -> None:
    """Print performance-improvement hypotheses as a compact table."""
    if hypotheses is None:
        hypotheses = get_performance_hypotheses()

    print("experiment | hypothesis | expected")
    print("---------- | ---------- | --------")
    for row in hypotheses:
        print(
            f"{row['experiment']} | "
            f"{row['hypothesis']} | "
            f"{row['expected']}"
        )


def run_auto_pretraining_experiment(
    corpus: str,
    device: torch.device | str | None = None,
    corpus_len: int = 10000,
    vocab_size: int = 3000,
    context_length: int = 64,
    emb_dim: int = 64,
    n_heads: int = 1,
    n_layers: int = 1,
    drop_rate: float = 0.0,
    activation: str = "relu",
    num_epochs: int = 20,
    batch_size: int = 16,
    train_ratio: float = 0.9,
    stride: int | None = None,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    patience: int = 3,
    min_delta: float = 0.0,
    eval_iter: int | None = None,
    checkpoint_path: str | Path | None = "best_checkpoint.pt",
    seed: int = 123,
    run_tests: bool = True,
    test_paths: list[str] | None = None,
    repo_root: str | Path | None = None,
) -> dict:
    """
    Run a minimal pretraining experiment with train/val split and early stop.

    Early-stop rule:
    - Track best validation loss.
    - If validation loss does not improve while train loss decreases for
      `patience` consecutive epochs, stop as possible overfitting.
    """
    try:
        from .bpe import BPETokenizer
        from .dataset import create_dataloader
    except ImportError:
        from bpe import BPETokenizer
        from dataset import create_dataloader

    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")

    if run_tests:
        run_pytest_before_training(test_paths=test_paths, repo_root=repo_root)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    torch.manual_seed(seed)

    train_corpus = corpus[:corpus_len]
    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.train(train_corpus)
    token_ids = tokenizer.encode(train_corpus)

    min_tokens_per_split = context_length + 1
    if len(token_ids) < min_tokens_per_split * 2:
        raise ValueError(
            "Not enough tokens for train/validation split. "
            "Increase corpus_len or reduce context_length."
        )

    split_idx = int(len(token_ids) * train_ratio)
    split_idx = max(min_tokens_per_split, split_idx)
    split_idx = min(split_idx, len(token_ids) - min_tokens_per_split)

    train_token_ids = token_ids[:split_idx]
    val_token_ids = token_ids[split_idx:]

    if stride is None:
        stride = context_length

    train_loader = create_dataloader(
        train_token_ids,
        context_length=context_length,
        batch_size=batch_size,
        stride=stride,
        shuffle=True,
    )
    val_loader = create_dataloader(
        val_token_ids,
        context_length=context_length,
        batch_size=batch_size,
        stride=stride,
        shuffle=False,
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

    model = GPTModel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    history = []
    best_val_loss = float("inf")
    best_epoch = 0
    global_step = 0
    overfit_count = 0
    previous_train_loss = None
    stopped_reason = "max_epochs completed"

    for epoch in range(1, num_epochs + 1):
        model.train()
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

        if batches_seen == 0:
            raise ValueError("train_loader produced no batches.")

        train_loss = epoch_loss / batches_seen
        val_loss = calc_loss_loader(val_loader, model, device, eval_iter)
        gap = val_loss - train_loss

        improved_val = val_loss < best_val_loss - min_delta
        if improved_val:
            best_val_loss = val_loss
            best_epoch = epoch
            overfit_count = 0
            if checkpoint_path is not None:
                checkpoint_path = Path(checkpoint_path)
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                save_checkpoint(
                    model,
                    optimizer,
                    epoch=epoch,
                    global_step=global_step,
                    path=str(checkpoint_path),
                )
        else:
            train_improved = (
                previous_train_loss is not None
                and train_loss < previous_train_loss - min_delta
            )
            if train_improved:
                overfit_count += 1
            else:
                overfit_count = 0

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

        previous_train_loss = train_loss

    print_loss_history(history)
    print_performance_hypotheses()
    print(f"Stopped reason: {stopped_reason}")
    print(f"Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")

    if checkpoint_path is not None and best_epoch > 0:
        load_checkpoint(model, optimizer, str(checkpoint_path), device)

    return {
        "model": model,
        "tokenizer": tokenizer,
        "optimizer": optimizer,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "config": config,
        "history": history,
        "hypotheses": get_performance_hypotheses(),
        "stopped_reason": stopped_reason,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "global_step": global_step,
    }


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
