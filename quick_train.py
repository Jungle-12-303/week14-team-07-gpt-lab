#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""짧은 CPU 학습으로 체크포인트를 빠르게 만드는 스크립트."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.bpe import BPETokenizer
from src.dataset import create_dataloader
from src.model import GPTModel
from src.train import calc_loss_batch


def main() -> None:
    text = Path("data/nsmc_lm_train.txt").read_text(encoding="utf-8")[:30000]
    output_dir = Path("artifacts/manual_quick")
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_chars = 8000
    vocab_size = 400
    context_length = 32
    config = {
        "vocab_size": vocab_size,
        "context_length": context_length,
        "emb_dim": 48,
        "n_heads": 4,
        "n_layers": 2,
        "drop_rate": 0.1,
        "qkv_bias": False,
        "pre_norm": True,
    }

    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.train(text[:tokenizer_chars])
    token_ids = tokenizer.encode(text)
    loader = create_dataloader(
        token_ids,
        context_length=context_length,
        batch_size=8,
        stride=4,
        shuffle=True,
    )

    model = GPTModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4)
    device = torch.device("cpu")
    model.to(device)

    losses: list[float] = []
    max_steps = 20
    for step, (inp, tgt) in enumerate(loader, start=1):
        loss = calc_loss_batch(inp, tgt, model, device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if step % 10 == 0 or step == 1:
            print(f"step={step} loss={loss.item():.4f}", flush=True)
        if step >= max_steps:
            break

    avg_loss = sum(losses) / len(losses)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": 1,
            "global_step": len(losses),
            "config": config,
            "tokenizer_chars": tokenizer_chars,
            "vocab_size": vocab_size,
        },
        output_dir / "checkpoint.pt",
    )

    (output_dir / "result.json").write_text(
        json.dumps(
            {
                "avg_loss": avg_loss,
                "steps": len(losses),
                "max_steps": max_steps,
                "config": config,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"saved_checkpoint={output_dir / 'checkpoint.pt'}", flush=True)
    print(f"avg_loss={avg_loss:.4f}", flush=True)


if __name__ == "__main__":
    main()
