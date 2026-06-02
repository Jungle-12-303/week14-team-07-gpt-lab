#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""콘솔에서 프롬프트를 넣어 GPT 생성 결과를 확인하는 최소 스크립트."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.bpe import BPETokenizer
from src.model import GPTModel
from src.train import generate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="콘솔 GPT 샘플 생성기")
    parser.add_argument("--text-file", default="data/nsmc_lm_train.txt")
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--vocab-size", type=int, default=300)
    parser.add_argument("--context-length", type=int, default=32)
    parser.add_argument("--emb-dim", type=int, default=32)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--drop-rate", type=float, default=0.0)
    parser.add_argument("--tokenizer-chars", type=int, default=5000)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--qkv-bias", action="store_true")
    parser.add_argument("--post-norm", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)

    text = Path(args.text_file).read_text(encoding="utf-8")
    tokenizer = BPETokenizer(vocab_size=args.vocab_size)
    tokenizer.train(text[: args.tokenizer_chars])

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
    model = GPTModel(config).to(device)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    print("프롬프트를 입력하세요. 종료하려면 빈 줄을 입력하세요.")
    while True:
        prompt = input("prompt> ").strip()
        if not prompt:
            break

        prompt_ids = tokenizer.encode(prompt, add_bos_eos=False)
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        out = generate(
            model=model,
            idx=idx,
            max_new_tokens=args.max_new_tokens,
            context_size=args.context_length,
            temperature=args.temperature,
            top_k=args.top_k,
            eos_id=tokenizer.get_eos_id(),
        )
        print(tokenizer.decode(out[0].tolist(), skip_special=True))


if __name__ == "__main__":
    main()
