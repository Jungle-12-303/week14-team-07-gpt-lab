#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Replay archived train/runs experiments with dense step/epoch logging."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.historical_replay import choose_device, load_historical_plan, replay_historical_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-file", required=True, help="Training corpus text file.")
    parser.add_argument(
        "--output-root",
        default="train/train2/replays",
        help="Directory where replay artifacts should be created.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device to use: auto, cpu, cuda, or mps.",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=1,
        help="Evaluate and log validation metrics every N steps.",
    )
    parser.add_argument(
        "--text-chars",
        type=int,
        default=None,
        help="Optional explicit text prefix length. If omitted, the runner can infer a historical prefix from archived token counts.",
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-id", type=int, help="Historical run id to replay.")
    source.add_argument("--plan-file", help="Path to a historical plan.json file.")
    source.add_argument("--all", action="store_true", help="Replay every archived run.")
    source.add_argument(
        "--run-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Replay a closed run id range.",
    )
    return parser.parse_args()


def iter_plan_paths(args: argparse.Namespace):
    runs_root = ROOT / "train" / "runs"
    if args.run_id is not None:
        yield runs_root / f"run_{args.run_id:03d}_artifacts" / "plan.json"
        return
    if args.plan_file is not None:
        yield Path(args.plan_file)
        return
    if args.all:
        yield from sorted(runs_root.glob("run_*_artifacts/plan.json"))
        return
    start, end = args.run_range
    for run_id in range(start, end + 1):
        yield runs_root / f"run_{run_id:03d}_artifacts" / "plan.json"


def main() -> None:
    args = parse_args()
    text = Path(args.text_file).read_text(encoding="utf-8")
    device = choose_device(args.device)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for plan_path in iter_plan_paths(args):
        plan = load_historical_plan(plan_path)
        expected_total_tokens = None
        sibling_results = Path(plan_path).with_name("results.jsonl")
        if sibling_results.exists():
            archived_result = json.loads(sibling_results.read_text(encoding="utf-8").splitlines()[0])
            expected_total_tokens = int(archived_result["train_token_count"]) + int(archived_result["val_token_count"])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        artifact_dir = output_root / f"run_{int(plan['run_id']):03d}_replay_{timestamp}"
        summary = replay_historical_plan(
            plan=plan,
            text=text,
            artifact_dir=artifact_dir,
            device=device,
            eval_every=max(args.eval_every, 1),
            text_char_limit=args.text_chars,
            expected_total_tokens=expected_total_tokens if args.text_chars is None else None,
        )
        print(
            f"[replay] run {summary['run_id']:03d} complete -> {artifact_dir} "
            f"(val_loss={summary['final_val_loss']:.6f}, fit_status={summary['fit_status']})"
        )


if __name__ == "__main__":
    main()
