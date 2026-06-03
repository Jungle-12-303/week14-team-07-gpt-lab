# -*- coding: utf-8 -*-
"""Integration-style tests for the new experiment runner logging schema."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def make_args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        text_file=str(tmp_path / "corpus.txt"),
        output_dir=str(tmp_path / "runs"),
        run_name="tiny_run",
        device="cpu",
        preset="tiny",
        vocab_size=300,
        min_frequency=1,
        context_length=4,
        emb_dim=32,
        n_heads=4,
        n_layers=1,
        drop_rate=0.0,
        lr=1e-3,
        weight_decay=0.01,
        grad_clip=1.0,
        batch_size=2,
        num_epochs=1,
        eval_batches=1,
        eval_every_steps=1,
        train_ratio=0.9,
        tokenizer_chars=120,
        stride=1,
        train_shards=1,
        patience=0,
        min_delta=1e-4,
        prompt="안녕",
        max_new_tokens=2,
        qkv_bias=False,
        post_norm=False,
        norm_first=True,
        activation_name="quick_gelu",
        ffn_mult=3,
        ffn_dropout_position="none",
        attention_impl="manual",
        tie_embeddings=True,
        init_std=0.018,
        norm_eps=1e-4,
        seed=7,
    )


def test_train_once_emits_dense_histories_and_metrics(tmp_path):
    import run_experiment

    args = make_args(tmp_path)
    text = "안녕하세요. 반갑습니다. " * 40
    (tmp_path / "corpus.txt").write_text(text, encoding="utf-8")

    result, *_ = run_experiment.train_once(args, text, return_state=True)

    assert result["config"]["activation_name"] == "quick_gelu"
    assert result["config"]["ffn_mult"] == 3
    assert result["config"]["tie_embeddings"] is True
    assert result["config"]["init_std"] == 0.018
    assert result["config"]["norm_first"] is True

    assert result["step_history"]
    assert result["eval_history"]
    assert result["epoch_history"]

    eval_row = result["eval_history"][0]
    assert "learning_rate" in eval_row
    assert "batch_size" in eval_row
    assert "emb_dim" in eval_row
    assert "dff" in eval_row
    assert "n_heads" in eval_row
    assert "n_layers" in eval_row
    assert "generalization_gap" in eval_row
    assert "overfit_score" in eval_row
    assert "fit_status" in eval_row

    assert "overfit_score" in result
    assert "fit_status" in result
    assert "generalization_gap_delta" in result


def test_run_train_command_writes_dense_artifacts(tmp_path):
    import run_experiment

    args = make_args(tmp_path)
    text = "이 영화는 정말 재미있다. " * 50
    corpus_path = Path(args.text_file)
    corpus_path.write_text(text, encoding="utf-8")

    run_experiment.run_train_command(args)

    run_dirs = sorted(Path(args.output_dir).iterdir())
    assert run_dirs, "run directory should be created"
    run_dir = run_dirs[-1]

    expected_files = [
        "step_history.csv",
        "step_history.jsonl",
        "eval_history.csv",
        "eval_history.jsonl",
        "epoch_history.csv",
        "epoch_history.jsonl",
        "results.csv",
        "results.jsonl",
        "plan.csv",
        "plan.json",
        "train_result.json",
        "config.json",
        "tokenizer.json",
        "run_metrics.svg",
        "step_loss_curve.svg",
        "epoch_loss_curve.svg",
        "best_checkpoint.pt",
        "final_checkpoint.pt",
    ]
    for filename in expected_files:
        assert (run_dir / filename).exists(), f"missing artifact: {filename}"


def test_run_train_experiment_returns_notebook_friendly_state_and_logs(tmp_path):
    import run_experiment

    corpus = "이 영화는 정말 재미있다. " * 50
    checkpoint_path = tmp_path / "best_checkpoint.pt"

    result = run_experiment.run_train_experiment(
        corpus=corpus,
        output_dir=tmp_path / "runs",
        run_name="notebook_run",
        device="cpu",
        corpus_len=300,
        vocab_size=200,
        context_length=8,
        emb_dim=32,
        n_heads=4,
        n_layers=1,
        drop_rate=0.1,
        activation="relu",
        lr=1e-3,
        num_epochs=1,
        batch_size=2,
        eval_iter=1,
        checkpoint_path=checkpoint_path,
    )

    assert result["model"] is not None
    assert result["tokenizer"] is not None
    assert result["config"]["activation_name"] == "relu"
    assert result["plan"]["activation_name"] == "relu"
    assert result["plan"]["batch_size"] == 2
    assert result["plan"]["context_length"] == 8
    assert result["plan"]["emb_dim"] == 32
    assert result["plan"]["n_heads"] == 4
    assert result["plan"]["n_layers"] == 1
    assert result["history"]
    assert {"epoch", "train_loss", "val_loss", "gap"} <= set(result["history"][0].keys())
    assert result["run_dir"].exists()
    assert (result["run_dir"] / "results.csv").exists()
    assert (result["run_dir"] / "step_history.csv").exists()
    assert checkpoint_path.exists()
    assert result["plan"]["tokenizer_chars"] == 300


def test_run_train_experiment_rejects_mismatched_tokenizer_chars_for_direct_corpus(tmp_path):
    import pytest
    import run_experiment

    corpus = "이 영화는 정말 재미있다. " * 50

    with pytest.raises(ValueError, match="tokenizer_chars must match"):
        run_experiment.run_train_experiment(
            corpus=corpus,
            output_dir=tmp_path / "runs",
            run_name="bad_notebook_run",
            corpus_len=300,
            tokenizer_chars=120,
            vocab_size=200,
            context_length=8,
            emb_dim=32,
            n_heads=4,
            n_layers=1,
            batch_size=2,
            num_epochs=1,
            device="cpu",
        )
