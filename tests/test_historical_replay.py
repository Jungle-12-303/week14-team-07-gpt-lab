# -*- coding: utf-8 -*-
"""Historical replay compatibility tests."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def test_bpe_respects_min_frequency():
    from bpe import BPETokenizer

    tokenizer = BPETokenizer(vocab_size=400, min_frequency=2)
    tokenizer.train("ab")
    assert len(tokenizer.id_to_token) == 260


def test_historical_overfit_score_formula_and_status():
    from train import compute_historical_fit_metrics

    metrics = compute_historical_fit_metrics(
        initial_train_loss=6.424758791923523,
        initial_val_loss=6.4218573570251465,
        final_train_loss=5.507611155509949,
        final_val_loss=5.55439821879069,
        max_steps=80,
    )
    assert abs(metrics["overfit_score"] - 0.14616405963897794) < 1e-9
    assert metrics["fit_status"] == "generalizing"

    metrics = compute_historical_fit_metrics(
        initial_train_loss=6.417887210845947,
        initial_val_loss=6.41053318977356,
        final_train_loss=5.710260987281799,
        final_val_loss=5.763571500778198,
        max_steps=40,
    )
    assert abs(metrics["overfit_score"] - 0.17463958263397217) < 1e-9
    assert metrics["fit_status"] == "overfit_risk"
