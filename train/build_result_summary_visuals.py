from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"
VIS_DIR = ROOT / "visuals"


COMPARISONS = [
    {
        "slug": "tie_embeddings",
        "title": "Tie Embeddings",
        "before_run": 2,
        "after_run": 4,
        "before_label": "run 002\nFalse",
        "after_label": "run 004\nTrue",
        "note": "tie_embeddings=False -> True",
    },
    {
        "slug": "activation_quick_gelu",
        "title": "Activation Swap",
        "before_run": 7,
        "after_run": 8,
        "before_label": "run 007\ngelu",
        "after_label": "run 008\nquick_gelu",
        "note": "activation_name=gelu -> quick_gelu",
    },
    {
        "slug": "context_length_48",
        "title": "Context Length",
        "before_run": 19,
        "after_run": 21,
        "before_label": "run 019\nctx=64",
        "after_label": "run 021\nctx=48",
        "note": "context_length=64 -> 48",
    },
    {
        "slug": "ffn_mult_3",
        "title": "FFN Width",
        "before_run": 63,
        "after_run": 66,
        "before_label": "run 063\nmult=4",
        "after_label": "run 066\nmult=3",
        "note": "ffn_mult=4 -> 3",
    },
    {
        "slug": "stride_rescue",
        "title": "Stride Rescue",
        "before_run": 85,
        "after_run": 86,
        "before_label": "run 085\nstride=24",
        "after_label": "run 086\nstride=16",
        "note": "seed303 rescue: stride=24 -> 16",
    },
    {
        "slug": "max_steps_105",
        "title": "Longer Horizon",
        "before_run": 108,
        "after_run": 109,
        "before_label": "run 108\n100 steps",
        "after_label": "run 109\n105 steps",
        "note": "max_steps=100 -> 105",
    },
]

METRICS = [
    ("final_val_loss", "Final Val Loss"),
    ("final_generalization_gap", "Generalization Gap"),
    ("overfit_score", "Overfit Score"),
]

SECTION_ONE_RUNS = [
    {
        "run": 2,
        "label": "run 002\nbaseline",
    },
    {
        "run": 3,
        "label": "run 003\ndrop_rate↑",
    },
    {
        "run": 5,
        "label": "run 005\nweight_decay↑",
    },
]

SECTION_THREE_ACTIVATION_RUNS = {
    "gelu_exact": [60, 61, 62],
    "silu": [63, 64, 65],
    "mish": [72, 73, 74],
    "quick_gelu": [75, 76, 77],
}

SECTION_FOUR_FFN_RUNS = [
    ("seed151", 65, 68),
    ("seed202", 63, 66),
    ("seed134", 64, 67),
]

SECTION_SIX_STRIDE_RUNS = [
    (24, 85),
    (20, 95),
    (16, 86),
]

SECTION_SEVEN_MAX_STEPS_RUNS = [98, 101, 102, 103, 104, 108, 109]


def load_run(run_id: int) -> dict:
    path = RUNS_DIR / f"run_{run_id:03d}_artifacts" / "results.jsonl"
    with path.open() as f:
        return json.loads(f.readline())


def save_csv(rows: list[dict]) -> None:
    out = ROOT / "result_summary_comparisons.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "comparison",
                "note",
                "before_run",
                "after_run",
                "metric",
                "before_value",
                "after_value",
                "delta_after_minus_before",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def build_section_one_plot() -> None:
    fig, axes = plt.subplots(ncols=3, figsize=(10, 3.6), constrained_layout=True)
    colors = ["#64748b", "#2563eb", "#0f766e"]
    runs = [load_run(item["run"]) for item in SECTION_ONE_RUNS]
    labels = [item["label"] for item in SECTION_ONE_RUNS]

    for ax, (metric_key, metric_title) in zip(axes, METRICS):
        values = [run[metric_key] for run in runs]
        ax.bar(range(len(values)), values, color=colors, width=0.6)
        ax.set_title(metric_title, fontsize=10)
        ax.set_xticks(range(len(values)), labels, fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.axhline(0, color="#cbd5e1", linewidth=0.8)
        for x, value in enumerate(values):
            ax.text(
                x,
                value,
                f"{value:.4f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=8,
            )

    fig.suptitle("Section 1: Baseline vs Regularization-only Tweaks", fontsize=12, fontweight="bold")
    fig.savefig(VIS_DIR / "section_1_baseline_regularization.svg", format="svg")
    fig.savefig(VIS_DIR / "section_1_baseline_regularization.png", dpi=180)
    plt.close(fig)


def build_section_two_plot(comp: dict) -> None:
    before = load_run(comp["before_run"])
    after = load_run(comp["after_run"])

    fig, ax = plt.subplots(figsize=(8, 3.8), constrained_layout=True)
    metric_labels = [m[1] for m in METRICS]
    before_vals = [before[k] for k, _ in METRICS]
    after_vals = [after[k] for k, _ in METRICS]
    deltas = [a - b for a, b in zip(after_vals, before_vals)]
    colors = ["#2563eb" if d < 0 else "#dc2626" for d in deltas]

    ax.barh(metric_labels, deltas, color=colors, alpha=0.85)
    ax.axvline(0, color="#0f172a", linewidth=1)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.set_title(f"{comp['title']}: change after intervention", fontsize=12, fontweight="bold")
    ax.set_xlabel("after - before (negative is better here)")
    for idx, d in enumerate(deltas):
        ax.text(d, idx, f" {d:+.4f}", va="center", ha="left" if d >= 0 else "right", fontsize=8)
    ax.text(
        0.02,
        0.02,
        f"{comp['before_label']}  ->  {comp['after_label']}\n{comp['note']}",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#cbd5e1"),
    )

    fig.savefig(VIS_DIR / f"section_{comp['slug']}.svg", format="svg")
    fig.savefig(VIS_DIR / f"section_{comp['slug']}.png", dpi=180)
    plt.close(fig)


def build_section_three_plot() -> None:
    fig, axes = plt.subplots(ncols=2, figsize=(10, 4), constrained_layout=True)
    ax1, ax2 = axes
    activation_order = list(SECTION_THREE_ACTIVATION_RUNS.keys())
    colors = {
        "gelu_exact": "#475569",
        "silu": "#2563eb",
        "mish": "#0f766e",
        "quick_gelu": "#7c3aed",
    }

    val_means = []
    overfit_means = []
    for act in activation_order:
        runs = [load_run(r) for r in SECTION_THREE_ACTIVATION_RUNS[act]]
        val_means.append(sum(r["final_val_loss"] for r in runs) / len(runs))
        overfit_means.append(sum(r["overfit_score"] for r in runs) / len(runs))

    ax1.bar(activation_order, val_means, color=[colors[a] for a in activation_order])
    ax1.set_title("Mean Final Val Loss by Activation", fontsize=11)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    for x, y in enumerate(val_means):
        ax1.text(x, y, f"{y:.4f}", ha="center", va="bottom", fontsize=8)

    ax2.bar(activation_order, overfit_means, color=[colors[a] for a in activation_order])
    ax2.set_title("Mean Overfit Score by Activation", fontsize=11)
    ax2.grid(axis="y", linestyle="--", alpha=0.3)
    for x, y in enumerate(overfit_means):
        ax2.text(x, y, f"{y:.4f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Section 3: activation comparison across seeds 151/202/134", fontsize=12, fontweight="bold")
    fig.savefig(VIS_DIR / "section_activation_quick_gelu.svg", format="svg")
    fig.savefig(VIS_DIR / "section_activation_quick_gelu.png", dpi=180)
    plt.close(fig)


def build_section_four_plot() -> None:
    fig, axes = plt.subplots(ncols=2, figsize=(10, 4), constrained_layout=True)
    ax1, ax2 = axes
    y_positions = [2, 1, 0]
    for y, (seed_label, run_mult4, run_mult3) in zip(y_positions, SECTION_FOUR_FFN_RUNS):
        before = load_run(run_mult4)
        after = load_run(run_mult3)
        ax1.plot([4, 3], [before["final_val_loss"], after["final_val_loss"]], marker="o", label=seed_label)
        ax2.plot([4, 3], [before["overfit_score"], after["overfit_score"]], marker="o", label=seed_label)

    ax1.set_title("FFN width vs Final Val Loss", fontsize=11)
    ax1.set_xlabel("ffn_mult")
    ax1.set_xticks([4, 3])
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    ax2.set_title("FFN width vs Overfit Score", fontsize=11)
    ax2.set_xlabel("ffn_mult")
    ax2.set_xticks([4, 3])
    ax2.grid(axis="y", linestyle="--", alpha=0.3)
    ax2.legend(title="seed")

    fig.suptitle("Section 4: effect of shrinking FFN width", fontsize=12, fontweight="bold")
    fig.savefig(VIS_DIR / "section_ffn_mult_3.svg", format="svg")
    fig.savefig(VIS_DIR / "section_ffn_mult_3.png", dpi=180)
    plt.close(fig)


def build_section_five_plot(comp: dict) -> None:
    before = load_run(comp["before_run"])
    after = load_run(comp["after_run"])
    fig, ax = plt.subplots(figsize=(8, 3.8), constrained_layout=True)
    metric_labels = ["Val Loss", "Gen Gap", "Overfit"]
    before_vals = [before["final_val_loss"], before["final_generalization_gap"], before["overfit_score"]]
    after_vals = [after["final_val_loss"], after["final_generalization_gap"], after["overfit_score"]]
    x = range(len(metric_labels))
    width = 0.35
    ax.bar([i - width/2 for i in x], before_vals, width=width, color="#94a3b8", label=comp["before_label"])
    ax.bar([i + width/2 for i in x], after_vals, width=width, color="#16a34a", label=comp["after_label"])
    ax.set_xticks(list(x), metric_labels)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend()
    ax.set_title("Section 5: context shortening changed all three signals", fontsize=12, fontweight="bold")
    for i, (b, a) in enumerate(zip(before_vals, after_vals)):
        ax.text(i - width/2, b, f"{b:.4f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + width/2, a, f"{a:.4f}", ha="center", va="bottom", fontsize=8)
    fig.savefig(VIS_DIR / "section_context_length_48.svg", format="svg")
    fig.savefig(VIS_DIR / "section_context_length_48.png", dpi=180)
    plt.close(fig)


def build_section_six_plot() -> None:
    fig, axes = plt.subplots(ncols=2, figsize=(10, 4), constrained_layout=True)
    ax1, ax2 = axes
    rows = [(stride, load_run(run)) for stride, run in SECTION_SIX_STRIDE_RUNS]
    strides = [s for s, _ in rows]
    val_losses = [r["final_val_loss"] for _, r in rows]
    overfits = [r["overfit_score"] for _, r in rows]

    ax1.plot(strides, val_losses, marker="o", color="#2563eb")
    ax1.set_title("Seed303 rescue: stride vs Final Val Loss", fontsize=11)
    ax1.set_xlabel("stride")
    ax1.grid(axis="both", linestyle="--", alpha=0.3)

    ax2.plot(strides, overfits, marker="o", color="#dc2626")
    ax2.set_title("Seed303 rescue: stride vs Overfit Score", fontsize=11)
    ax2.set_xlabel("stride")
    ax2.grid(axis="both", linestyle="--", alpha=0.3)

    for ax, ys in [(ax1, val_losses), (ax2, overfits)]:
        for x, y in zip(strides, ys):
            ax.text(x, y, f"{y:.4f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Section 6: stride as a rescue knob on a difficult seed", fontsize=12, fontweight="bold")
    fig.savefig(VIS_DIR / "section_stride_rescue.svg", format="svg")
    fig.savefig(VIS_DIR / "section_stride_rescue.png", dpi=180)
    plt.close(fig)


def build_section_seven_plot() -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4), constrained_layout=True)
    runs = [load_run(r) for r in SECTION_SEVEN_MAX_STEPS_RUNS]
    xs = [r["max_steps"] for r in runs]
    ys = [r["final_val_loss"] for r in runs]
    sizes = [80 + 500 * r["overfit_score"] for r in runs]
    colors = ["#dc2626" if r["fit_status"] == "overfit_risk" else "#2563eb" for r in runs]
    ax.scatter(xs, ys, s=sizes, c=colors, alpha=0.8)
    ax.grid(axis="both", linestyle="--", alpha=0.3)
    ax.set_xlabel("max_steps")
    ax.set_ylabel("final_val_loss")
    ax.set_title("Section 7: longer training helps, but seed-dependent overfit grows", fontsize=12, fontweight="bold")
    for r, x, y in zip(runs, xs, ys):
        ax.text(x, y, f"run{r['run_id']}", fontsize=8, ha="left", va="bottom")
    fig.savefig(VIS_DIR / "section_max_steps_105.svg", format="svg")
    fig.savefig(VIS_DIR / "section_max_steps_105.png", dpi=180)
    plt.close(fig)


def build_plot() -> None:
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        nrows=len(COMPARISONS),
        ncols=len(METRICS),
        figsize=(12, 16),
        constrained_layout=True,
    )

    csv_rows: list[dict] = []

    before_color = "#94a3b8"
    after_color = "#2563eb"

    for row_idx, comp in enumerate(COMPARISONS):
        before = load_run(comp["before_run"])
        after = load_run(comp["after_run"])

        for col_idx, (metric_key, metric_title) in enumerate(METRICS):
            ax = axes[row_idx, col_idx]
            before_val = before[metric_key]
            after_val = after[metric_key]
            delta = after_val - before_val

            ax.bar(
                [0, 1],
                [before_val, after_val],
                color=[before_color, after_color],
                width=0.6,
            )
            ax.set_xticks([0, 1], [comp["before_label"], comp["after_label"]], fontsize=8)
            ax.set_title(metric_title, fontsize=10, pad=10)
            ax.axhline(0, color="#cbd5e1", linewidth=0.8)
            ax.grid(axis="y", linestyle="--", alpha=0.3)

            for x, value in enumerate([before_val, after_val]):
                ax.text(
                    x,
                    value,
                    f"{value:.4f}",
                    ha="center",
                    va="bottom" if value >= 0 else "top",
                    fontsize=8,
                )

            if col_idx == 0:
                ax.set_ylabel(f"{comp['title']}\n{comp['note']}", fontsize=9)

            ax.text(
                0.5,
                0.95,
                f"delta={delta:+.4f}",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#cbd5e1"),
            )

            csv_rows.append(
                {
                    "comparison": comp["slug"],
                    "note": comp["note"],
                    "before_run": comp["before_run"],
                    "after_run": comp["after_run"],
                    "metric": metric_key,
                    "before_value": before_val,
                    "after_value": after_val,
                    "delta_after_minus_before": delta,
                }
            )

    fig.suptitle(
        "Mini GPT Key Hypothesis Comparisons\n(Lower is better for val loss, gap, overfit score)",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(VIS_DIR / "result_summary_comparisons.svg", format="svg")
    fig.savefig(VIS_DIR / "result_summary_comparisons.png", dpi=180)
    plt.close(fig)
    save_csv(csv_rows)
    build_section_one_plot()
    build_section_two_plot(COMPARISONS[0])
    build_section_three_plot()
    build_section_four_plot()
    build_section_five_plot(COMPARISONS[2])
    build_section_six_plot()
    build_section_seven_plot()


if __name__ == "__main__":
    build_plot()
