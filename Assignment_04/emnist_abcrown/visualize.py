"""
visualize.py — Visualization of α,β-Crown verification results.

Generates four figures:
  fig1: Verified/Falsified/Timeout bar chart per letter (α,β-Crown)
  fig2: Verification time comparison scatter (Marabou vs α,β-Crown)
  fig3: Time distribution histograms (both tools side-by-side)
  fig4: Result agreement heatmap between Marabou and α,β-Crown
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Common style ───────────────────────────────────────────────────────
COLORS = {
    "verified" : "#1D9E75",   # green
    "falsified": "#E24B4A",   # red
    "timeout"  : "#F0A500",   # amber
    "marabou"  : "#378ADD",   # blue
    "abcrown"  : "#9B59B6",   # purple
    "agree"    : "#1D9E75",
    "disagree" : "#E24B4A",
}
plt.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "figure.facecolor" : "white",
    "axes.facecolor"   : "white",
})

LETTERS = [chr(ord('a') + i) for i in range(26)]


def _normalize_result(r: str) -> str:
    """Map Marabou or α,β-Crown result to verified/falsified/timeout."""
    r = str(r).lower().strip()
    if r in ("unsat", "verified", "safe"):
        return "verified"
    if r in ("sat", "falsified", "unsafe"):
        return "falsified"
    return "timeout"


# ── Figure 1: SAT/UNSAT/Timeout bar per letter ─────────────────────────
def plot_result_bar(results: list, out_dir: str) -> None:
    df = pd.DataFrame(results)
    df["result_norm"] = df["result"].map(_normalize_result)

    verified_counts  = []
    falsified_counts = []
    timeout_counts   = []

    for letter in LETTERS:
        sub = df[df["label"] == letter]
        verified_counts.append((sub["result_norm"] == "verified").sum())
        falsified_counts.append((sub["result_norm"] == "falsified").sum())
        timeout_counts.append((sub["result_norm"] == "timeout").sum())

    x = np.arange(26)
    width = 0.6
    fig, ax = plt.subplots(figsize=(14, 5))

    b1 = ax.bar(x, verified_counts, width, label="Verified (robust)",
                color=COLORS["verified"], alpha=0.85)
    b2 = ax.bar(x, falsified_counts, width, bottom=verified_counts,
                label="Falsified (counterexample)", color=COLORS["falsified"], alpha=0.85)
    b3 = ax.bar(x, timeout_counts, width,
                bottom=[v + f for v, f in zip(verified_counts, falsified_counts)],
                label="Timeout", color=COLORS["timeout"], alpha=0.85)

    # Label falsified counts > 0
    for i, (v, f) in enumerate(zip(verified_counts, falsified_counts)):
        if f > 0:
            ax.text(i, v + f + 0.1, str(f), ha="center", va="bottom",
                    fontsize=9, color=COLORS["falsified"], fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(LETTERS, fontsize=11)
    ax.set_ylabel("Number of samples", fontsize=11)
    ax.set_title("α,β-Crown Verification Results per Letter  (ε = 0.01, 10 samples each)",
                 fontsize=13, pad=12)
    ax.legend(fontsize=10, loc="upper right")

    plt.tight_layout()
    path = os.path.join(out_dir, "fig1_result_bar.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {path}")


# ── Figure 2: Timing scatter — Marabou vs α,β-Crown ───────────────────
def plot_time_scatter(results: list, marabou_df: pd.DataFrame, out_dir: str) -> None:
    df_abc = pd.DataFrame(results)
    df_abc["result_norm"] = df_abc["result"].map(_normalize_result)
    df_abc["sample_idx"] = df_abc["sample_idx"].astype(str)

    mar = marabou_df.copy()
    mar["sample_idx"] = mar["sample_idx"].astype(str)

    # Merge on sample_idx and label
    merged = pd.merge(
        df_abc,
        mar.rename(columns={"marabou_time_s": "time_marabou",
                             "marabou_result": "result_marabou"}),
        on=["sample_idx", "label"],
        how="inner"
    )
    if merged.empty:
        print("  [fig2] No matched samples — skipping scatter plot")
        return

    merged["result_marabou_norm"] = merged["result_marabou"].map(_normalize_result)

    fig, ax = plt.subplots(figsize=(7, 6))

    for r in ("verified", "falsified", "timeout"):
        sub = merged[merged["result_norm"] == r]
        if not sub.empty:
            ax.scatter(
                sub["time_marabou"], sub["time_s"],
                c=COLORS[r], label=f"α,β-Crown: {r} (n={len(sub)})",
                alpha=0.7, s=40, edgecolors="none"
            )

    # Diagonal y=x reference line (equal speed)
    lim_max = max(merged["time_marabou"].max(), merged["time_s"].max()) * 1.1
    ax.plot([0, lim_max], [0, lim_max], "k--", linewidth=0.8, alpha=0.4, label="Equal speed")
    ax.set_xlim(0, lim_max)
    ax.set_ylim(0, max(merged["time_s"].max() * 1.2, 1))

    ax.set_xlabel("Marabou time (s)", fontsize=11)
    ax.set_ylabel("α,β-Crown time (s)", fontsize=11)
    ax.set_title("Verification Time: Marabou vs α,β-Crown  (per sample)", fontsize=12)
    ax.legend(fontsize=9)

    # Annotate median speedup
    speedup = (merged["time_marabou"] / merged["time_s"].replace(0, np.nan)).dropna()
    if len(speedup) > 0:
        ax.text(0.97, 0.05, f"Median speedup: {speedup.median():.1f}×",
                transform=ax.transAxes, ha="right", fontsize=10,
                color=COLORS["abcrown"], fontweight="bold")

    plt.tight_layout()
    path = os.path.join(out_dir, "fig2_time_comparison_scatter.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {path}")


# ── Figure 3: Time distribution histograms ─────────────────────────────
def plot_time_histogram(results: list, marabou_df: pd.DataFrame, out_dir: str) -> None:
    abc_times = [r["time_s"] for r in results if r["time_s"] > 0 and r["result"] != "timeout"]
    mar_times = marabou_df["marabou_time_s"].dropna().tolist() if marabou_df is not None else []

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # ── Left: α,β-Crown distribution ────────────────────────────────
    df = pd.DataFrame(results)
    df["result_norm"] = df["result"].map(_normalize_result)
    v_times = df[df["result_norm"] == "verified"]["time_s"]
    f_times = df[df["result_norm"] == "falsified"]["time_s"]

    bins = np.linspace(0, max(max(abc_times, default=1), 1) * 1.1, 30)
    axes[0].hist(v_times, bins=bins, color=COLORS["verified"], alpha=0.75,
                 label=f"Verified (n={len(v_times)})", edgecolor="white")
    axes[0].hist(f_times, bins=bins, color=COLORS["falsified"], alpha=0.75,
                 label=f"Falsified (n={len(f_times)})", edgecolor="white")
    axes[0].set_xlabel("Verification time (s)", fontsize=11)
    axes[0].set_ylabel("Number of samples", fontsize=11)
    axes[0].set_title("α,β-Crown Time Distribution", fontsize=12)
    axes[0].legend(fontsize=9)

    # ── Right: Marabou vs α,β-Crown overlay ─────────────────────────
    all_times = abc_times + [t for t in mar_times if t < 30]
    bins2 = np.linspace(0, max(all_times, default=1) * 1.1, 25) if all_times else np.arange(0, 10)
    axes[1].hist(abc_times, bins=bins2, color=COLORS["abcrown"], alpha=0.65,
                 label=f"α,β-Crown (n={len(abc_times)})", edgecolor="white")
    if mar_times:
        mar_clipped = [t for t in mar_times if t <= max(bins2)]
        axes[1].hist(mar_clipped, bins=bins2, color=COLORS["marabou"], alpha=0.65,
                     label=f"Marabou ≤{max(bins2):.0f}s (n={len(mar_clipped)})", edgecolor="white")
    axes[1].set_xlabel("Verification time (s)", fontsize=11)
    axes[1].set_ylabel("Number of samples", fontsize=11)
    axes[1].set_title("Tool Comparison: Time Distribution", fontsize=12)
    axes[1].legend(fontsize=9)

    fig.suptitle("Verification Time Distributions  (ε = 0.01, 260 samples)", fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig3_time_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {path}")


# ── Figure 4: Agreement heatmap ─────────────────────────────────────────
def plot_agreement_heatmap(results: list, marabou_df: pd.DataFrame, out_dir: str) -> None:
    if marabou_df is None or marabou_df.empty:
        print("  [fig4] No Marabou data — skipping agreement heatmap")
        return

    df_abc = pd.DataFrame(results)
    df_abc["result_norm"] = df_abc["result"].map(_normalize_result)
    df_abc["sample_idx"] = df_abc["sample_idx"].astype(str)

    mar = marabou_df.copy()
    mar["sample_idx"] = mar["sample_idx"].astype(str)

    merged = pd.merge(
        df_abc,
        mar.rename(columns={"marabou_result": "result_marabou"}),
        on=["sample_idx", "label"],
        how="inner"
    )
    if merged.empty:
        print("  [fig4] No matched samples — skipping")
        return

    merged["result_marabou_norm"] = merged["result_marabou"].map(_normalize_result)

    cats = ["verified", "falsified", "timeout"]
    matrix = np.zeros((3, 3), dtype=int)
    for i, mar_cat in enumerate(cats):
        for j, abc_cat in enumerate(cats):
            matrix[i, j] = ((merged["result_marabou_norm"] == mar_cat) &
                            (merged["result_norm"] == abc_cat)).sum()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels([f"α,β-Crown\n{c}" for c in cats], fontsize=9)
    ax.set_yticklabels([f"Marabou\n{c}" for c in cats], fontsize=9)
    ax.set_xlabel("α,β-Crown result", fontsize=11)
    ax.set_ylabel("Marabou result", fontsize=11)
    ax.set_title("Result Agreement: Marabou vs α,β-Crown", fontsize=12)

    for i in range(3):
        for j in range(3):
            color = "white" if matrix[i, j] > matrix.max() * 0.5 else "black"
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    fontsize=14, fontweight="bold", color=color)

    plt.colorbar(im, ax=ax, label="Count")
    plt.tight_layout()
    path = os.path.join(out_dir, "fig4_agreement_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {path}")


# ── Public API ──────────────────────────────────────────────────────────
def generate_visualizations(results: list, marabou_df, out_dir: str) -> None:
    """Generate all four figures. marabou_df may be None if unavailable."""
    os.makedirs(out_dir, exist_ok=True)

    print("  [1/4] Result bar chart per letter...")
    plot_result_bar(results, out_dir)

    if marabou_df is not None and not marabou_df.empty:
        print("  [2/4] Time comparison scatter...")
        plot_time_scatter(results, marabou_df, out_dir)
    else:
        print("  [2/4] Skipped (no Marabou data)")

    print("  [3/4] Time distribution histograms...")
    plot_time_histogram(results, marabou_df, out_dir)

    if marabou_df is not None and not marabou_df.empty:
        print("  [4/4] Agreement heatmap...")
        plot_agreement_heatmap(results, marabou_df, out_dir)
    else:
        print("  [4/4] Skipped (no Marabou data)")
