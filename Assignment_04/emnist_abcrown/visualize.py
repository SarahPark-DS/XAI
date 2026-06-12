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
import matplotlib.font_manager as fm

# NanumSquare 한글 폰트 등록 (없으면 DejaVu Sans 사용)
_KR_FONT = "DejaVu Sans"
for _fp in [
    "/usr/share/fonts/truetype/nanum/NanumSquareR.ttf",
    "/usr/share/fonts/truetype/nanum/NanumSquareRoundR.ttf",
]:
    if os.path.exists(_fp):
        fm.fontManager.addfont(_fp)
        _KR_FONT = fm.FontProperties(fname=_fp).get_name()
        break

plt.rcParams.update({
    "font.family"      : _KR_FONT,
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

    # 0 시간 방지 (log scale용 최솟값 클리핑)
    eps_t = 1e-3
    merged["time_s_safe"] = merged["time_s"].clip(lower=eps_t)
    merged["time_marabou_safe"] = merged["time_marabou"].clip(lower=eps_t)
    speedup = (merged["time_marabou_safe"] / merged["time_s_safe"]).dropna()

    # ── 2-패널 레이아웃 ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Verification Time: Marabou vs α,β-Crown  (per sample, 260 instances)",
                 fontsize=13, y=1.01)

    LABELS = {"verified": "Verified", "falsified": "Falsified", "timeout": "Timeout"}

    # ── 왼쪽: 로그-로그 산점도 ─────────────────────────────────────
    ax = axes[0]
    for r in ("verified", "falsified"):
        sub = merged[merged["result_norm"] == r]
        if not sub.empty:
            ax.scatter(
                sub["time_marabou_safe"], sub["time_s_safe"],
                c=COLORS[r], label=f"{LABELS[r]} (n={len(sub)})",
                alpha=0.75, s=45, edgecolors="white", linewidths=0.4
            )

    # y=x 기준선
    lim_min = min(merged["time_marabou_safe"].min(), merged["time_s_safe"].min()) * 0.5
    lim_max = max(merged["time_marabou_safe"].max(), merged["time_s_safe"].max()) * 2
    ax.plot([lim_min, lim_max], [lim_min, lim_max],
            "k--", linewidth=1.2, alpha=0.5, label="y = x  (동일 속도)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_xlabel("Marabou 검증 시간 (s, 로그 스케일)", fontsize=11)
    ax.set_ylabel("α,β-Crown 검증 시간 (s, 로그 스케일)", fontsize=11)
    ax.set_title("로그-로그 산점도\n(y=x 아래 = α,β-Crown 더 빠름)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", linestyle="--", alpha=0.3)

    if len(speedup) > 0:
        ax.text(0.97, 0.05, f"중앙값 속도 향상: {speedup.median():.0f}×",
                transform=ax.transAxes, ha="right", fontsize=10,
                color=COLORS["abcrown"], fontweight="bold")

    # ── 오른쪽: 속도 향상(speedup) 히스토그램 ─────────────────────
    ax2 = axes[1]
    speedup_log = np.log10(speedup.clip(lower=1))
    bins = np.linspace(0, speedup_log.max() * 1.05, 25)

    for r in ("verified", "falsified"):
        sub = merged[merged["result_norm"] == r]
        if not sub.empty:
            sp = np.log10((sub["time_marabou_safe"] / sub["time_s_safe"]).clip(lower=1))
            ax2.hist(sp, bins=bins, color=COLORS[r], alpha=0.75,
                     label=f"{LABELS[r]} (n={len(sub)})", edgecolor="white")

    # x축 눈금을 실제 배수로 표시
    tick_vals = [1, 10, 100, 1000]
    ax2.set_xticks([np.log10(v) for v in tick_vals])
    ax2.set_xticklabels([f"{v}×" for v in tick_vals], fontsize=10)
    ax2.axvline(np.log10(speedup.median()), color="gray", linestyle="--",
                linewidth=1.2, label=f"중앙값 {speedup.median():.0f}×")

    ax2.set_xlabel("Marabou 대비 속도 향상 배수 (로그 스케일)", fontsize=11)
    ax2.set_ylabel("샘플 수", fontsize=11)
    ax2.set_title("속도 향상 분포\n(α,β-Crown이 Marabou보다 몇 배 빠른가)", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.3)

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
