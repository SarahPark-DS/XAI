# visualize.py
# 실험 결과 시각화: SAT/UNSAT 비율, 검증 시간 분포, adversarial 이미지
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import onnxruntime as ort
from torchvision import datasets, transforms
from verify import load_dataset, run_verification

RESULT_DIR = "results"
CSV_PATH   = f"{RESULT_DIR}/verification_results.csv"
ONNX_PATH  = "models/emnist_fc.onnx"
os.makedirs(RESULT_DIR, exist_ok=True)

# ── 공통 스타일 ───────────────────────────────────────────────────────
COLORS = {
    "sat"  : "#E24B4A",
    "unsat": "#1D9E75",
    "time" : "#378ADD",
    "bg"   : "#F8F8F8",
}
plt.rcParams.update({
    "font.family"     : "DejaVu Sans",
    "axes.spines.top" : False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor"  : "white",
})

# ── 데이터 로드 ───────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
exp1 = df[df['experiment'] == 'robustness'].copy()
exp2 = df[df['experiment'] == 'counterexample'].copy()
exp1['result'] = exp1['result'].str.upper()


# ════════════════════════════════════════════════════════════════════
# 그래프 1: 알파벳별 SAT/UNSAT 비율 막대그래프
# ════════════════════════════════════════════════════════════════════
def plot_sat_unsat_bar():
    letters = [chr(ord('a') + i) for i in range(26)]
    sat_counts   = []
    unsat_counts = []

    for l in letters:
        sub = exp1[exp1['label'] == l]
        sat_counts.append((sub['result'] == 'SAT').sum())
        unsat_counts.append((sub['result'] == 'UNSAT').sum())

    x      = np.arange(26)
    width  = 0.6
    fig, ax = plt.subplots(figsize=(14, 5))

    bars_u = ax.bar(x, unsat_counts, width, label='UNSAT (robust)',
                    color=COLORS['unsat'], alpha=0.85)
    bars_s = ax.bar(x, sat_counts, width, bottom=unsat_counts,
                    label='SAT (counterexample)', color=COLORS['sat'], alpha=0.85)

    # SAT 수 레이블
    for i, (s, u) in enumerate(zip(sat_counts, unsat_counts)):
        if s > 0:
            ax.text(i, u + s + 0.1, str(s), ha='center', va='bottom',
                    fontsize=9, color=COLORS['sat'], fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(letters, fontsize=11)
    ax.set_yticks([0, 2, 4, 6, 8, 10])
    ax.set_ylabel("Number of samples", fontsize=11)
    ax.set_title("SAT / UNSAT Verification Results per Letter  (ε = 0.01, 10 samples each)",
                 fontsize=13, pad=12)
    ax.legend(fontsize=10, loc='upper right')

    # 'i' 강조
    ax.get_xticklabels()[8].set_color(COLORS['sat'])
    ax.get_xticklabels()[8].set_fontweight('bold')

    plt.tight_layout()
    path = f"{RESULT_DIR}/fig1_sat_unsat_bar.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"saved → {path}")


# ════════════════════════════════════════════════════════════════════
# 그래프 2: 검증 시간 분포
# ════════════════════════════════════════════════════════════════════
def plot_time_distribution():
    sat_times   = exp1[exp1['result'] == 'SAT'  ]['time_s'].values
    unsat_times = exp1[exp1['result'] == 'UNSAT']['time_s'].values

    # outlier(>30s) 제거해서 분포 잘 보이게
    unsat_main    = unsat_times[unsat_times <= 30]
    unsat_outlier = unsat_times[unsat_times >  30]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # ── 왼쪽: boxplot ─────────────────────────────────────────────
    bp = axes[0].boxplot(
        [sat_times, unsat_main],
        labels=['SAT', 'UNSAT\n(≤30s)'],
        patch_artist=True,
        medianprops=dict(color='white', linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker='o', markersize=5, alpha=0.5),
    )
    bp['boxes'][0].set_facecolor(COLORS['sat'])
    bp['boxes'][1].set_facecolor(COLORS['unsat'])
    bp['boxes'][0].set_alpha(0.8)
    bp['boxes'][1].set_alpha(0.8)

    axes[0].set_ylabel("Verification time (s)", fontsize=11)
    axes[0].set_title("SAT vs UNSAT Verification Time (Boxplot)", fontsize=12)
    if len(unsat_outlier) > 0:
        axes[0].text(0.98, 0.97,
                     f"{len(unsat_outlier)} UNSAT outlier(s)\n(93-114s) excluded",
                     transform=axes[0].transAxes,
                     ha='right', va='top', fontsize=8,
                     color='gray')

    # ── 오른쪽: histogram ────────────────────────────────────────
    bins = np.arange(0, 12, 0.5)
    axes[1].hist(sat_times,   bins=bins, color=COLORS['sat'],
                 alpha=0.7, label=f'SAT (n={len(sat_times)})', edgecolor='white')
    axes[1].hist(unsat_main,  bins=bins, color=COLORS['unsat'],
                 alpha=0.7, label=f'UNSAT ≤30s (n={len(unsat_main)})', edgecolor='white')

    axes[1].set_xlabel("Verification time (s)", fontsize=11)
    axes[1].set_ylabel("Number of samples", fontsize=11)
    axes[1].set_title("Verification Time Distribution (Histogram)", fontsize=12)
    axes[1].legend(fontsize=9)

    fig.suptitle("Marabou Verification Time Distribution  (Experiment 1, ε = 0.01)", fontsize=13, y=1.01)
    plt.tight_layout()
    path = f"{RESULT_DIR}/fig2_time_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"saved → {path}")


# ════════════════════════════════════════════════════════════════════
# 그래프 3: SAT 샘플의 adversarial 이미지 시각화
# ════════════════════════════════════════════════════════════════════
def plot_adversarial_examples(n_examples=6):
    """SAT된 샘플들에 대해 원본 + adversarial + perturbation 시각화"""

    dataset  = load_dataset()
    sess     = ort.InferenceSession(ONNX_PATH)
    sat_rows = exp1[exp1['result'] == 'SAT'].head(n_examples)

    fig, axes = plt.subplots(n_examples, 3,
                             figsize=(9, n_examples * 2.5))
    fig.suptitle(f"Adversarial Examples for SAT Samples  (ε = 0.01, top {n_examples})",
                 fontsize=13, y=1.01)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1736,), (0.3317,))
    ])

    for row_idx, (_, row) in enumerate(sat_rows.iterrows()):
        sample_idx = int(row['sample_idx'])
        true_label = ord(row['label']) - ord('a')
        adv_label  = ord(row['adv_class']) - ord('a') if row['adv_class'] != '-' else None

        image, _ = dataset[sample_idx]
        x_orig   = image.numpy().flatten()

        # Marabou로 adversarial input 재획득
        res = run_verification(x_orig, true_class=true_label, epsilon=0.01)

        if res['exitCode'] != 'sat' or res['adv_input'] is None:
            for c in range(3):
                axes[row_idx][c].axis('off')
            continue

        x_adv = res['adv_input']
        diff  = x_adv - x_orig

        letter_true = chr(ord('a') + true_label)
        letter_adv  = chr(ord('a') + res['adv_class'])

        # 원본
        axes[row_idx][0].imshow(x_orig.reshape(28, 28), cmap='gray', vmin=-1, vmax=3)
        axes[row_idx][0].set_title(f"Original: '{letter_true}'", fontsize=10)
        axes[row_idx][0].axis('off')

        # adversarial
        axes[row_idx][1].imshow(x_adv.reshape(28, 28), cmap='gray', vmin=-1, vmax=3)
        axes[row_idx][1].set_title(f"Adversarial: '{letter_adv}'", fontsize=10)
        axes[row_idx][1].axis('off')

        # perturbation (×10)
        im = axes[row_idx][2].imshow(
            np.abs(diff).reshape(28, 28) * 10,
            cmap='hot', vmin=0, vmax=1
        )
        axes[row_idx][2].set_title(
            f"Perturbation ×10\nmax|δ|={np.max(np.abs(diff)):.4f}",
            fontsize=9
        )
        axes[row_idx][2].axis('off')

    col_labels = ["Original Image", "Adversarial Image", "Perturbation (x10 amplified)"]
    for c, lbl in enumerate(col_labels):
        axes[0][c].set_title(lbl + "\n" + axes[0][c].get_title(), fontsize=10)

    plt.tight_layout()
    path = f"{RESULT_DIR}/fig3_adversarial_examples.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"saved → {path}")


# ════════════════════════════════════════════════════════════════════
# 그래프 4: 알파벳별 평균 검증 시간 heatmap
# ════════════════════════════════════════════════════════════════════
def plot_time_heatmap():
    letters = [chr(ord('a') + i) for i in range(26)]
    avg_times = [exp1[exp1['label'] == l]['time_s'].mean() for l in letters]

    data = np.array(avg_times).reshape(2, 13)
    row_labels = ['a – m', 'n – z']
    col_labels_1 = [chr(ord('a') + i) for i in range(13)]
    col_labels_2 = [chr(ord('a') + i) for i in range(13, 26)]

    fig, axes = plt.subplots(2, 1, figsize=(13, 4))
    fig.suptitle("Average Verification Time per Letter (seconds)", fontsize=13, y=1.02)

    for row in range(2):
        col_labels = col_labels_1 if row == 0 else col_labels_2
        im = axes[row].imshow(
            data[row:row+1], aspect='auto',
            cmap='YlOrRd', vmin=0, vmax=20
        )
        axes[row].set_xticks(range(13))
        axes[row].set_xticklabels(col_labels, fontsize=12)
        axes[row].set_yticks([])

        for col in range(13):
            val = data[row, col]
            color = 'white' if val > 12 else 'black'
            axes[row].text(col, 0, f"{val:.1f}s",
                           ha='center', va='center',
                           fontsize=10, color=color)

    plt.colorbar(im, ax=axes, orientation='vertical',
                 label='Avg. time (s)', shrink=0.8, pad=0.02)
    plt.tight_layout()
    path = f"{RESULT_DIR}/fig4_time_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"saved → {path}")


# ════════════════════════════════════════════════════════════════════
# 실행
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=== Generating visualizations ===\n")

    print("[1/4] SAT/UNSAT bar chart per letter...")
    plot_sat_unsat_bar()

    print("[2/4] Verification time distribution...")
    plot_time_distribution()

    print("[3/4] Adversarial image visualization (top 6 SAT samples)...")
    print("      Note: Marabou will re-run, may take a few minutes.")
    plot_adversarial_examples(n_examples=6)

    print("[4/4] Average verification time heatmap...")
    plot_time_heatmap()

    print("\n=== Done! ===")
    print(f"Saved to: {RESULT_DIR}/")
    print("  fig1_sat_unsat_bar.png")
    print("  fig2_time_distribution.png")
    print("  fig3_adversarial_examples.png")
    print("  fig4_time_heatmap.png")