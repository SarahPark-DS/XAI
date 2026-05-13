"""
test.py - Marabou verification demo on EMNIST Letters FC network

Usage:
    python test.py

This script demonstrates:
1. Loading the trained EMNIST Letters model (ONNX format)
2. Running a local adversarial robustness verification query (UNSAT)
3. Finding and visualizing a counterexample near the decision boundary (SAT)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import onnxruntime as ort
from torchvision import datasets, transforms
from verify import load_dataset, find_sample, run_verification

ONNX_PATH  = "models/emnist_fc.onnx"
EPSILON    = 0.01
RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)


def visualize_counterexample(x_original, x_adv, true_label, adv_label,
                              save_path="results/counterexample.png"):
    """원본 이미지와 adversarial 이미지, perturbation을 나란히 시각화"""
    diff = x_adv - x_original

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    # 원본 이미지
    axes[0].imshow(x_original.reshape(28, 28), cmap='gray', vmin=-1, vmax=3)
    axes[0].set_title(f"Original\nlabel: '{chr(ord('a') + true_label)}'", fontsize=12)
    axes[0].axis('off')

    # adversarial 이미지
    axes[1].imshow(x_adv.reshape(28, 28), cmap='gray', vmin=-1, vmax=3)
    axes[1].set_title(f"Adversarial\npredicted: '{chr(ord('a') + adv_label)}'", fontsize=12)
    axes[1].axis('off')

    # perturbation (×10 amplified for visibility)
    im = axes[2].imshow(np.abs(diff).reshape(28, 28) * 10, cmap='hot')
    axes[2].set_title(f"Perturbation (×10 amplified)\nmax |δ|={np.max(np.abs(diff)):.4f}", fontsize=12)
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.suptitle(
        f"Counterexample  |  ε = {EPSILON}  |  L∞ distance = {np.max(np.abs(diff)):.4f}",
        fontsize=11, y=1.02
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Counterexample image saved → {save_path}")


def demo_unsat():
    """Demo 1: 강건성 증명 (UNSAT)"""
    print("=" * 60)
    print("Demo 1: Robustness Verification (expected: UNSAT)")
    print("=" * 60)

    dataset = load_dataset()
    idx, x, label = find_sample(dataset, target_label=0)
    letter = chr(ord('a') + int(label))

    print(f"  Sample index : {idx}")
    print(f"  True label   : '{letter}'")
    print(f"  Epsilon      : {EPSILON}")
    print(f"  Query        : Are all inputs within ε={EPSILON} of this")
    print(f"                 sample classified as '{letter}'?")
    print()

    res = run_verification(x, int(label), epsilon=EPSILON)

    print(f"  Result : {res['exitCode'].upper()}")
    print(f"  Time   : {res['time']:.2f}s")
    if res['exitCode'] == 'unsat':
        print(f"  ✅ VERIFIED: All inputs within ε={EPSILON} of '{letter}' are robust.")
    else:
        adv = chr(ord('a') + res['adv_class'])
        print(f"  ⚠️  SAT: Counterexample found → adversarial class '{adv}'")


def demo_sat():
    """Demo 2: 반례 탐색 및 시각화 (SAT)"""
    print()
    print("=" * 60)
    print("Demo 2: Counterexample Search & Visualization (expected: SAT)")
    print("=" * 60)

    dataset = load_dataset()
    sess    = ort.InferenceSession(ONNX_PATH)

    # a↔w 경계 샘플 생성
    label1, label2 = 0, 22  # a, w
    letter1 = chr(ord('a') + label1)
    letter2 = chr(ord('a') + label2)

    image1, _ = dataset[0]
    x1 = image1.numpy().flatten()

    # w 클래스 첫 번째 샘플 찾기
    x2 = None
    for idx in range(len(dataset)):
        _, lbl = dataset[idx]
        if int(lbl) == label2:
            image2, _ = dataset[idx]
            x2 = image2.numpy().flatten()
            break

    if x2 is None:
        print("  Could not find sample for label 'w'. Skipping.")
        return

    # 선형 보간으로 결정 경계 근처 샘플 탐색
    x_mid = None
    for alpha in np.arange(0.1, 1.0, 0.1):
        x_blend = ((1 - alpha) * x1 + alpha * x2).astype(np.float32)
        out  = sess.run(None, {"input": x_blend.reshape(1, 1, 28, 28)})[0]
        pred = int(np.argmax(out))
        if pred != label1:
            x_mid = x_blend
            print(f"  Decision boundary sample found at alpha={alpha:.1f}")
            print(f"  Original label  : '{letter1}'")
            print(f"  Predicted label : '{chr(ord('a') + pred)}'")
            break

    if x_mid is None:
        print("  Decision boundary sample not found. Skipping SAT demo.")
        return

    print(f"  Epsilon : {EPSILON}")
    print(f"  Query   : Does a perturbation within ε={EPSILON} exist")
    print(f"            that changes the prediction from '{letter1}'?")
    print()

    res = run_verification(x_mid, true_class=label1, epsilon=EPSILON)

    print(f"  Result : {res['exitCode'].upper()}")
    print(f"  Time   : {res['time']:.2f}s")

    if res['exitCode'] == 'sat':
        adv_label = res['adv_class']
        adv_input = res['adv_input']
        adv_letter = chr(ord('a') + adv_label)

        print(f"  ⚠️  SAT: Counterexample found!")
        print(f"       True class       : '{letter1}'")
        print(f"       Adversarial class: '{adv_letter}'")
        print(f"       Max perturbation : {np.max(np.abs(adv_input - x_mid)):.6f} (≤ ε={EPSILON})")

        # 시각화
        visualize_counterexample(
            x_original=x_mid,
            x_adv=adv_input,
            true_label=label1,
            adv_label=adv_label,
            save_path=f"{RESULT_DIR}/counterexample_{letter1}_vs_{adv_letter}.png"
        )
    else:
        print(f"  ✅ UNSAT: No counterexample found within ε={EPSILON}.")


if __name__ == "__main__":
    print()
    print("  Marabou Verification Demo")
    print("  Model  : EMNIST Letters FC Network (784→256→128→26)")
    print("  Dataset: EMNIST Letters (a–z, 28×28 grayscale)")
    print()

    demo_unsat()
    demo_sat()

    print()
    print("Done.")