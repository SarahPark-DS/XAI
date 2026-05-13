"""
test.py - Marabou verification demo on EMNIST Letters FC network

Usage:
    python test.py

This script demonstrates:
1. Loading the trained EMNIST Letters model (ONNX format)
2. Running a local adversarial robustness verification query
3. Searching for a counterexample using a decision boundary sample
"""

import numpy as np
import onnxruntime as ort
from torchvision import datasets, transforms
from verify import load_dataset, find_sample, run_verification

ONNX_PATH = "models/emnist_fc.onnx"
EPSILON   = 0.01

def demo_unsat():
    """실험 1: 강건성 증명 (UNSAT)"""
    print("=" * 60)
    print("Demo 1: Robustness Verification (expected: UNSAT)")
    print("=" * 60)

    dataset = load_dataset()

    # 'a' 샘플 검증
    idx, x, label = find_sample(dataset, target_label=0)
    letter = chr(ord('a') + int(label))
    print(f"Sample index : {idx}")
    print(f"True label   : '{letter}'")
    print(f"Epsilon      : {EPSILON}")
    print(f"Query        : Are all inputs within ε={EPSILON} of this sample classified as '{letter}'?")
    print()

    res = run_verification(x, int(label), epsilon=EPSILON)
    print(f"Result  : {res['exitCode'].upper()}")
    print(f"Time    : {res['time']:.2f}s")
    if res['exitCode'] == 'unsat':
        print(f"✅ VERIFIED: All inputs within ε={EPSILON} of sample '{letter}' are classified as '{letter}'.")
    else:
        adv = chr(ord('a') + res['adv_class'])
        print(f"⚠️  SAT: Counterexample found → adversarial class '{adv}'")

def demo_sat():
    """실험 2: 반례 탐색 (SAT) - 결정 경계 근처 샘플"""
    print()
    print("=" * 60)
    print("Demo 2: Counterexample Search (expected: SAT)")
    print("=" * 60)

    dataset = load_dataset()
    sess    = ort.InferenceSession(ONNX_PATH)

    # a↔w 경계 샘플 생성
    label1, label2 = 0, 22  # a, w
    letter1 = chr(ord('a') + label1)
    letter2 = chr(ord('a') + label2)

    image1, _ = dataset[0]
    # w 클래스 첫 번째 샘플 찾기
    for idx in range(len(dataset)):
        _, lbl = dataset[idx]
        if int(lbl) == label2:
            image2, _ = dataset[idx]
            break

    x1 = image1.numpy().flatten()
    x2 = image2.numpy().flatten()

    # 선형 보간으로 경계 샘플 탐색
    x_mid = None
    for alpha in np.arange(0.1, 1.0, 0.1):
        x_blend = ((1 - alpha) * x1 + alpha * x2).astype(np.float32)
        out  = sess.run(None, {"input": x_blend.reshape(1, 1, 28, 28)})[0]
        pred = int(np.argmax(out))
        if pred != label1:
            x_mid = x_blend
            print(f"Decision boundary sample found at alpha={alpha:.1f}")
            print(f"  Original label : '{letter1}'")
            print(f"  Predicted label: '{chr(ord('a') + pred)}'")
            break

    if x_mid is None:
        print("Decision boundary sample not found. Skipping SAT demo.")
        return

    print(f"Epsilon : {EPSILON}")
    print(f"Query   : Does a perturbation within ε={EPSILON} exist that changes the prediction?")
    print()

    res = run_verification(x_mid, true_class=label1, epsilon=EPSILON)
    print(f"Result  : {res['exitCode'].upper()}")
    print(f"Time    : {res['time']:.2f}s")
    if res['exitCode'] == 'sat':
        adv = chr(ord('a') + res['adv_class'])
        print(f"⚠️  SAT: Counterexample found → adversarial class '{adv}'")
        print(f"   This confirms the model is not robust near the decision boundary.")
    else:
        print(f"✅ UNSAT: No counterexample found within ε={EPSILON}.")

if __name__ == "__main__":
    print()
    print("  Marabou Verification Demo")
    print("  Model  : EMNIST Letters FC Network (784→256→128→26)")
    print("  Dataset: EMNIST Letters (a–z, 28x28 grayscale)")
    print()

    demo_unsat()
    demo_sat()

    print()
    print("Done.")
