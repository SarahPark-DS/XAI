# run_all.py
import csv
import os
import numpy as np
import onnxruntime as ort
from collections import defaultdict
from verify import load_dataset, run_verification

EPSILON           = 0.01
SAMPLES_PER_CLASS = 10
OUTPUT_CSV        = "results/verification_results.csv"
ONNX_PATH         = "models/emnist_fc.onnx"

os.makedirs("results", exist_ok=True)

dataset = load_dataset()
sess    = ort.InferenceSession(ONNX_PATH)

# ── 알파벳당 10개 샘플 수집 ───────────────────────────────────────────
class_samples = defaultdict(list)
for idx in range(len(dataset)):
    _, label = dataset[idx]
    label = int(label)
    if len(class_samples[label]) < SAMPLES_PER_CLASS:
        class_samples[label].append(idx)
# break 없이 전체 순회 → 모든 클래스 채움

rows = []

# ── 실험 1: UNSAT (robustness 증명) ──────────────────────────────────
print("=" * 60)
print("실험 1: Robustness verification (ε=0.01)")
print("=" * 60)
print(f"{'Label':<8} {'Idx':<8} {'Result':<10} {'Time(s)'}")
print("-" * 40)

for label in range(26):
    letter = chr(ord('a') + label)
    for idx in class_samples[label]:
        image, _ = dataset[idx]
        x = image.numpy().flatten()

        res     = run_verification(x, true_class=label, epsilon=EPSILON)
        status  = res['exitCode'].upper()
        adv_str = chr(ord('a') + res['adv_class']) if res['adv_class'] is not None else "-"

        print(f"{letter:<8} {idx:<8} {status:<10} {res['time']:.2f}s")
        rows.append({
            "experiment": "robustness",
            "sample_idx": idx,
            "label"     : letter,
            "epsilon"   : EPSILON,
            "result"    : status,
            "time_s"    : round(res['time'], 3),
            "adv_class" : adv_str
        })

# ── 실험 2: SAT (counterexample 시연) ────────────────────────────────
print("\n" + "=" * 60)
print("실험 2: Counterexample search (decision boundary samples)")
print("=" * 60)
print(f"{'Pair':<12} {'Result':<10} {'Adv':<8} {'Time(s)'}")
print("-" * 40)

confusable_pairs = [
    (0, 22),   # a ↔ w
    (8, 11),   # i ↔ l
    (14, 16),  # o ↔ q
    (2, 4),    # c ↔ e
    (13, 8),   # n ↔ i
    (20, 21),  # u ↔ v
]

for (label1, label2) in confusable_pairs:
    if not class_samples[label1] or not class_samples[label2]:
        print(f"{chr(ord('a')+label1)}↔{chr(ord('a')+label2)}: samples not found, skipping")
        continue

    idx1 = class_samples[label1][0]
    idx2 = class_samples[label2][0]
    image1, _ = dataset[idx1]
    image2, _ = dataset[idx2]
    x1 = image1.numpy().flatten()
    x2 = image2.numpy().flatten()

    letter1 = chr(ord('a') + label1)
    letter2 = chr(ord('a') + label2)

    # alpha 0.1씩 올리면서 prediction 바뀌는 지점 찾기
    x_mid     = None
    for alpha in np.arange(0.1, 1.0, 0.1):
        x_blend = ((1 - alpha) * x1 + alpha * x2).astype(np.float32)
        out  = sess.run(None, {"input": x_blend.reshape(1, 1, 28, 28)})[0]
        pred = int(np.argmax(out))
        if pred != label1:
            x_mid = x_blend
            break

    if x_mid is None:
        print(f"{letter1}↔{letter2:<9}  boundary not found, skipping")
        continue

    res     = run_verification(x_mid, true_class=label1, epsilon=EPSILON)
    status  = res['exitCode'].upper()
    adv_str = chr(ord('a') + res['adv_class']) if res['adv_class'] is not None else "-"

    print(f"{letter1}↔{letter2:<9}  {status:<10} {adv_str:<8} {res['time']:.2f}s")
    rows.append({
        "experiment": "counterexample",
        "sample_idx": f"{idx1}-{idx2}-mid",
        "label"     : letter1,
        "epsilon"   : EPSILON,
        "result"    : status,
        "time_s"    : round(res['time'], 3),
        "adv_class" : adv_str
    })

# ── CSV 저장 ──────────────────────────────────────────────────────────
if rows:
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nDone! Results saved → {OUTPUT_CSV}")
else:
    print("No results to save.")