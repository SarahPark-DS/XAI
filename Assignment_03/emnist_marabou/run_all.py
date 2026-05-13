import csv
from verify import load_dataset, find_sample, run_verification

EPSILON = 0.01
OUTPUT_CSV = "results/verification_results.csv"

import os
os.makedirs("results", exist_ok=True)

dataset = load_dataset()

rows = []
print(f"{'Label':<8} {'Idx':<8} {'Result':<10} {'Time(s)':<10} {'Adv class'}")
print("-" * 50)

for target_label in range(26):
    letter = chr(ord('a') + target_label)
    idx, x, label = find_sample(dataset, target_label)

    if x is None:
        print(f"{letter:<8} {'N/A':<8} {'NOT FOUND':<10}")
        continue

    res = run_verification(x, true_class=target_label, epsilon=EPSILON)

    adv_str = chr(ord('a') + res['adv_class']) if res['adv_class'] is not None else "-"
    status  = res['exitCode'].upper()
    print(f"{letter:<8} {idx:<8} {status:<10} {res['time']:.2f}s      {adv_str}")

    rows.append({
        "label"    : letter,
        "sample_idx": idx,
        "epsilon"  : EPSILON,
        "result"   : status,
        "time_s"   : round(res['time'], 3),
        "adv_class": adv_str
    })

# CSV 저장
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"\nResults saved → {OUTPUT_CSV}")