"""
test.py — α,β-Crown Verification of EMNIST Letters FC Network
=============================================================
Verifies local L∞ robustness (ε=0.01) of the same model used in Assignment 3
(Marabou-based verification) using α,β-Crown bound propagation + BaB.

Pipeline:
  1. Load EMNIST Letters test set (same 260 samples as Assignment 3)
  2. Generate VNNlib specification files (one per sample)
  3. Write VNN-COMP instances.csv
  4. Run α,β-Crown via subprocess (single batch call for efficiency)
  5. Parse per-sample results: verified / falsified / timeout
  6. Compare with Marabou results from Assignment 3
  7. Save CSVs and generate visualizations

Usage:
  conda activate abcrown
  cd emnist_abcrown
  python test.py [--samples-per-class N] [--epsilon E] [--timeout T]
"""
import os
import sys
import re
import csv
import time
import argparse
import subprocess
from collections import defaultdict

import numpy as np
import pandas as pd
from torchvision import datasets, transforms

# Local modules (in same directory)
from generate_specs import generate_all_specs, write_instances_csv
from parse_results import parse_abcrown_output
from visualize import generate_visualizations

# ── Paths ──────────────────────────────────────────────────────────────
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(THIS_DIR, "..", "alpha-beta-CROWN")
ABCROWN_SCRIPT = os.path.join(REPO_ROOT, "complete_verifier", "abcrown.py")
ASSIGNMENT3_DIR = os.path.join(THIS_DIR, "..", "..", "Assignment_03", "emnist_marabou")
DATA_DIR = os.path.join(ASSIGNMENT3_DIR, "data")
MARABOU_CSV = os.path.join(ASSIGNMENT3_DIR, "results", "verification_results.csv")

ONNX_RELPATH = "models/emnist_fc.onnx"   # relative to THIS_DIR (emnist_abcrown/)
SPECS_DIR = os.path.join(THIS_DIR, "specs")
RESULTS_DIR = os.path.join(THIS_DIR, "results")
INSTANCES_CSV = os.path.join(THIS_DIR, "instances.csv")
CONFIG_YAML = os.path.join(THIS_DIR, "abcrown_config.yaml")
RESULTS_CSV = os.path.join(RESULTS_DIR, "verification_results.csv")
COMPARISON_CSV = os.path.join(RESULTS_DIR, "comparison_results.csv")

# EMNIST Letters normalization (from Assignment 3)
MEAN = 0.1736
STD = 0.3317

# ── Helpers ────────────────────────────────────────────────────────────

def load_emnist_samples(samples_per_class: int) -> list:
    """Load EMNIST Letters test set and collect samples_per_class per class.

    Returns list of dicts: {idx, x (784,), label (0-25), letter}
    Reuses the data directory from Assignment 3 to avoid re-downloading.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MEAN,), (STD,))
    ])
    dataset = datasets.EMNIST(
        root=DATA_DIR, split="letters", train=False,
        download=False, transform=transform
    )
    # EMNIST Letters labels are 1–26; convert to 0–25
    dataset.targets -= 1

    class_samples = defaultdict(list)
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        label = int(label)
        if len(class_samples[label]) < samples_per_class:
            class_samples[label].append(idx)
        if all(len(v) >= samples_per_class for v in class_samples.values()) \
                and len(class_samples) == 26:
            break

    samples = []
    for label in range(26):
        letter = chr(ord('a') + label)
        for idx in class_samples[label]:
            image, _ = dataset[idx]
            x = image.numpy().flatten().astype(np.float32)
            samples.append({"idx": idx, "x": x, "label": label, "letter": letter})
    return samples


def run_abcrown_batch(n_instances: int, timeout_per_instance: int) -> str:
    """Run α,β-Crown on all instances in instances.csv and return captured stdout."""
    cmd = [
        "conda", "run", "--no-capture-output", "-n", "abcrown",
        "python", ABCROWN_SCRIPT,
        "--config", CONFIG_YAML,
        "--root_path", THIS_DIR,
        "--csv_name", "instances.csv",
        "--start", "0",
        "--end", str(n_instances),
    ]
    print(f"\nRunning α,β-Crown on {n_instances} instances...")
    print(f"Command: {' '.join(cmd)}\n")
    t0 = time.time()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=n_instances * timeout_per_instance * 2 + 300,  # generous overall timeout
        cwd=THIS_DIR,
    )
    total_time = time.time() - t0
    print(f"\nα,β-Crown batch finished in {total_time:.1f}s")
    if proc.returncode != 0:
        print("[WARNING] abcrown returned non-zero exit code:", proc.returncode)
        print("STDERR:", proc.stderr[-2000:] if proc.stderr else "(empty)")
    return proc.stdout, total_time


def parse_batch_output(stdout: str, samples: list) -> list:
    """Parse batch abcrown output into per-sample result dicts.

    abcrown prints per-instance headers and results:
        %%% idx: N, vnnlib ID: M %%%
        ...
        Result: safe in X.XXXX seconds        (→ verified)
        Result: unsafe in X.XXXX seconds      (→ falsified)
        Result: unknown in X.XXXX seconds     (→ timeout)
    """
    # Split stdout into blocks, one per instance
    block_pattern = re.compile(r'%+\s*idx:\s*(\d+),\s*vnnlib ID:\s*(\d+)\s*%+')
    result_pattern = re.compile(r'Result:\s*(\w+)\s+in\s+([0-9.]+)\s+seconds', re.IGNORECASE)
    result_pattern_simple = re.compile(r'Result:\s*(\w+)', re.IGNORECASE)

    # Find all instance header positions
    headers = [(m.start(), int(m.group(1))) for m in block_pattern.finditer(stdout)]

    # Split into per-instance blocks
    blocks = []
    for i, (pos, idx) in enumerate(headers):
        end = headers[i + 1][0] if i + 1 < len(headers) else len(stdout)
        blocks.append((idx, stdout[pos:end]))

    results = []
    for i, sample in enumerate(samples):
        result = "timeout"
        time_s = 0.0

        # Find block for this instance (idx == i)
        block_text = ""
        for block_idx, text in blocks:
            if block_idx == i:
                block_text = text
                break

        if block_text:
            # Try to parse "Result: XXX in Y.YYYY seconds"
            m = result_pattern.search(block_text)
            if m:
                token = m.group(1).lower()
                time_s = float(m.group(2))
                if token in ("safe", "unsat"):
                    result = "verified"
                elif "unsafe" in token or token in ("sat", "falsified"):
                    result = "falsified"
                elif "unknown" in token or "timeout" in token:
                    result = "timeout"
            else:
                # Fallback: simpler pattern
                m2 = result_pattern_simple.search(block_text)
                if m2:
                    token = m2.group(1).lower()
                    if token in ("safe", "unsat"):
                        result = "verified"
                    elif "unsafe" in token or token in ("sat", "falsified"):
                        result = "falsified"

        results.append({
            "experiment": "robustness",
            "sample_idx": sample["idx"],
            "label": sample["letter"],
            "epsilon": 0.01,  # filled in after
            "result": result,
            "time_s": round(time_s, 4),
        })
    return results


def load_marabou_results() -> pd.DataFrame:
    """Load Assignment 3 Marabou results for comparison."""
    if not os.path.exists(MARABOU_CSV):
        print(f"[WARNING] Marabou results not found at {MARABOU_CSV}")
        return pd.DataFrame()
    df = pd.read_csv(MARABOU_CSV)
    df = df[df["experiment"] == "robustness"].copy()
    # Normalize Marabou result labels (UNSAT→verified, SAT→falsified)
    df["marabou_result"] = df["result"].map(
        lambda r: "verified" if str(r).upper() == "UNSAT"
        else ("falsified" if str(r).upper() == "SAT" else "timeout")
    )
    df = df.rename(columns={"time_s": "marabou_time_s"})
    return df[["sample_idx", "label", "marabou_result", "marabou_time_s"]]


def print_summary(results: list, marabou_df: pd.DataFrame) -> None:
    """Print summary table to console."""
    n = len(results)
    n_verified = sum(1 for r in results if r["result"] == "verified")
    n_falsified = sum(1 for r in results if r["result"] == "falsified")
    n_timeout = sum(1 for r in results if r["result"] == "timeout")
    times = [r["time_s"] for r in results if r["time_s"] > 0]
    avg_time = sum(times) / len(times) if times else 0

    print("\n" + "=" * 60)
    print("α,β-Crown Verification Summary")
    print("=" * 60)
    print(f"Total instances   : {n}")
    print(f"Verified (robust) : {n_verified}  ({100*n_verified/n:.1f}%)")
    print(f"Falsified         : {n_falsified}  ({100*n_falsified/n:.1f}%)")
    print(f"Timeout           : {n_timeout}  ({100*n_timeout/n:.1f}%)")
    print(f"Avg time per inst : {avg_time:.2f}s")

    if not marabou_df.empty:
        df = pd.DataFrame(results)
        df["sample_idx"] = df["sample_idx"].astype(str)
        marabou_df = marabou_df.copy()
        marabou_df["sample_idx"] = marabou_df["sample_idx"].astype(str)
        merged = pd.merge(df, marabou_df, on=["sample_idx", "label"], how="inner")
        agree = (merged["result"] == merged["marabou_result"]).sum()
        total = len(merged)
        speedup_vals = merged.apply(
            lambda row: row["marabou_time_s"] / row["time_s"]
            if row["time_s"] > 0 else float("nan"),
            axis=1
        ).dropna()
        print(f"\nComparison with Marabou ({total} matched samples):")
        print(f"  Agreement rate    : {100*agree/total:.1f}%  ({agree}/{total})")
        if len(speedup_vals) > 0:
            print(f"  Median speedup    : {speedup_vals.median():.1f}x  "
                  f"(α,β-Crown vs Marabou)")


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="EMNIST α,β-Crown Verification")
    parser.add_argument("--samples-per-class", type=int, default=10,
                        help="Samples per letter class (default: 10)")
    parser.add_argument("--epsilon", type=float, default=0.01,
                        help="L∞ perturbation radius (default: 0.01)")
    parser.add_argument("--timeout", type=int, default=30,
                        help="Per-instance timeout seconds (default: 30)")
    args = parser.parse_args()

    os.makedirs(SPECS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Step 1: Load EMNIST samples ────────────────────────────────────
    print(f"Loading EMNIST Letters (test split, {args.samples_per_class}/class)...")
    samples = load_emnist_samples(args.samples_per_class)
    print(f"  Loaded {len(samples)} samples across 26 classes")

    # ── Step 2: Generate VNNlib specs ──────────────────────────────────
    print(f"\nGenerating VNNlib specifications (ε={args.epsilon})...")
    spec_paths = generate_all_specs(samples, SPECS_DIR, args.epsilon)
    spec_relpaths = [os.path.relpath(p, THIS_DIR) for p in spec_paths]
    print(f"  Generated {len(spec_paths)} .vnnlib files in {SPECS_DIR}/")

    # ── Step 3: Write instances.csv ────────────────────────────────────
    write_instances_csv(samples, spec_relpaths, ONNX_RELPATH, args.timeout, INSTANCES_CSV)
    print(f"  Written instances.csv ({len(samples)} entries)")

    # ── Step 4: Run α,β-Crown ─────────────────────────────────────────
    stdout, total_wall_time = run_abcrown_batch(len(samples), args.timeout)

    # Save raw output for debugging
    raw_log = os.path.join(RESULTS_DIR, "abcrown_raw.log")
    with open(raw_log, "w") as f:
        f.write(stdout)
    print(f"  Raw output saved to {raw_log}")

    # ── Step 5: Parse results ──────────────────────────────────────────
    results = parse_batch_output(stdout, samples)
    # Fill epsilon value
    for r in results:
        r["epsilon"] = args.epsilon

    # ── Step 6: Save verification_results.csv ─────────────────────────
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {RESULTS_CSV}")

    # ── Step 7: Load Marabou results and merge ─────────────────────────
    marabou_df = load_marabou_results()
    if not marabou_df.empty:
        df_abcrown = pd.DataFrame(results)
        # Ensure consistent types for merge key
        df_abcrown["sample_idx"] = df_abcrown["sample_idx"].astype(str)
        marabou_df["sample_idx"] = marabou_df["sample_idx"].astype(str)
        comparison = pd.merge(df_abcrown, marabou_df, on=["sample_idx", "label"], how="outer")
        comparison.to_csv(COMPARISON_CSV, index=False)
        print(f"Comparison saved to {COMPARISON_CSV}")

    # ── Step 8: Print summary ──────────────────────────────────────────
    print_summary(results, marabou_df)

    # ── Step 9: Generate visualizations ───────────────────────────────
    print("\nGenerating visualizations...")
    marabou_df_for_plot = marabou_df if not marabou_df.empty else None
    generate_visualizations(results, marabou_df_for_plot, RESULTS_DIR)
    print("Done. All outputs in results/")


if __name__ == "__main__":
    main()
