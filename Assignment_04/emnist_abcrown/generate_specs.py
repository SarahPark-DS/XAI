"""
generate_specs.py — VNNlib specification generator for EMNIST Letters verification.

Generates one .vnnlib file per sample encoding the L∞ robustness property:
  - Input: L∞ ball of radius epsilon around the normalized pixel values
  - Output property: no adversarial class can outscore the true class
    (encoded as disjunction, so unsat = robust, sat = counterexample found)
"""
import os
import numpy as np


N_INPUT = 784    # 28x28 flattened
N_CLASSES = 26   # EMNIST Letters a-z


def write_vnnlib(x: np.ndarray, true_label: int, epsilon: float, filepath: str) -> None:
    """Write a single VNNlib specification file.

    Args:
        x: (784,) float32 array of normalized pixel values
        true_label: ground-truth class index (0-25)
        epsilon: L∞ perturbation radius
        filepath: output .vnnlib path
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    lines = []

    # Input variable declarations (X_0 .. X_783)
    for i in range(N_INPUT):
        lines.append(f"(declare-const X_{i} Real)")
    lines.append("")

    # Output variable declarations (Y_0 .. Y_25)
    for j in range(N_CLASSES):
        lines.append(f"(declare-const Y_{j} Real)")
    lines.append("")

    # Input bounds: L∞ ball [x_i - eps, x_i + eps]
    for i in range(N_INPUT):
        lb = float(x[i]) - epsilon
        ub = float(x[i]) + epsilon
        lines.append(f"(assert (>= X_{i} {lb:.8f}))")
        lines.append(f"(assert (<= X_{i} {ub:.8f}))")
    lines.append("")

    # Output property: negation of robustness (DNF format required by abcrown)
    # Each clause (and (>= Y_j Y_true)) represents "class j beats true class"
    # (assert (or (and ...) (and ...) ...))
    # unsat = robust (no counterexample exists)
    # sat   = falsified (counterexample found)
    lines.append("; Property: exists adversarial class that beats true class (DNF)")
    lines.append("; unsat -> robust (verified), sat -> not robust (falsified)")
    lines.append("(assert (or")
    for j in range(N_CLASSES):
        if j != true_label:
            lines.append(f"    (and (>= Y_{j} Y_{true_label}))")
    lines.append("))")

    with open(filepath, "w") as f:
        f.write("\n".join(lines) + "\n")


def generate_all_specs(samples: list, specs_dir: str, epsilon: float) -> list:
    """Generate VNNlib specs for all samples.

    Args:
        samples: list of dicts with keys: idx, x (784,), label (0-25), letter
        specs_dir: directory to write .vnnlib files
        epsilon: L∞ perturbation radius

    Returns:
        List of spec filepaths (same order as samples)
    """
    os.makedirs(specs_dir, exist_ok=True)
    spec_paths = []
    for s in samples:
        fname = f"sample_{s['idx']:04d}_{s['letter']}.vnnlib"
        fpath = os.path.join(specs_dir, fname)
        write_vnnlib(s["x"], s["label"], epsilon, fpath)
        spec_paths.append(fpath)
    return spec_paths


def write_instances_csv(samples: list, spec_paths: list, onnx_path: str,
                        timeout: int, csv_path: str) -> None:
    """Write VNN-COMP format instances.csv.

    Each line: onnx_model_path,vnnlib_spec_path,timeout_seconds
    Paths are written as-is (absolute or relative to root_path).
    """
    with open(csv_path, "w") as f:
        for s, sp in zip(samples, spec_paths):
            f.write(f"{onnx_path},{sp},{timeout}\n")
