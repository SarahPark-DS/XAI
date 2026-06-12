# EMNIST Letters Verification with α,β-Crown

Neural network robustness verification using **α,β-Crown** (alpha-beta-CROWN) on the EMNIST Letters fully-connected network. This is Assignment 4, a companion to Assignment 3 (Marabou verification), enabling direct tool comparison.

## Overview

- **Model**: 784 → 256 → 128 → 26 FC network (EMNIST Letters, a–z)  
- **Property**: Local L∞ robustness with ε = 0.01 (same as Assignment 3)  
- **Verifier**: α,β-Crown — bound propagation + branch-and-bound (BaB)  
- **Comparison**: Results are matched against Marabou (SMT-based) from Assignment 3

## Directory Structure

```
Assignment_04/
├── alpha-beta-CROWN/       # Cloned α,β-Crown repository
└── emnist_abcrown/
    ├── models/
    │   └── emnist_fc.onnx          # FC network (copied from Assignment 3)
    ├── specs/                      # Auto-generated VNNlib spec files (260 total)
    ├── results/
    │   ├── verification_results.csv
    │   ├── comparison_results.csv
    │   ├── fig1_result_bar.png
    │   ├── fig2_time_comparison_scatter.png
    │   ├── fig3_time_distribution.png
    │   └── fig4_agreement_heatmap.png
    ├── instances.csv               # VNN-COMP format instances list
    ├── abcrown_config.yaml         # α,β-Crown configuration
    ├── test.py                     # Main verification pipeline
    ├── generate_specs.py           # VNNlib spec generator
    ├── parse_results.py            # abcrown output parser
    ├── visualize.py                # Visualization module
    ├── environment.yml             # Conda environment spec
    └── requirements.txt            # pip dependencies
```

## Environment Setup

### 1. Clone α,β-Crown

```bash
cd Assignment_04
git clone https://github.com/Verified-Intelligence/alpha-beta-CROWN.git
cd alpha-beta-CROWN
git submodule update --init --recursive
```

### 2. Create conda environment

```bash
conda create -n abcrown python=3.11 -y
conda activate abcrown

# Install PyTorch (adjust for your CUDA version)
# For CUDA 12.1 driver: (recommended for GPU)
pip install torch==2.5.1+cu121 torchvision --index-url https://download.pytorch.org/whl/cu121

# OR default (CPU or latest CUDA):
pip install torch==2.11.0 torchvision

# Install α,β-Crown dependencies
pip install -e alpha-beta-CROWN/auto_LiRPA
pip install -r emnist_abcrown/requirements.txt
```

### 3. Verify installation

```bash
conda activate abcrown
python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
```

## Usage

### Run verification (full pipeline)

```bash
conda activate abcrown
cd emnist_abcrown
python test.py
```

Options:
```
--samples-per-class N   Number of samples per letter class (default: 10)
--epsilon E             L∞ perturbation radius (default: 0.01)
--timeout T             Per-instance timeout in seconds (default: 30)
```

### Quick smoke test (1 sample per class, 26 total)

```bash
python test.py --samples-per-class 1
```

## Model Architecture

```
Input (28×28 image)
  └→ Flatten → (784,)
  └→ Linear(784, 256) + ReLU
  └→ Linear(256, 128) + ReLU
  └→ Linear(128, 26)   [logits for a–z]
```

- Training: EMNIST Letters test set, 10 epochs, Adam (lr=1e-3)
- Test accuracy: ~90.64%
- Format: ONNX opset 11

## Verification Protocol

### VNNlib Specification

Each sample generates a `.vnnlib` file encoding the robustness property:
- **Input bounds**: L∞ ball `[x_i − ε, x_i + ε]` around each normalized pixel
- **Output property** (negated): `OR over all j ≠ true_class of (Y_j ≥ Y_true_class)`
  - `unsat` → verified: property holds, model is robust
  - `sat` → falsified: counterexample found, model is not robust
  - `timeout` → undecided within time limit

### α,β-Crown Configuration

Key YAML parameters (`abcrown_config.yaml`):
- `solver.bound_prop_method: crown` — CROWN-based bound propagation
- `bab.branching.method: kfsb` — k-FSB branching (optimal for FC networks)
- `bab.timeout: 30` — 30-second per-instance limit
- `attack.enabled: true` — PGD pre-attack for fast SAT detection

### Comparison with Marabou (Assignment 3)

| Aspect | Marabou (Assignment 3) | α,β-Crown (Assignment 4) |
|--------|----------------------|-------------------------|
| Method | SMT / LP / ILP | Bound propagation + BaB |
| SAT detection | SMT constraint solving | PGD attack + BaB |
| UNSAT proof | SMT refutation | Linear relaxation bound |
| Per-query overhead | 25 sub-queries (class pairs) | 1 monolithic query (disjunction) |
| GPU support | No | Yes (CUDA) |

## Results Summary

See `results/verification_results.csv` after running `test.py`.

**Figures**:
- `fig1_result_bar.png` — Verified/Falsified/Timeout counts per letter
- `fig2_time_comparison_scatter.png` — Speed comparison: Marabou vs α,β-Crown
- `fig3_time_distribution.png` — Time distribution histograms
- `fig4_agreement_heatmap.png` — Result agreement matrix between tools
