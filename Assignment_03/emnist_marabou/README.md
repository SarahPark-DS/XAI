# EMNIST Letters Verification with Marabou

Neural network robustness verification using [Marabou](https://github.com/NeuralNetworkVerification/Marabou) on the EMNIST Letters dataset.

## Overview

This project trains a small fully connected network on EMNIST Letters (a–z) and verifies local adversarial robustness using Marabou's SMT-based verification engine.

**Verification query**: For an input image `x` classified as letter `c`, prove that all inputs `x'` within an ℓ∞-ball of radius ε are also classified as `c`.

---

## Environment Setup

### 1. Install Marabou

```bash
git clone https://github.com/NeuralNetworkVerification/Marabou.git
cd Marabou

# Install dependencies
sudo apt install gfortran libopenblas-dev

# Build
mkdir build && cd build
cmake .. -DBUILD_PYTHON=ON \
         -DOPENBLAS_DIR=/usr \
         -DCMAKE_BUILD_TYPE=Release
make -j4
```

> **Note**: If you encounter `Illegal instruction` during tests, your CPU may not support AVX-512.
> Rebuild OpenBLAS with `TARGET=HASWELL`:
> ```bash
> cd tools/OpenBLAS-0.3.19
> make TARGET=HASWELL
> make install PREFIX=$(pwd)/../installed/OpenBLAS
> ```

### 2. Set PYTHONPATH

```bash
export PYTHONPATH=/path/to/Marabou:$PYTHONPATH
# Add to ~/.bashrc to persist
echo 'export PYTHONPATH=/path/to/Marabou:$PYTHONPATH' >> ~/.bashrc
```

### 3. Create Python environment

```bash
conda create -n marabou_env python=3.12
conda activate marabou_env
pip install -r requirements.txt
```

---

## Project Structure

```
emnist_marabou/
├── train.py          # Train EMNIST Letters FC network + export to ONNX
├── verify.py         # Marabou verification query (core logic)
├── run_all.py        # Run verification on all 26 letters (260 samples)
├── test.py           # Demo: UNSAT + SAT verification examples
├── requirements.txt  # Python dependencies
├── report.pdf        # Assignment report
├── models/
│   └── emnist_fc.onnx  # Trained model (ONNX opset 11)
├── data/             # EMNIST dataset (auto-downloaded, gitignored)
└── results/
    └── verification_results.csv  # Experiment results
```

---

## Usage

### Step 1: Train the model

```bash
python train.py
```

Trains for 10 epochs and exports `models/emnist_fc.onnx`.  
Expected test accuracy: ~90%.

### Step 2: Run the demo

```bash
python test.py
```

Demonstrates two verification scenarios:
- **Demo 1 (UNSAT)**: Verifies robustness of a normal test sample within ε=0.01
- **Demo 2 (SAT)**: Finds a counterexample near the a/w decision boundary

### Step 3: Run full experiment

```bash
python run_all.py
```

Runs verification on 10 samples per letter (260 total) and saves results to `results/verification_results.csv`.

---

## Model Architecture

```
Input: 28×28 grayscale image → flatten → 784
→ Linear(784, 256) → ReLU
→ Linear(256, 128) → ReLU
→ Linear(128, 26)
→ Output: logits for 26 letters (a–z)
```

---

## Key Results

| Metric | Value |
|--------|-------|
| Test accuracy | 90.64% |
| Total samples verified | 260 (10 per letter) |
| UNSAT (robust) | 230 (88.5%) |
| SAT (counterexample found) | 30 (11.5%) |
| Most vulnerable letter | 'i' (60% SAT) |
| Avg. verification time (UNSAT) | ~4.5s |
| Avg. verification time (SAT) | ~2.5s |

---

## Requirements

See `requirements.txt`. Key dependencies:
- `torch`, `torchvision`
- `onnx`, `onnxruntime`
- `numpy`
- Marabou (installed separately, see above)
