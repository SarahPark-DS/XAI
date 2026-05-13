# verify.py
import numpy as np
import os
import time
from torchvision import datasets, transforms
from maraboupy import Marabou

ONNX_PATH = "models/emnist_fc.onnx"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1736,), (0.3317,))
])

def load_dataset():
    test_dataset = datasets.EMNIST(
        root="data", split="letters", train=False,
        download=False, transform=transform
    )
    test_dataset.targets -= 1
    return test_dataset

def find_sample(dataset, target_label):
    """특정 label의 첫 번째 샘플 반환"""
    for idx in range(len(dataset)):
        image, label = dataset[idx]
        if label == target_label:
            return idx, image.numpy().flatten(), label
    return None, None, None

def _solve_silent(network):
    """C 레벨 stdout 차단하고 solve 실행"""
    options = Marabou.createOptions(verbosity=0)

    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    old_fd     = os.dup(1)
    os.dup2(devnull_fd, 1)

    try:
        exitCode, vals, stats = network.solve(options=options, verbose=False)
    finally:
        os.dup2(old_fd, 1)
        os.close(old_fd)
        os.close(devnull_fd)

    return exitCode, vals, stats

def run_verification(x, true_class, epsilon=0.01):
    """
    x          : (784,) numpy array, normalized input
    true_class : int, 0–25
    epsilon    : float, ℓ∞ perturbation radius
    return     : dict with exitCode, time, adv_class, adv_input
    """
    elapsed = 0.0

    for j in range(26):
        if j == true_class:
            continue

        # j마다 새로운 network 로드
        network_j    = Marabou.read_onnx(ONNX_PATH)
        inputVars_j  = network_j.inputVars[0].flatten()
        outputVars_j = network_j.outputVars[0].flatten()

        # input constraints (ℓ∞-ball)
        for i, var in enumerate(inputVars_j):
            network_j.setLowerBound(var, float(x[i]) - epsilon)
            network_j.setUpperBound(var, float(x[i]) + epsilon)

        # output constraint: output[j] >= output[true_class]
        # → output[true_class] - output[j] <= 0
        network_j.addInequality(
            [outputVars_j[true_class], outputVars_j[j]],
            [1, -1],
            0
        )

        start = time.time()
        exitCode, vals, _ = _solve_silent(network_j)
        elapsed += time.time() - start

        if exitCode == "sat":
            adv_input  = np.array([vals[v] for v in inputVars_j])
            adv_output = np.array([vals[v] for v in outputVars_j])
            return {
                "exitCode" : "sat",
                "time"     : elapsed,
                "adv_class": int(np.argmax(adv_output)),
                "adv_input": adv_input
            }

    # 모든 j에 대해 UNSAT → property 성립
    return {"exitCode": "unsat", "time": elapsed, "adv_class": None, "adv_input": None}


if __name__ == "__main__":
    dataset = load_dataset()
    idx, x, label = find_sample(dataset, target_label=0)
    print(f"Sample index : {idx}")
    print(f"True label   : {label} ('{chr(ord('a') + int(label))}')")

    print(f"\nRunning Marabou (ε=0.01) ...")
    res = run_verification(x, int(label), epsilon=0.01)

    print(f"Exit code : {res['exitCode']}")
    print(f"Time      : {res['time']:.2f}s")
    if res["exitCode"] == "unsat":
        print(f"Result    : UNSAT ✅")
        print(f"  → All inputs within ε=0.01 of sample '{chr(ord('a') + int(label))}' are verified to predict the same class.")
    elif res["exitCode"] == "sat":
        print(f"Result    : SAT ⚠️  (counterexample found)")
        print(f"  → Adversarial class: {res['adv_class']} ('{chr(ord('a') + res['adv_class'])}')")