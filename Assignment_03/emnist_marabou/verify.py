# verify.py
import numpy as np
import torch
from torchvision import datasets, transforms
from maraboupy import Marabou
import time

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

def run_verification(x, true_class, epsilon=0.01):
    """
    x          : (784,) numpy array, normalized input
    true_class : int, 0–25
    epsilon    : float, ℓ∞ perturbation radius
    return     : dict with exitCode, time, adv_class (if SAT)
    """
    network = Marabou.read_onnx(ONNX_PATH)
    inputVars  = network.inputVars[0].flatten()
    outputVars = network.outputVars[0].flatten()

    # input constraints (ℓ∞-ball)
    for i, var in enumerate(inputVars):
        network.setLowerBound(var, float(x[i]) - epsilon)
        network.setUpperBound(var, float(x[i]) + epsilon)

    # output constraints: output[j] >= output[true_class] for some j != true_class
    for j in range(26):
        if j == true_class:
            continue
        network.addInequality(
            [outputVars[true_class], outputVars[j]],
            [1, -1],
            0
        )

    start = time.time()
    exitCode, vals, stats = network.solve(verbose=False)
    elapsed = time.time() - start

    result = {"exitCode": exitCode, "time": elapsed, "adv_class": None}

    if exitCode == "sat" and vals:
        adv_output = np.array([vals[v] for v in outputVars])
        result["adv_class"] = int(np.argmax(adv_output))

    return result


if __name__ == "__main__":
    # 단독 실행 시 샘플 0번 테스트
    dataset = load_dataset()
    idx, x, label = find_sample(dataset, target_label=0)
    print(f"Sample index : {idx}")
    print(f"True label   : {label} ('{chr(ord('a') + label)}')")

    print(f"\nRunning Marabou (ε=0.01) ...")
    res = run_verification(x, label, epsilon=0.01)

    print(f"Exit code : {res['exitCode']}")
    print(f"Time      : {res['time']:.2f}s")
    if res["exitCode"] == "unsat":
        print(f"Result    : UNSAT ✅")
        print(f"  → All inputs within ε=0.01 of sample '{chr(ord('a') + label)}' are verified to predict the same class.")
    elif res["exitCode"] == "sat":
        print(f"Result    : SAT ⚠️  (counterexample found)")
        print(f"  → Adversarial class: {res['adv_class']} ('{chr(ord('a') + res['adv_class'])}')")