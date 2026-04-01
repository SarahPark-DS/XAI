import os
import glob
import csv

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

from Model.mnist_cnn import MNIST_CNN
from Model.resnet18 import ResNet18
from Attack.fgsm import fgsm_targeted, fgsm_untargeted
from Attack.pgd import pgd_targeted, pgd_untargeted
from Model.utils import train_model, evaluate_model

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULT_DIR   = "./results"
PGD_K        = 40
EPS_STEP     = 0.01
EPSILON_LIST = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
TARGET_CLASS = 8       # 실험 A: epsilon별 실험의 고정 target
VIS_EPS      = 0.3     # 실험 C / 시각화용 고정 eps

# CSV 컬럼 정의 (통합)
# dataset  : MNIST / CIFAR-10
# attack   : FGSM_Targeted / FGSM_Untargeted / PGD_Targeted / PGD_Untargeted
# exp_type : A (epsilon별, target 고정) / B (epsilon별, untargeted) / C (class별, eps 고정)
# eps      : perturbation magnitude
# target_class : targeted이면 0~9, untargeted이면 비워둠
# asr      : attack success rate
# n_samples: 유효 샘플 수
CSV_FIELDS = ["exp_type", "dataset", "attack", "eps", "target_class", "asr", "n_samples"]
CSV_FNAME  = "results.csv"

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

os.makedirs(RESULT_DIR, exist_ok=True)


# ──────────────────────────────────────────────
# CSV 저장 헬퍼
# ──────────────────────────────────────────────
def init_csv():
    """results.csv 초기화 (헤더 작성)"""
    path = os.path.join(RESULT_DIR, CSV_FNAME)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()


def append_csv(rows):
    """rows: list of dict → results.csv에 행 추가"""
    path = os.path.join(RESULT_DIR, CSV_FNAME)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writerows(rows)


def make_row(exp_type, dataset_name, attack_name, eps,
             asr, n, target_class=""):
    return {
        "exp_type":     exp_type,
        "dataset":      dataset_name,
        "attack":       attack_name,
        "eps":          eps,
        "target_class": target_class,   # untargeted는 빈 문자열
        "asr":          round(asr, 6),
        "n_samples":    n,
    }


# ──────────────────────────────────────────────
# 1. 범용 ASR 평가 함수
# ───────────────────────────────── ────────────
def evaluate_asr(model, loader, attack_fn, attack_kwargs,
                 targeted, target_class=None,
                 device=DEVICE, num_batches=float('inf')):
    """
    범용 ASR 평가 함수.
    targeted=True  → 성공 기준: adv_pred == target_class
    targeted=False → 성공 기준: adv_pred != true_label
    """
    model.eval()
    total_valid    = 0
    attack_success = 0

    for i, (images, labels) in enumerate(loader):
        if i >= num_batches:
            break

        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            clean_preds = model(images).argmax(dim=1)

        valid_mask = (clean_preds == labels)
        if targeted:
            valid_mask &= (labels != target_class)

        if valid_mask.sum() == 0:
            continue

        valid_images = images[valid_mask]
        valid_labels = labels[valid_mask]

        if targeted:
            targets = torch.full_like(valid_labels, target_class)
            x_adv = attack_fn(model, valid_images, targets, **attack_kwargs)
        else:
            x_adv = attack_fn(model, valid_images, valid_labels, **attack_kwargs)

        with torch.no_grad():
            adv_preds = model(x_adv).argmax(dim=1)

        if targeted:
            attack_success += (adv_preds == target_class).sum().item()
        else:
            attack_success += (adv_preds != valid_labels).sum().item()

        total_valid += valid_mask.sum().item()

    asr = attack_success / total_valid if total_valid > 0 else 0.0
    return asr, total_valid


# ──────────────────────────────────────────────
# 2. 클래스별 ASR 평가 (target 0~9 순회)
# ──────────────────────────────────────────────
def evaluate_targeted_asr_per_class(model, loader, attack_fn, attack_kwargs,
                                    device=DEVICE):
    """
    target class를 0~9로 순회하며 각 target별 ASR 측정.
    반환: { target_class(int): (asr, n) }
    """
    results = {}
    for target_class in range(10):
        asr, n = evaluate_asr(
            model, loader, attack_fn, attack_kwargs,
            targeted=True, target_class=target_class, device=device,
        )
        results[target_class] = (asr, n)
        print(f"  target={target_class} | ASR: {asr*100:6.2f}%  (n={n})")
    return results


# ──────────────────────────────────────────────
# 3. 시각화 함수
# ──────────────────────────────────────────────
def denormalize_cifar(tensor, mean=(0.4914, 0.4822, 0.4465),
                      std=(0.2470, 0.2435, 0.2616)):
    m = torch.tensor(mean).view(3, 1, 1)
    s = torch.tensor(std).view(3, 1, 1)
    return torch.clamp(tensor.cpu() * s + m, 0.0, 1.0)


def collect_per_class_samples(model, dataset, attack_fn, attack_kwargs,
                               targeted, target_class=None,
                               num_classes=10, device=DEVICE):
    """클래스 0~9 각각에서 공격 성공 샘플 1개씩 수집."""
    model.eval()
    collected = {}

    for idx in range(len(dataset)):
        if len(collected) == num_classes:
            break

        image, label = dataset[idx]
        if label in collected:
            continue
        if targeted and target_class is not None and label == target_class:
            continue

        image = image.unsqueeze(0).to(device)
        label_tensor = torch.tensor([label]).to(device)

        with torch.no_grad():
            clean_pred = model(image).argmax(dim=1).item()
        if clean_pred != label:
            continue

        if targeted:
            target_tensor = torch.tensor([target_class]).to(device)
            x_adv = attack_fn(model, image, target_tensor, **attack_kwargs)
        else:
            x_adv = attack_fn(model, image, label_tensor, **attack_kwargs)

        with torch.no_grad():
            adv_pred = model(x_adv).argmax(dim=1).item()

        if targeted and adv_pred != target_class:
            continue
        if not targeted and adv_pred == label:
            continue

        collected[label] = (image, x_adv, label, clean_pred, adv_pred)

    return collected


def visualize_per_class(collected, eps, fname,
                        is_cifar=False, class_names=None,
                        targeted=True, target_class=None):
    """3행 × N열 그리드로 시각화 후 저장."""
    cols = sorted(collected.keys())
    n_cols = len(cols)
    if n_cols == 0:
        print("  [경고] 수집된 샘플 없음")
        return

    def label_name(i):
        return class_names[i] if class_names else str(i)

    fig, axes = plt.subplots(3, n_cols, figsize=(2.5 * n_cols, 8))
    if n_cols == 1:
        axes = [[axes[r]] for r in range(3)]

    row_titles = ["Original",
                  f"Perturbation\n(eps={eps:.2f}, magnified)",
                  "Adversarial"]
    for r, title in enumerate(row_titles):
        axes[r][0].set_ylabel(title, fontsize=10, labelpad=8)

    for col_idx, cls in enumerate(cols):
        image, x_adv, true_l, clean_p, adv_p = collected[cls]

        if is_cifar:
            orig_np = denormalize_cifar(image.squeeze(0)).numpy().transpose(1, 2, 0)
            adv_np  = denormalize_cifar(x_adv.squeeze(0)).numpy().transpose(1, 2, 0)
            cmap = None
        else:
            orig_np = image.squeeze().cpu().numpy()
            adv_np  = x_adv.squeeze().cpu().numpy()
            cmap = "gray"

        pert_np  = adv_np - orig_np
        pert_vis = (pert_np - pert_np.min()) / (pert_np.max() - pert_np.min() + 1e-8)

        axes[0][col_idx].set_title(f"Class {label_name(cls)}", fontsize=9)
        axes[0][col_idx].imshow(orig_np, cmap=cmap)
        axes[0][col_idx].set_xlabel(f"Pred: {label_name(clean_p)}", fontsize=8)
        axes[0][col_idx].tick_params(left=False, bottom=False,
                                     labelleft=False, labelbottom=False)

        axes[1][col_idx].imshow(pert_vis, cmap=cmap)
        axes[1][col_idx].tick_params(left=False, bottom=False,
                                     labelleft=False, labelbottom=False)

        axes[2][col_idx].imshow(adv_np, cmap=cmap)
        adv_xlabel = f"Pred: {label_name(adv_p)}"
        if targeted and target_class is not None:
            adv_xlabel += f"\n(target={label_name(target_class)})"
        axes[2][col_idx].set_xlabel(adv_xlabel, fontsize=8)
        axes[2][col_idx].tick_params(left=False, bottom=False,
                                     labelleft=False, labelbottom=False)

    plt.suptitle(fname.replace("_", " ").replace(".png", ""), fontsize=12, y=1.01)
    plt.tight_layout()
    save_path = os.path.join(RESULT_DIR, fname)
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  PNG 저장 ({n_cols}개 클래스): results/{fname}")


# ──────────────────────────────────────────────
# 4. main
# ──────────────────────────────────────────────
def main():
    print(f"Using device: {DEVICE}")

    # ── MNIST 데이터 로드 ──────────────────────
    transform_mnist = transforms.Compose([transforms.ToTensor()])
    mnist_train = datasets.MNIST(root="./data", train=True,  download=True, transform=transform_mnist)
    mnist_test  = datasets.MNIST(root="./data", train=False, download=True, transform=transform_mnist)
    mnist_train_loader = DataLoader(mnist_train, batch_size=64,   shuffle=True)
    mnist_test_loader  = DataLoader(mnist_test,  batch_size=1000, shuffle=False)

    mnist_cnn = MNIST_CNN()
    files = glob.glob("./Model/training_result/mnist*.pt")
    if files:
        latest = max(files, key=lambda x: os.path.basename(x))
        mnist_cnn.load_state_dict(torch.load(latest, map_location=DEVICE))
        print(f"[MNIST] 모델 로드: {latest}")
    else:
        print("[MNIST] 저장된 모델 없음 → 학습 시작")
        mnist_cnn, _ = train_model(
            mnist_cnn, mnist_train_loader, mnist_test_loader,
            epochs=10, lr=0.001, device=DEVICE, mode="mnist_cnn",
            save_path="./Model/training_result/mnist_cnn.pt"
        )
    mnist_cnn.to(DEVICE).eval()
    print(f"[MNIST] Clean Accuracy: {evaluate_model(mnist_cnn, mnist_test_loader, DEVICE):.4f}")

    # ── CIFAR-10 데이터 로드 ───────────────────
    CIFAR_MEAN_VAL = (0.4914, 0.4822, 0.4465)
    CIFAR_STD_VAL  = (0.2470, 0.2435, 0.2616)
    transform_cifar = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR_MEAN_VAL, std=CIFAR_STD_VAL),
    ])
    CIFAR_MEAN = torch.tensor(CIFAR_MEAN_VAL).view(3, 1, 1).to(DEVICE)
    CIFAR_STD  = torch.tensor(CIFAR_STD_VAL ).view(3, 1, 1).to(DEVICE)
    CIFAR_MIN  = (0.0 - CIFAR_MEAN) / CIFAR_STD
    CIFAR_MAX  = (1.0 - CIFAR_MEAN) / CIFAR_STD

    cifar_train = datasets.CIFAR10(root="./data", train=True,  download=True, transform=transform_cifar)
    cifar_test  = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_cifar)
    cifar_train_loader = DataLoader(cifar_train, batch_size=64,   shuffle=True)
    cifar_test_loader  = DataLoader(cifar_test,  batch_size=1000, shuffle=False)

    resnet = ResNet18()
    cifar_files = glob.glob("./Model/training_result/cifar_resnet18*.pt")
    if cifar_files:
        finetune = [f for f in cifar_files if "finetune" in f]
        latest   = max(finetune if finetune else cifar_files,
                       key=lambda x: os.path.basename(x))
        resnet.load_state_dict(torch.load(latest, map_location=DEVICE))
        print(f"[CIFAR-10] 모델 로드: {latest}")
    else:
        print("[CIFAR-10] 저장된 모델 없음 → 학습 시작")
        resnet, _ = train_model(
            resnet, cifar_train_loader, cifar_test_loader,
            epochs=20, lr=0.001, device=DEVICE, mode="cifar_resnet18",
            save_path="./Model/training_result/cifar_resnet18.pt"
        )
    resnet.to(DEVICE).eval()
    print(f"[CIFAR-10] Clean Accuracy: {evaluate_model(resnet, cifar_test_loader, DEVICE):.4f}")

    cifar_clip = {"clip_min": CIFAR_MIN, "clip_max": CIFAR_MAX}

    # CSV 초기화 (헤더 작성)
    init_csv()
    print(f"\nCSV 초기화: results/{CSV_FNAME}")

    # ══════════════════════════════════════════
    # [실험 A] Epsilon별 ASR — target=8 고정, Targeted
    # ══════════════════════════════════════════
    print("\n" + "="*70)
    print(f"[실험 A] Epsilon별 ASR  (Targeted, target class={TARGET_CLASS} 고정)")
    print("="*70)

    eps_targeted_experiments = [
        ("MNIST",    "FGSM_Targeted", mnist_cnn, mnist_test_loader,
         fgsm_targeted, {}),
        ("MNIST",    "PGD_Targeted",  mnist_cnn, mnist_test_loader,
         pgd_targeted,  {"k": PGD_K, "eps_step": EPS_STEP}),
        ("CIFAR-10", "FGSM_Targeted", resnet,    cifar_test_loader,
         fgsm_targeted, cifar_clip),
        ("CIFAR-10", "PGD_Targeted",  resnet,    cifar_test_loader,
         pgd_targeted,  {"k": PGD_K, "eps_step": EPS_STEP, **cifar_clip}),
    ]

    for (dataset_name, attack_name, model, loader,
         attack_fn, extra_kwargs) in eps_targeted_experiments:
        print(f"\n{dataset_name} | {attack_name}")
        rows = []
        pbar = tqdm(EPSILON_LIST, desc = f"{dataset_name}  |  {attack_name}")
        for eps in pbar:
            kwargs = {"eps": eps, **extra_kwargs}
            start = time.time()
            asr, n = evaluate_asr(
                model, loader, attack_fn, kwargs,
                targeted=True, target_class=TARGET_CLASS,
            )
            elapsed = time.time() - start
            pbar.set_postfix({
            "eps": f"{eps:.2f}",
            "ASR": f"{asr*100:.2f}%",
            "time": f"{elapsed:.1f}s"
        })
            
            print(f"  eps={eps:.2f}  ASR: {asr*100:6.2f}%  (n={n})")
            rows.append(make_row("A", dataset_name, attack_name,
                                 eps, asr, n, target_class=TARGET_CLASS))
        append_csv(rows)

    # ══════════════════════════════════════════
    # [실험 B] Epsilon별 ASR — Untargeted
    # ══════════════════════════════════════════
    print("\n" + "="*70)
    print("[실험 B] Epsilon별 ASR  (Untargeted)")
    print("="*70)

    eps_untargeted_experiments = [
        ("MNIST",    "FGSM_Untargeted", mnist_cnn, mnist_test_loader,
         fgsm_untargeted, {}),
        ("MNIST",    "PGD_Untargeted",  mnist_cnn, mnist_test_loader,
         pgd_untargeted,  {"k": PGD_K, "eps_step": EPS_STEP}),
        ("CIFAR-10", "FGSM_Untargeted", resnet,    cifar_test_loader,
         fgsm_untargeted, cifar_clip),
        ("CIFAR-10", "PGD_Untargeted",  resnet,    cifar_test_loader,
         pgd_untargeted,  {"k": PGD_K, "eps_step": EPS_STEP, **cifar_clip}),
    ]

    for (dataset_name, attack_name, model, loader,
         attack_fn, extra_kwargs) in eps_untargeted_experiments:
        print(f"\n{dataset_name} | {attack_name}")
        rows = []
        for eps in EPSILON_LIST:
            kwargs = {"eps": eps, **extra_kwargs}
            asr, n = evaluate_asr(
                model, loader, attack_fn, kwargs,
                targeted=False,
            )
            print(f"  eps={eps:.2f}  ASR: {asr*100:6.2f}%  (n={n})")
            # untargeted: target_class는 빈 문자열
            rows.append(make_row("B", dataset_name, attack_name,
                                 eps, asr, n))
        append_csv(rows)

    # ══════════════════════════════════════════
    # [실험 C] Target class별 ASR — eps=0.3 고정
    # ══════════════════════════════════════════
    print("\n" + "="*70)
    print(f"[실험 C] Target class별 ASR  (eps={VIS_EPS} 고정, target 0~9 순회)")
    print("="*70)

    per_class_experiments = [
        ("MNIST",    "FGSM_Targeted", mnist_cnn, mnist_test_loader,
         fgsm_targeted, {}),
        ("MNIST",    "PGD_Targeted",  mnist_cnn, mnist_test_loader,
         pgd_targeted,  {"k": PGD_K, "eps_step": EPS_STEP}),
        ("CIFAR-10", "FGSM_Targeted", resnet,    cifar_test_loader,
         fgsm_targeted, cifar_clip),
        ("CIFAR-10", "PGD_Targeted",  resnet,    cifar_test_loader,
         pgd_targeted,  {"k": PGD_K, "eps_step": EPS_STEP, **cifar_clip}),
    ]

    for (dataset_name, attack_name, model, loader,
         attack_fn, extra_kwargs) in per_class_experiments:
        print(f"\n{dataset_name} | {attack_name}")
        kwargs = {"eps": VIS_EPS, **extra_kwargs}
        per_class_results = evaluate_targeted_asr_per_class(
            model, loader, attack_fn, kwargs,
        )
        rows = [
            make_row("C", dataset_name, attack_name,
                     VIS_EPS, asr, n, target_class=tc)
            for tc, (asr, n) in per_class_results.items()
        ]
        append_csv(rows)

    print(f"\n  CSV 저장 완료: results/{CSV_FNAME}")

    # ══════════════════════════════════════════
    # [시각화] 3행 × N열 (eps=0.3, 클래스별 1샘플)
    # ══════════════════════════════════════════
    print("\n" + "="*70)
    print(f"Saving per-class visualizations (eps={VIS_EPS}) ...")
    print("="*70)

    vis_experiments = [
        ("MNIST    | FGSM Targeted  ", mnist_cnn, mnist_test,
         fgsm_targeted,   {},                                                True,  False, None),
        ("MNIST    | FGSM Untargeted", mnist_cnn, mnist_test,
         fgsm_untargeted, {},                                                False, False, None),
        ("MNIST    | PGD  Targeted  ", mnist_cnn, mnist_test,
         pgd_targeted,    {"k": PGD_K, "eps_step": EPS_STEP},               True,  False, None),
        ("MNIST    | PGD  Untargeted", mnist_cnn, mnist_test,
         pgd_untargeted,  {"k": PGD_K, "eps_step": EPS_STEP},               False, False, None),
        ("CIFAR-10 | FGSM Targeted  ", resnet, cifar_test,
         fgsm_targeted,   cifar_clip,                                        True,  True,  CIFAR10_CLASSES),
        ("CIFAR-10 | FGSM Untargeted", resnet, cifar_test,
         fgsm_untargeted, cifar_clip,                                        False, True,  CIFAR10_CLASSES),
        ("CIFAR-10 | PGD  Targeted  ", resnet, cifar_test,
         pgd_targeted,    {"k": PGD_K, "eps_step": EPS_STEP, **cifar_clip}, True,  True,  CIFAR10_CLASSES),
        ("CIFAR-10 | PGD  Untargeted", resnet, cifar_test,
         pgd_untargeted,  {"k": PGD_K, "eps_step": EPS_STEP, **cifar_clip}, False, True,  CIFAR10_CLASSES),
    ]

    for (name, model, dataset, attack_fn, extra_kwargs,
         targeted, is_cifar, class_names) in vis_experiments:

        kwargs = {"eps": VIS_EPS, **extra_kwargs}
        tc = TARGET_CLASS if targeted else None

        collected = collect_per_class_samples(
            model, dataset, attack_fn, kwargs,
            targeted=targeted, target_class=tc,
        )
        tag = (name.strip()
               .replace(" ", "_").replace("|", "")
               .replace("-", "").replace("__", "_"))
        visualize_per_class(
            collected, eps=VIS_EPS, fname=f"{tag}.png",
            is_cifar=is_cifar, class_names=class_names,
            targeted=targeted, target_class=tc,
        )

    print(f"\n완료. 모든 결과가 {RESULT_DIR}/ 에 저장되었습니다.")


if __name__ == "__main__":
    main()