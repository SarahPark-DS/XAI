import os
import glob

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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
TARGET_CLASS = 4        # targeted attack 목표 클래스
PGD_K        = 40       # PGD iteration 수
EPS_STEP     = 0.01     # PGD step size
EPSILON_LIST = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]   # 실험 Epsilon 리스트
VIS_EPS      = 0.3      # 시각화용 epsilon

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

os.makedirs(RESULT_DIR, exist_ok=True)


# ──────────────────────────────────────────────
# 1. 범용 ASR 평가 함수
# ──────────────────────────────────────────────
def evaluate_asr(model, loader, attack_fn, attack_kwargs,
                 targeted, target_class=None,
                 device=DEVICE, num_batches=float('inf')):
    """
    범용 ASR 평가 함수.
    targeted=True  → attack_fn(model, x, target_tensor, **attack_kwargs)
                     성공 기준: adv_pred == target_class
    targeted=False → attack_fn(model, x, label_tensor,  **attack_kwargs)
                     성공 기준: adv_pred != true_label
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

        # clean 예측이 맞는 샘플만 선택
        valid_mask = (clean_preds == labels)
        # targeted: 이미 target class인 샘플 제외
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


def evaluate_targeted_asr_per_class(model, loader, attack_fn, attack_kwargs, device = DEVICE):
    """
    target class 0-9 각각에 대해 ASR 측정
    Return: {target_class: asr}
    """
    results = {}
    for target_class in range(10):
        asr, n = evaluate_asr(
            model, loader, attack_fn, attack_kwargs, targeted = True, target_class = target_class, device = device
        )
        results[target_class] = asr
        print(f"  target={target_class} | ASR: {asr*100:6.2f}%  (n={n})")
    return results


# ──────────────────────────────────────────────
# 2. 시각화 함수
# ──────────────────────────────────────────────
def denormalize_cifar(tensor, mean=(0.4914, 0.4822, 0.4465),
                      std=(0.2470, 0.2435, 0.2616)):
    """normalize된 CIFAR-10 텐서를 [0,1] 픽셀 공간으로 복원. shape: (C,H,W)"""
    m = torch.tensor(mean).view(3, 1, 1)
    s = torch.tensor(std).view(3, 1, 1)
    return torch.clamp(tensor.cpu() * s + m, 0.0, 1.0)


def collect_per_class_samples(model, dataset, attack_fn, attack_kwargs,
                               targeted, target_class=None,
                               num_classes=10, device=DEVICE):
    """
    클래스 0~9 각각에서 공격 성공 샘플을 1개씩 수집.
    targeted=True일 때 true label == target_class인 샘플은 제외 (9개 수집).
    반환: { class_idx: (image, x_adv, true_label, clean_pred, adv_pred) }
    """
    model.eval()
    collected = {}

    for idx in range(len(dataset)):
        if len(collected) == num_classes:
            break

        image, label = dataset[idx]

        # 이미 해당 클래스 샘플 수집 완료
        if label in collected:
            continue
        # targeted: true label이 target class인 샘플 제외
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
    """
    3행 × N열 그리드로 시각화 후 저장.
    행: Original (0행) / Perturbation magnified (1행) / Adversarial (2행)
    열: 클래스별 샘플 (수집된 순서대로)
    """
    cols = sorted(collected.keys())
    n_cols = len(cols)
    if n_cols == 0:
        print("  [경고] 수집된 샘플 없음")
        return

    def label_name(i):
        return class_names[i] if class_names else str(i)

    fig, axes = plt.subplots(3, n_cols, figsize=(2.5 * n_cols, 8))
    # n_cols=1인 경우 axes를 2D로 맞춤
    if n_cols == 1:
        axes = [[axes[r]] for r in range(3)]

    # 행 레이블
    row_titles = [
        f"Original",
        f"Perturbation\n(eps={eps:.2f}, magnified)",
        f"Adversarial",
    ]
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

        # 열 제목: 클래스 이름
        axes[0][col_idx].set_title(f"Class {label_name(cls)}", fontsize=9)

        # 0행: Original
        axes[0][col_idx].imshow(orig_np, cmap=cmap)
        axes[0][col_idx].set_xlabel(f"Pred: {label_name(clean_p)}", fontsize=8)
        axes[0][col_idx].tick_params(left=False, bottom=False,
                                     labelleft=False, labelbottom=False)

        # 1행: Perturbation
        axes[1][col_idx].imshow(pert_vis, cmap=cmap)
        axes[1][col_idx].tick_params(left=False, bottom=False,
                                     labelleft=False, labelbottom=False)

        # 2행: Adversarial
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
    print(f"  저장 완료 ({n_cols}개 클래스): results/{fname}")


# ──────────────────────────────────────────────
# 3. main
# ──────────────────────────────────────────────
def main():
    print(f"Using device: {DEVICE}")

    # ── MNIST 데이터 로드 ──────────────────────
    transform_mnist = transforms.Compose([
        transforms.ToTensor(),  # normalize 없음 → 유효 범위 [0, 1]
    ])
    mnist_train = datasets.MNIST(root="./data", train=True,  download=True, transform=transform_mnist)
    mnist_test  = datasets.MNIST(root="./data", train=False, download=True, transform=transform_mnist)
    mnist_train_loader = DataLoader(mnist_train, batch_size=64,   shuffle=True)
    mnist_test_loader  = DataLoader(mnist_test,  batch_size=1000, shuffle=False)

    # ── MNIST 모델 로드 (없으면 학습) ──────────
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

    # normalize된 공간에서의 유효 픽셀 범위 (채널별, shape: (3,1,1))
    CIFAR_MEAN = torch.tensor(CIFAR_MEAN_VAL).view(3, 1, 1).to(DEVICE)
    CIFAR_STD  = torch.tensor(CIFAR_STD_VAL ).view(3, 1, 1).to(DEVICE)
    CIFAR_MIN  = (0.0 - CIFAR_MEAN) / CIFAR_STD
    CIFAR_MAX  = (1.0 - CIFAR_MEAN) / CIFAR_STD

    cifar_train = datasets.CIFAR10(root="./data", train=True,  download=True, transform=transform_cifar)
    cifar_test  = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_cifar)
    cifar_train_loader = DataLoader(cifar_train, batch_size=64,   shuffle=True)
    cifar_test_loader  = DataLoader(cifar_test,  batch_size=1000, shuffle=False)

    # ── CIFAR-10 모델 로드 (없으면 학습) ───────
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

    # ── attack kwargs 정의 ─────────────────────
    cifar_clip = {"clip_min": CIFAR_MIN, "clip_max": CIFAR_MAX}

    experiments = [
        ("MNIST    | FGSM Targeted  ", mnist_cnn, mnist_test_loader, mnist_test,
         fgsm_targeted,   {},                                          True,  False, None),
        ("MNIST    | FGSM Untargeted", mnist_cnn, mnist_test_loader, mnist_test,
         fgsm_untargeted, {},                                          False, False, None),
        ("MNIST    | PGD  Targeted  ", mnist_cnn, mnist_test_loader, mnist_test,
         pgd_targeted,    {"k": PGD_K, "eps_step": EPS_STEP},         True,  False, None),
        ("MNIST    | PGD  Untargeted", mnist_cnn, mnist_test_loader, mnist_test,
         pgd_untargeted,  {"k": PGD_K, "eps_step": EPS_STEP},         False, False, None),

        ("CIFAR-10 | FGSM Targeted  ", resnet, cifar_test_loader, cifar_test,
         fgsm_targeted,   cifar_clip,                                  True,  True, CIFAR10_CLASSES),
        ("CIFAR-10 | FGSM Untargeted", resnet, cifar_test_loader, cifar_test,
         fgsm_untargeted, cifar_clip,                                  False, True, CIFAR10_CLASSES),
        ("CIFAR-10 | PGD  Targeted  ", resnet, cifar_test_loader, cifar_test,
         pgd_targeted,    {"k": PGD_K, "eps_step": EPS_STEP, **cifar_clip}, True,  True, CIFAR10_CLASSES),
        ("CIFAR-10 | PGD  Untargeted", resnet, cifar_test_loader, cifar_test,
         pgd_untargeted,  {"k": PGD_K, "eps_step": EPS_STEP, **cifar_clip}, False, True, CIFAR10_CLASSES),
    ]

    # ══════════════════════════════════════════
    # ASR 평가 (epsilon_list 전체)
    # ══════════════════════════════════════════
    print("\n" + "="*70)
    print(f"{'Attack':<35} {'eps':>6} {'ASR':>8}  {'n':>7}")
    print("="*70)

    for (name, model, loader, dataset,
         attack_fn, extra_kwargs, targeted,
         is_cifar, class_names) in experiments:

        print(f"\n{name.strip()}")
        for eps in EPSILON_LIST:
            kwargs = {"eps": eps, **extra_kwargs}
            asr, n = evaluate_asr(
                model, loader, attack_fn, kwargs,
                targeted=targeted,
                target_class=TARGET_CLASS if targeted else None,
            )
            print(f"  eps={eps:.2f}  ASR: {asr*100:6.2f}%  (n={n})")

    # ══════════════════════════════════════════
    # 시각화 (VIS_EPS 기준, 3행 × N열)
    # 행: Original / Perturbation / Adversarial
    # 열: 클래스 0~9 (targeted는 target class 제외하여 9열)
    # ══════════════════════════════════════════
    print("\n" + "="*70)
    print(f"Saving per-class visualizations (eps={VIS_EPS}) ...")
    print("="*70)

    for (name, model, loader, dataset,
         attack_fn, extra_kwargs, targeted,
         is_cifar, class_names) in experiments:

        kwargs = {"eps": VIS_EPS, **extra_kwargs}

        collected = collect_per_class_samples(
            model, dataset, attack_fn, kwargs,
            targeted=targeted,
            target_class=TARGET_CLASS if targeted else None,
        )

        tag = (name.strip()
               .replace(" ", "_")
               .replace("|", "")
               .replace("-", "")
               .replace("__", "_"))
        fname = f"{tag}.png"

        visualize_per_class(
            collected, eps=VIS_EPS, fname=fname,
            is_cifar=is_cifar, class_names=class_names,
            targeted=targeted,
            target_class=TARGET_CLASS if targeted else None,
        )

    print(f"\n완료. 모든 결과가 {RESULT_DIR}/ 에 저장되었습니다.")


if __name__ == "__main__":
    main()