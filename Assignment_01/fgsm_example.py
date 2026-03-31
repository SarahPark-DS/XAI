#%%
import sys
import os
import glob

from Attack.fgsm import fgsm_targeted, fgsm_untargeted
sys.path.append(os.path.join(os.path.dirname(__file__), 'Model'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Attack'))

import torch 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from mnist_cnn import *
from fgsm import *
from Model.utils import *

def test_image(image, label, model, target_class, eps = 0.1, device = "cuda" if torch.cuda.is_available() else "cpu"):
    """
    하나의 이미지에 대해서만 테스트 수행
    """
    image = image.to(device) # (1, 1, 28, 28)
    true_label_tensor = torch.tensor([label]).to(device)   

    model.eval()

    # Clean Prediction
    with torch.no_grad():
        clean_pred = model(image).argmax(dim = 1).item()

    # Attack
    target_tensor = torch.tensor([target_class]).to(device)
    x_adv = fgsm_targeted(model, image, target_tensor, eps)

    # Adversarial Prediction
    with torch.no_grad():
        adv_pred = model(x_adv).argmax(dim = 1).item()
        
    
    # 결과 출력
    print(f"True Label  : {label}")
    print(f"Clean Pred  : {clean_pred}")
    print(f"Adv Pred    : {adv_pred}")
    print(f"Attack Goal : {target_class}")
    print(f"Success     : {adv_pred == target_class}")

    return x_adv, clean_pred, adv_pred

def evaluate_targeted_asr(model, loader, target_class, eps, device, num_batches = 100):
    """
    모델과 데이터로더를 이용하여 targeted attack의 성공률(ASR)을 평가
    ASR = (target class로 분류된 샘플 수) / (전체 샘플 수 (target class 제외))
    """

    model.eval()
    total_valid = 0
    attack_success = 0

    for i, (images, labels) in enumerate(loader):
        if i >= num_batches:
            break

        images, labels = images.to(device), labels.to(device)

        # Clean prediction
        with torch.no_grad():
            clean_preds = model(images).argmax(dim = 1)

        # 올바르게 분류디ㅗ고, target class 아닌 샘플만 선택
        valid_mask = (clean_preds == labels) & (labels != target_class)
        if valid_mask.sum() == 0:
            continue

        valid_images = images[valid_mask]
        valid_labels = labels[valid_mask]
        targets = torch.full_like(valid_labels, target_class)

        # Adversarial attack 수행
        x_adv = fgsm_targeted(model, valid_images, targets, eps)

        # Adversarial prediction
        with torch.no_grad():
            adv_preds = model(x_adv).argmax(dim = 1)
        
        attack_success += (adv_preds == target_class).sum().item()
        total_valid += valid_mask.sum().item()

    asr = attack_success / total_valid if total_valid > 0 else 0
    print(f"[eps={eps}] Target Class: {target_class} | "
          f"Valid Samples: {total_valid} | "
          f"Attack Success: {attack_success} | "
          f"ASR: {asr:.4f}")
    
    return asr 


def evaluate_untargeted_asr(model, loader, eps, device, num_batches = 100):
    """
    모델과 데이터로더를 이용하여 untargeted attack의 성공률(ASR)을 평가
    ASR = (원래 클래스에서 벗어난 샘플 수) / (전체 샘플 수)
    """

    model.eval()
    total = 0
    correct_clean = 0
    correct_adv = 0

    for i, (images, labels) in enumerate(loader):
        if i >= num_batches:
            break

        images, labels = images.to(device), labels.to(device)

        # Clean prediction
        with torch.no_grad():
            clean_preds = model(images).argmax(dim = 1)
            correct_clean += (clean_preds == labels).sum().item()

        valid_mask = (clean_preds == labels)
        if valid_mask.sum() == 0:
            total += labels.size(0)
            continue

        # Adversarial attack 수행
        valid_images = images[valid_mask]
        valid_labels = labels[valid_mask]

        x_adv = fgsm_untargeted(model, valid_images, valid_labels, eps)

        # Adversarial prediction
        with torch.no_grad():
            adv_preds = model(x_adv).argmax(dim = 1)
            correct_adv += (adv_preds != valid_labels).sum().item()
        
        total += labels.size(0)

    asr = correct_adv / correct_clean if correct_clean > 0 else 0
    print(f"[eps={eps}] Clean Acc: {correct_clean/total:.4f} | "
          f"Adv Acc: {correct_adv/total:.4f} | "
          f"ASR: {asr:.4f}")

    return asr


def visualize_attack(image, x_adv,  clean_pred, adv_pred, true_label, target_class, eps):
    """
    원본 이미지와 Adversarial 이미지 시각화
    """
     # Visualization of Image before & after attack
    fig, axes = plt.subplots(1, 3, figsize = (10, 3))

    axes[0].imshow(image.squeeze().cpu().numpy(), cmap = 'gray')
    axes[0].set_title(f"Original\nTrue: {true_label} / Pred: {clean_pred}")
    axes[0].axis("off")

    perturbation = (x_adv - image).squeeze().cpu().numpy()
    axes[1].imshow(perturbation, cmap = "RdBu", vmin = -eps, vmax = eps)
    axes[1].set_title(f"Perturbation\n(eps = {eps})")
    axes[1].axis("off")

    axes[2].imshow(x_adv.squeeze().cpu().numpy(), cmap = "gray")
    axes[2].set_title(f"Adversarial\nPred: {adv_pred} (target = {target_class})")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 데이터 로드
    transform = transforms.Compose([
    transforms.ToTensor(), # PyTorch 텐서로 변환
    # transforms.Normalize((0.1307,), (0.3081,)) # MNIST 평균 0.1307, 표준편차 0.3081로 정규화
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # 모델 로드
    model = MNIST_CNN()
    model, _ = train_model(model, DataLoader(train_dataset, batch_size = 128, shuffle = True), device = device, epochs = 5, mode = "mnist")
    # files = glob.glob("./Model/training_result/mnist*.pt")
    # latest_file = max(files, key = lambda x: os.path.basename(x).replace("mnist_", ""))
    # model.load_state_dict(torch.load(latest_file, map_location = device))


    epsilon_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    for eps in epsilon_list:
        print("="*70)
        print("Evaluating targeted attack for eps =", eps)
        t = 4
        evaluate_targeted_asr(model, DataLoader(test_dataset, batch_size = 128, shuffle = False), target_class = t, eps = eps, device = device, num_batches= float('inf'))

        # print("="*70)
        # print("Evaluating untargeted attack for eps =", eps)
        # evaluate_untargeted_asr(model, DataLoader(test_dataset, batch_size = 128, shuffle = False), eps = eps, device = device, num_batches= 100)



# %%
