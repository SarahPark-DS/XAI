
import torch
import torch.nn as nn

def fgsm_targeted(model, x, target, eps):
    """
    model   : the neural network
    x       : input image tensor (reqruies_grad should be set)
    target  : the desired (wrong) class label
    eps     : perturbation magnitude (e.g., 0.1, 0.3)
    return  : adversarial image x_adv
    """

    # requires_grad 설정
    x = x.clone().detach().requires_grad_(True)

    # Forward pass를 통한 logit 계산
    output = model(x)

    # target 레이블에 대한 loss 계산 (cross-entropy)
    loss = nn.CrossEntropyLoss()(output, target)

    # Gradient 계산
    model.zero_grad()
    loss.backward()

    # Adversarial image 생성
    x_adv = x - eps * x.grad.sign()
    x_adv = torch.clamp(x_adv, 0, 1)

    return x_adv.detach()


def evaluate_targeted_attack(model, loader, target_class, eps, device, num_batches = 10):
    model.eval()
    correct_clean = correct_adv = total = 0

    for i, (images, labels) in enumerate(loader):
        if i >= num_batches:
            break

        images, labels = images.to(device), labels.to(device)

        # target label: 모든 샘플을 target_class로 분류하도록 유도
        targets = torch.full_like(labels, target_class)

        # Clean 정확도
        with torch.no_grad():
            clean_preds = model(images).argmax(dim = 1)
            correct_clean += (clean_preds == labels).sum().item()

        # Adversarial 이미지 생성 및 예측
        x_adv = fgsm_targeted(model, images, targets, eps)
        with torch.no_grad():
            adv_preds = model(x_adv).argmax(dim = 1)
            correct_adv += (adv_preds == labels).sum().item()

        total += labels.size(0)

    
    print(f"[eps={eps}] Clean Acc: {correct_clean/total:.4f} | "
          f"Adv Acc (targeted→{target_class}): {correct_adv/total:.4f} | "
          f"Attack Success: {1 - correct_adv/total:.4f}")