
import torch
import torch.nn as nn

def fgsm_targeted(model, x, target, eps, clip_min = 0.0, clip_max = 1.0):
    """
    model   : the neural network
    x       : input image tensor (reqruies_grad should be set)
    target  : the desired (wrong) class label
    eps     : perturbation magnitude (e.g., 0.1, 0.3)
    return  : adversarial image x_adv
    """

    # requires_grad 설정
    model.eval()
    x = x.clone().detach().requires_grad_(True)

    # Forward pass를 통한 logit 계산
    output = model(x)

    # target 레이블에 대한 loss 계산 (cross-entropy)
    loss = nn.CrossEntropyLoss()(output, target)

    # Gradient 계산
    model.zero_grad()
    loss.backward()

    # Adversarial image 생성
    x_adv = x - eps * torch.sign(x.grad)
    x_adv = torch.clamp(x_adv, clip_min, clip_max)

    return x_adv.detach()


def fgsm_untargeted(model, x, label, eps, clip_min = 0.0, clip_max = 1.0):
    """
    model   : the neural network
    x       : input image tensor (reqruies_grad should be set)
    label   : the true class label
    eps     : perturbation magnitude
    return  : adversarial image x_adv
    """

    # requires_grad 설정
    model.eval()
    x = x.clone().detach().requires_grad_(True)

    # Forward pass를 통한 logit 계산
    output = model(x)

    # true 레이블에 대한 loss 계산 (cross-entropy)
    loss = nn.CrossEntropyLoss()(output, label)

    # Gradient 계산
    model.zero_grad()
    loss.backward()

    # Adversarial image 생성
    x_adv = x + eps * torch.sign(x.grad)

    x_adv = torch.clamp(x_adv, clip_min, clip_max)

    return x_adv.detach()


