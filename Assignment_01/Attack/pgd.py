import torch
import torch.nn as nn   

def pgd_targeted(model, x, target, k, eps, eps_step, clip_min = 0.0, clip_max = 1.0):
    """
    model    : the neural network
    x        : input image tensor
    target   : desired (wrong) class label
    k        : number of iterations (e.g., 10, 40)
    eps      : total perturbation budget
    eps_step : step size per iteration
    return   : adversarial image x_adv
    """

    model.eval()
    x_adv = x.clone().detach()

    for _ in range(k):
        x_adv.requires_grad_(True)
        output = model(x_adv)
        loss = nn.CrossEntropyLoss()(output, target)
        
        model.zero_grad()
        loss.backward()

        x_adv = x_adv - eps_step * torch.sign(x_adv.grad)
        x_adv = torch.clamp(x_adv, x - eps, x + eps)
        x_adv = torch.clamp(x_adv, clip_min, clip_max).detach()

    return x_adv


def pgd_untargeted(model, x, label, k, eps, eps_step, clip_min = 0.0, clip_max = 1.0):
    """
    model       : the neural network
    x           : input image tensor
    label       : the correct class label
    k           : number of iterations (e.g., 10, 40)
    eps         : total perturbation budget
    eps_step    : step size per iteration
    return      : adversarial image x_adv
    """
    model.eval()

    x_adv = x.clone().detach()

    for _ in range(k):
        x_adv.requires_grad_(True)
        output = model(x_adv)
        loss = nn.CrossEntropyLoss()(output, label)

        model.zero_grad()
        loss.backward()

        x_adv = x_adv + eps_step * torch.sign(x_adv.grad) # untargeted로 loss 최대화
        x_adv = torch.clamp(x_adv, x - eps, x + eps) # 원본 x에서 eps 이상 벗어나지 않도록 함
        x_adv = torch.clamp(x_adv, clip_min, clip_max).detach()

    return x_adv
