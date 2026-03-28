import torchvision.models as models
import torch.nn as nn

class ResNet18(nn.Module):
    def __init__(self, num_classes = 10, freeze_backbone = False):
        super().__init__()
        # pre-trained weights 로드
        self.model = models.resnet18(weights = models.ResNet18_Weights.IMAGENET1K_V1)

        # CIFAR-10 32X32 맞게 수정
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.model.maxpool = nn.Identity()

        # backbone 동결 옵션
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        # Classifier만 교체
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.model(x)