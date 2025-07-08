import torch.nn as nn
from torchvision import models


def build_resnet18(num_classes: int, *, pretrained: bool = True) -> nn.Module:
    """ResNet‑18 분류 모델을 반환합니다.

    Args:
        num_classes (int): 데이터셋 클래스 수.
        pretrained (bool): ImageNet 사전학습 가중치 사용 여부.

    Returns:
        nn.Module: 커스텀 출력층을 가진 ResNet‑18 모델.
    """
    model = models.resnet18(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model