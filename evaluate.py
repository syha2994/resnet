import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import build_resnet18
import config as cfg


def imshow(tensor, title: str = "") -> None:
    """(C,H,W) Tensor를 클리핑/정규화 해제 후 출력"""
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = (img * std) + mean
    img  = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")


def main() -> None:
    # -------------------------
    # 1. 데이터 로딩
    # -------------------------
    transform = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_dataset = datasets.ImageFolder(cfg.val_dir, transform=transform)
    val_loader  = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers)

    class_names = val_dataset.classes

    # -------------------------
    # 2. 모델 로딩
    # -------------------------
    model = build_resnet18(num_classes=len(class_names), pretrained=False).to(cfg.device)
    if not Path(cfg.save_path).exists():
        raise FileNotFoundError(f"저장된 모델이 없습니다 → {cfg.save_path}")
    model.load_state_dict(torch.load(cfg.save_path, map_location=cfg.device))
    model.eval()

    # -------------------------
    # 3. 정확도 계산
    # -------------------------
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(cfg.device), labels.to(cfg.device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    print(f"✅ 검증 정확도: {correct / total:.4f} ({correct}/{total})")

    # -------------------------
    # 4. 샘플 이미지 10개 시각화
    # -------------------------
    sample_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    plt.figure(figsize=(12, 6))

    for i, (img, label) in enumerate(sample_loader):
        if i >= 10:
            break
        img = img.to(cfg.device)
        with torch.no_grad():
            output = model(img)
            _, pred = torch.max(output, 1)
        plt.subplot(2, 5, i + 1)
        imshow(img[0], title=f"GT: {class_names[label.item()]}\nPred: {class_names[pred.item()]}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()