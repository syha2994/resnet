import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import vessl

from model import build_resnet18


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", 32)))
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", 10)))
    parser.add_argument("--learning-rate", type=float, default=float(os.environ.get("LEARNING_RATE", 1e-3)))
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("NUM_WORKERS", 4)))
    parser.add_argument("--image-size", type=int, default=int(os.environ.get("IMAGE_SIZE", 224)))
    parser.add_argument("--train-dir", type=str, default=os.environ.get("TRAIN_DATA_DIR", "/Users/seungyeon/PycharmProjects/MLOps_example/classification/cat_and_dog/train"))
    parser.add_argument("--val-dir", type=str, default=os.environ.get("VAL_DATA_DIR", "/Users/seungyeon/PycharmProjects/MLOps_example/classification/cat_and_dog/val"))
    parser.add_argument("--save-path", type=str, default=os.environ.get("MODEL_DIR", "/Users/seungyeon/PycharmProjects/MLOps_example/classification/resnet/best_model.pth"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------
    # 1. 데이터 로더 정의
    # ----------------------------
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(args.train_dir, transform=transform)
    val_dataset   = datasets.ImageFolder(args.val_dir,   transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)

    num_classes = len(train_dataset.classes)

    # ----------------------------
    # 2. 모델 / 손실 / 옵티마이저
    # ----------------------------
    model = build_resnet18(num_classes=num_classes, pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_acc: float = 0.0

    # ----------------------------
    # 3. 학습 루프
    # ----------------------------
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.4f}")

        # ------------------------
        # 4. 검증 루프
        # ------------------------
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

        val_acc = correct / total
        print(f"[Epoch {epoch+1}] Val Acc: {val_acc:.4f}")

        # ------------------------
        # 5. 최고 성능 모델 저장
        # ------------------------
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), args.save_path)
            print(f"✅ 모델 저장: {args.save_path} (Val Acc: {val_acc:.4f})")

        vessl.log({"accuracy": best_acc, "loss": avg_loss})
    print("🎉 학습 완료 – Best Val Acc:", best_acc)


if __name__ == "__main__":
    main()
