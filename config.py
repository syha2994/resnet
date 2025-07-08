import torch

# 하이퍼파라미터
batch_size: int = 32
epochs: int = 10
learning_rate: float = 1e-3
num_workers: int = 4
image_size: int = 224

# 디바이스 설정 (GPU 우선)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 경로 (프로젝트 루트 기준)
train_dir: str = "/Users/seungyeon/PycharmProjects/MLOps_example/classification/cat_and_dog/train"
val_dir: str   = "/Users/seungyeon/PycharmProjects/MLOps_example/classification/cat_and_dog/val"

# 모델 저장 경로
save_path: str = "/Users/seungyeon/PycharmProjects/MLOps_example/classification/resnet/best_model.pth"