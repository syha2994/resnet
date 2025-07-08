import torch
import torch.nn.functional as F
from model import build_resnet18
from torchvision import transforms
from PIL import Image
import io
import os

# 모델 로드
def model_fn(model_dir):
    num_classes = 2  # 예시: 고양이/강아지
    model = build_resnet18(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(os.path.join(model_dir, "best_model.pth"), map_location="cpu"))
    model.eval()
    return model

# 입력 디코딩
def input_fn(request_body, content_type="application/x-image"):
    if content_type == "application/x-image":
        image = Image.open(io.BytesIO(request_body)).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        return transform(image).unsqueeze(0)
    raise Exception(f"Unsupported content type: {content_type}")

# 추론 실행
def predict_fn(input_tensor, model):
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
    return probabilities

# 결과 포맷
def output_fn(prediction, accept="application/json"):
    pred_list = prediction.squeeze().tolist()
    return str(pred_list)