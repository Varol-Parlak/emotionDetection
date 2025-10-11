import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torchvision.models import EfficientNet_B0_Weights
from PIL import Image
import os

model = torch.load("emotion_model.pth")
img_path = "C:/Users/Varol/Desktop/analyze/images/angry_person.jpg"
dataset = datasets.ImageFolder(root="C:/Users/Varol/Desktop/dataset_emotions") 
class_names = dataset.classes
model_path = "C:/Users/Varol/Desktop/analyze/emotion_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
preprocces = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 4)

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
model = model.to(device)

try:
    img = Image.open(img_path).convert("RGB")
    input_tensor = preprocces(img)
    input_batch = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_batch)

    probs = torch.nn.functional.softmax(output[0], dim = 0) 
    emo_index = torch.argmax(probs).item()

    emotion = class_names[emo_index]
    confidence = probs[emo_index].item() * 100

    print(f"Predicted Emotion: {emotion}")
    print(f"Confidence: {confidence:.2f}%")
except Exception as e:
    print(f"bir hata oluştu {e}")

# My model seems to bias the sad emotion i still havent figured out why
