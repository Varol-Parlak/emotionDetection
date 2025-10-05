import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os 
from torchvision.models import EfficientNet_B0_Weights

preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),
    transforms.RandomHorizontalFlip(), 
    transforms.RandomRotation(15),   
    transforms.ColorJitter(brightness=0.1, contrast=0.1), 
])
train_dir = "C:/Users/Varol/Desktop/dataset_emotions"

dataset = datasets.ImageFolder(root = train_dir, transform = preprocess)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

num_features = model.classifier[1].in_features
num_classes = len(dataset.classes)
model.classifier[1] = nn.Linear(num_features, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
weights = torch.tensor([1, 1.3, 1.3, 0.6], dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight = weights)
optimizer = optim.Adam(model.parameters(), lr=5e-4)


for epoch in range(20):
    model.train()
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}: Loss ={loss.item(): .4f}")

torch.save(model.state_dict(), "emotion_model.pth")
    