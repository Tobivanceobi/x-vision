import timm
import torch
import torchvision.transforms as transforms
from PIL.ImageFile import ImageFile
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from torchvision.datasets import ImageFolder
from torch import nn, optim

from dataLoader import DataLoader

ImageFile.LOAD_TRUNCATED_IMAGES = True

data_config = {
    # "Lung Lesion": 1000,
    # "Edema": 1000,
    "No Finding": 3000,
    # "Consolidation": 1000,
    "Pneumonia": 3000,
    # "Atelectasis": 1000,
    # "Pneumothorax": 1000
}

dl = DataLoader()
dl.get_image_paths(data_config)
x_train, y_train = dl.load_images_from_files()

X_train, Y_train, X_val, Y_val = train_test_split(x_train, y_train, stratify=True, test_size=0.2)

# Convert arrays to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.float32)
Y_val = torch.tensor(Y_val, dtype=torch.long)

# Create TensorDatasets
train_dataset = TensorDataset(X_train, Y_train)
val_dataset = TensorDataset(X_val, Y_val)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Load a Pre-trained ViT Model
model_pre = timm.create_model('vit_base_patch16_224', pretrained=True)

# Modify the Model for Your Task
num_classes = Y_train.unique().size(0)
model_pre.head = nn.Linear(model_pre.head.in_features, num_classes)

# Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_pre.parameters(), lr=0.001)


# Training Loop
def train(model, crt, opt, trl, vall, epochs=10):
    model.train()
    for epoch in range(epochs):
        for images, labels in trl:
            opt.zero_grad()
            outputs = model(images)
            loss = crt(outputs, labels)
            loss.backward()
            opt.step()

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

        # Validation
        validate(model, vall)


def validate(model, vall):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in vall:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f'Validation Accuracy: {100 * correct / total}%')


# Train the Model
train(model_pre, criterion, optimizer, train_loader, val_loader)
