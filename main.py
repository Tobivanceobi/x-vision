from PIL import Image
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet50_Weights
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
num_features = model.fc.in_features
print(num_features)
