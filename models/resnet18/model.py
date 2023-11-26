import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


class ClonedResNet18Head(nn.Sequential):
    def forward(self, *input):
        return super().forward(*input).clone()


class XRayResNet18(nn.Module):
    def __init__(self, hl1, hl2, num_hl, dropout_head):
        super(XRayResNet18, self).__init__()
        self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        num_features = self.model.fc.in_features

        self.hidden_layers = [num_features]
        if num_hl == 1:
            self.hidden_layers = [num_features, hl1]
        else:
            self.hidden_layers = [num_features, hl1, hl2]

        self.model.fc = self.build_classification_head(num_features, 2, dropout_head)

    def build_classification_head(self, input_size, output_size, dropout):
        layers = []
        if len(self.hidden_layers) > 1:
            for hidden_layer_size in self.hidden_layers:
                layers.append(nn.Linear(int(input_size), int(hidden_layer_size)))
                layers.append(nn.BatchNorm1d(int(hidden_layer_size)))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                input_size = int(hidden_layer_size)

        layers.append(nn.Linear(int(self.hidden_layers[-1]), output_size))
        layers.append(nn.Softmax(dim=1))

        return ClonedResNet18Head(*layers)

    def freeze_pretrained_layers(self):
        # Freeze pre-trained resnet18
        for param in self.model.parameters():
            param.requires_grad = False
        # Unfreeze the classification head
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def unfreeze_pretrained_layers(self):
        # Unfreeze pre-trained resnet18
        for param in self.model.parameters():
            param.requires_grad = True

    def set_inplace_false(self):
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False

    def forward(self, x):
        return self.model(x)

