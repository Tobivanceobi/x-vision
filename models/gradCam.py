import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import transforms

from models.helper.dataLoader import KermanyXRayImageFolder
from models.resnet18.model import XRayResNet18
from models.resnet50.model import XRayResNet50


def apply_gradcam(input_tensor, model, target_layer):
    """
    Apply Grad-CAM to the given tensor and model.
    """
    # Ensure model is in eval mode
    model.eval()

    # Register hook for the gradients
    gradients = []
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    handle = target_layer.register_backward_hook(backward_hook)

    # Forward pass
    output = model(input_tensor)

    # Target for backprop
    one_hot_output = torch.FloatTensor(1, output.size()[-1]).zero_()
    one_hot_output[0][output.argmax()] = 1

    # Backward pass
    model.zero_grad()
    output.backward(gradient=one_hot_output, retain_graph=True)

    # Remove the hook
    handle.remove()

    # Get the gradients and feature maps
    gradients = gradients[0]
    target_layer_output = target_layer.output  # Assuming the feature maps are saved here

    # Weighted combination of feature maps
    weights = torch.mean(gradients, [0, 2, 3])
    cam = torch.sum(weights * target_layer_output, dim=1)

    # Normalize and convert to image format
    cam = cam.detach().cpu().numpy()[0]
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)

    return cam




CURR_MODEL = "resnet50"
MODEL_PATH = CURR_MODEL + "/cache/" + CURR_MODEL + ".pth"
# MODEL_PARAMS = [128, 6, 5, 0.00011038211496918009, 5.916018460561741e-05, 0.3952291290587717, 256, 256, 1]
MODEL_PARAMS = [32, 6, 3, 0.00015, 0.00005, 0.7, 64, 256, 0]
batch_size, freez_ep, unfreez_ep, lr_1, lr_2, dropout_head, hl1, hl2, num_hl = MODEL_PARAMS

if os.path.isfile(MODEL_PATH):
    print("Found pretrained model in cache")
    print("")
    if CURR_MODEL == "resnet18":
        model = XRayResNet18(hl1, hl2, num_hl, dropout_head)
    elif CURR_MODEL == "resnet50":
        model = XRayResNet50(hl1, hl2, num_hl, dropout_head)
    model.load_state_dict(torch.load(MODEL_PATH))
else:
    from models.helper.training import step_train_model
    print("No pretrained model in cache found")
    print("")
    score, model = step_train_model(
        MODEL_PARAMS,
        pt_model=CURR_MODEL,
        return_model=True)
    torch.save(model.state_dict(), MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

val_dataset = KermanyXRayImageFolder('../data/val', transform=transform)

X = val_dataset.data
Y = val_dataset.targets

print("========== Computing Shap Values ==========")
print("")

model.set_inplace_false()
model.eval()

Y_preds = []
Y_preds_class = []
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=int(128), shuffle=True)

with torch.no_grad():
    for data, target, _ in val_loader:
        data, target = data.to(device), target.to(device)
        preds = model(data)
        p_class = torch.argmax(preds, dim=1)

        Y_preds.extend(preds.cpu().numpy())
        Y_preds_class.extend(p_class.cpu().numpy())


# Choose an image from the validation dataset
input_image, _ = val_dataset[0]  # Replace 0 with the index of the image you want to visualize
input_tensor = input_image.unsqueeze(0).to(device)

# Assuming target_layer is the last convolutional layer of the ResNet model
target_layer = model.model.layer4[-1]  # Adjust this according to your model structure

# Compute Grad-CAM
cam = apply_gradcam(input_tensor, model, target_layer)

# Display the Grad-CAM heatmap
plt.imshow(cam, cmap='jet')
plt.axis('off')
plt.show()

def overlay_heatmap_on_image(heatmap, original_image, alpha=0.4):
    """
    Overlay the Grad-CAM heatmap on the original image.
    """
    # Convert heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Convert tensor image to numpy array
    original_image = original_image.permute(1, 2, 0).cpu().numpy()
    original_image = np.uint8(255 * original_image)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)

    # Overlay the heatmap on the original image
    overlayed_image = cv2.addWeighted(original_image, alpha, heatmap, 1 - alpha, 0)

    return overlayed_image

# Choose an image from the validation dataset
input_image, _ = val_dataset[0]  # Replace 0 with the index of the image you want to visualize
input_tensor = input_image.unsqueeze(0).to(device)

# Compute Grad-CAM
cam = apply_gradcam(input_tensor, model, target_layer)

# Overlay heatmap on the original image
overlayed_image = overlay_heatmap_on_image(cam, input_image)

# Display the overlayed image
plt.imshow(cv2.cvtColor(overlayed_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

