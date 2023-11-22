import os.path

import numpy as np
import shap
import torch
from matplotlib import pyplot as plt
from torchvision.transforms import transforms

from models.helper.dataLoader import KermanyXRayImageFolder
from models.helper.eval import eval_model
from models.helper.training import step_train_model
from models.resnet18.model import XRayResNet18
from models.resnet50.model import XRayResNet50

# Resnet50, Vision Transformer
# 2 class and 3 class
# shap vals 2 and 3 class
# confusion matrix, precision recall curve
# train test split 70 15 15

CURR_MODEL = "resnet50"
MODEL_PATH = CURR_MODEL + "/cache/" + CURR_MODEL + ".pth"
# resnet18 MODEL_PARAMS = [128, 6, 5, 0.00011038211496918009, 5.916018460561741e-05, 0.3952291290587717, 256, 256, 1]
MODEL_PARAMS = [32, 5, 6, 0.00015, 0.000489, 0.7, 64, 256, 0]
batch_size, freez_ep, unfreez_ep, lr_1, lr_2, dropout_head, hl1, hl2, num_hl = MODEL_PARAMS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if os.path.isfile(MODEL_PATH):
    print("Found pretrained model in cache")
    print("")
    if CURR_MODEL == "resnet18":
        model = XRayResNet18(hl1, hl2, num_hl, dropout_head)
    elif CURR_MODEL == "resnet50":
        model = XRayResNet50(hl1, hl2, num_hl, dropout_head)
    model.load_state_dict(torch.load(MODEL_PATH))
else:
    print("No pretrained model in cache found")
    print("")
    score, model = step_train_model(
        MODEL_PARAMS,
        pt_model=CURR_MODEL,
        return_model=True)
    torch.save(model.state_dict(), MODEL_PATH)

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


def preprocess_images_for_opencv(images):
    processed_images = []
    for img in images:
        # Convert to OpenCV format (H, W, C)
        img = img.permute(1, 2, 0).numpy()
        # Scale to 0-255 and convert to 8-bit unsigned integer
        img_processed = (img * 255).astype(np.uint8)
        processed_images.append(img_processed)
    return np.array(processed_images)

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
X_processed = preprocess_images_for_opencv(X)

Y_preds_class = np.array(Y_preds_class)
Y_preds = np.array(Y_preds)

accuracies = np.array([Y_preds[l][Y[l]] for l in range(len(Y))])

sorted_idx = np.argsort(accuracies)
X_processed = X_processed[sorted_idx]
Y = Y[sorted_idx]
Y_preds_class = Y_preds_class[sorted_idx]
accuracies = accuracies[sorted_idx]

X_sample_idx = [idx for l in [0, 1, 2] for idx in np.where(Y == l)[0][-6:]]

X_sample = np.array(X_processed[X_sample_idx])
Y_sample = np.array(Y[X_sample_idx])
Y_sample_preds = Y_preds_class[X_sample_idx]
sample_accuracies = accuracies[X_sample_idx]

print(Y_sample_preds)
print(Y_sample)
print(sample_accuracies)


def f(x):
    tmp = x.transpose(0, 3, 1, 2)
    tmp = tmp.astype(np.float32) / 255

    tmp = torch.tensor(tmp)

    tmp = tmp.to(next(model.parameters()).device)

    with torch.no_grad():
        output = model(tmp)

    if len(output.shape) == 1:
        output = output.unsqueeze(0)
    return output.cpu().numpy()


masker = shap.maskers.Image("blur(128,128)", X_processed[0].shape)

class_names = val_dataset.classes
explainer = shap.Explainer(f, masker, output_names=class_names)

shap_values = explainer(X_sample[:-1], max_evals=5000, batch_size=50, outputs=shap.Explanation.argsort.flip[:3])

for i in range(len(shap_values)):
    shap.image_plot(shap_values[i])
    print(Y_sample_preds[i])

# e = shap.DeepExplainer(model, background)
# shap_values = e.shap_values(test_images, check_additivity=False)
#
# test_images_np = test_images.cpu().numpy()
# test_images_np = np.transpose(test_images_np, (0, 2, 3, 1))
#
# selected_images = test_images_np[:5]
# selected_shap_values = [values[:5] for values in shap_values]
#
# shap.image_plot(selected_shap_values, -selected_images)
