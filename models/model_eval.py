import os

import torch
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from torchvision.transforms import transforms

from models.helper.dataLoader import KermanyXRayImageFolder
from models.helper.eval import eval_model
from models.helper.plots import plot_cm, plot_prc
from models.helper.training import step_train_model
from models.resnet18.model import XRayResNet18
from models.resnet50.model import XRayResNet50

CURR_MODEL = "resnet50"
MODEL_PATH = CURR_MODEL + "/cache/" + CURR_MODEL + ".pth"
# resnet18 MODEL_PARAMS = [128, 6, 5, 0.00011038211496918009, 5.916018460561741e-05, 0.3952291290587717, 256, 256, 1]
MODEL_PARAMS = [32, 5, 6, 0.00015, 0.000489, 0.7, 64, 256, 0]
batch_size, freez_ep, unfreez_ep, lr_1, lr_2, dropout_head, hl1, hl2, num_hl = MODEL_PARAMS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#     ])
# val_dataset = KermanyXRayImageFolder('../data/test', transform=transform)
# val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=int(128), shuffle=True)

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

all_preds, all_true = eval_model(model, set_path='train')
auc = accuracy_score(all_true, all_preds)
print(auc)

all_preds, all_true = eval_model(model, set_path='test')
auc = accuracy_score(all_true, all_preds)
print(auc)

all_preds, all_true = eval_model(model, set_path='val')
auc = accuracy_score(all_true, all_preds)
print(auc)

# plot_cm(all_true, all_preds, classes=labels)
# plt.show()
# plot_prc(all_true, all_preds, classes=labels)
# plt.show()

