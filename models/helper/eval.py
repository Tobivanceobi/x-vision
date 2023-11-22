import torch
from torchvision.transforms import transforms

from models.helper.dataLoader import KermanyXRayImageFolder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_model(model, set_path="val"):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    val_dataset = KermanyXRayImageFolder('../data/' + set_path, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=int(128), shuffle=True)

    model.to(device)
    model.eval()
    all_preds = []
    all_true = []

    with torch.no_grad():
        for data, target, _ in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            preds = torch.argmax(output, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_true.extend(target.cpu().numpy())

    return all_preds, all_true
