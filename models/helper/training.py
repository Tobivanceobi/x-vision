import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from models.helper.dataLoader import KermanyXRayImageFolder
from models.resnet18.model import XRayResNet18
from models.resnet50.model import XRayResNet50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_kermany_dataset(bs):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = KermanyXRayImageFolder('../data/train', transform=transform)
    test_dataset = KermanyXRayImageFolder('../data/test', transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=int(bs), shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=int(bs), shuffle=False)
    return train_loader, test_loader


def step_train_model(params, pt_model, return_model=False):
    batch_size, freez_ep, unfreez_ep, lr_1, lr_2, dropout_head, hl1, hl2, num_hl = params

    train_loader, test_loader = get_kermany_dataset(batch_size)

    print("Batch Size: ", batch_size)
    print("Epochs Freez: ", freez_ep)
    print("Epochs Unfreez: ", unfreez_ep)
    print("Learning rate 1: ", lr_1)
    print("Learning rate 2: ", lr_2)
    print("Dropout: ", dropout_head)
    print("Hidden Layers: ", num_hl)
    print("")

    if pt_model == "resnet18":
        model = XRayResNet18(hl1, hl2, num_hl, dropout_head)
    elif pt_model == "resnet50":
        model = XRayResNet50(hl1, hl2, num_hl, dropout_head)

    model.freeze_pretrained_layers()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr_1)

    print(f"========== Starting First Training {pt_model} on Device {device} ==========")
    print("")
    trained_model, _, _ = train(
        model,
        train_loader, test_loader,
        criterion, optimizer,
        num_epochs=freez_ep)

    model.unfreeze_pretrained_layers()

    optimizer = optim.Adam(model.parameters(), lr_2)

    print(f"========== Starting Second Training {pt_model} on Device {device} ==========")
    print("")
    trained_model, best_score, auc = train(
        trained_model,
        train_loader, test_loader,
        criterion, optimizer,
        num_epochs=unfreez_ep)

    print("Best AUC: ", auc)
    print("Best Loss: ", best_score)
    print("")

    if return_model:
        return best_score, trained_model
    return best_score


def train(model, train_loader, val_loader, criterion, optimizer, num_epochs=25):
    best_score = 10
    best_model = None
    best_auc = 0
    patience = 3
    counter = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, labels, _ in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}")
        if (total_loss / len(train_loader)) < best_score:
            best_score = (total_loss / len(train_loader))

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels, _ in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        auc = 100 * correct / total
        print(f"Validation Accuracy: {auc}%")

        if best_auc < auc:
            best_auc = auc
            counter = 0
            best_model = model
        else:
            counter += 1
            print(f"Early stopping: round - {counter}")
            if counter >= patience:
                print("Early stopping triggered")
                break
        print("")
    return best_model, best_score, best_auc
