import torch
import torch.nn as nn
import torch.optim as optim
import copy
import os
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

def train_model(model, train_loader, val_loader = None, epochs = 5, lr = 0.001, optimizer = None, device = None, mode = "cifar", save_path = None):
    if save_path is None:
        f_path = "./training_result/"
        f_name = mode + "_" + datetime.now().strftime("%Y%m%d%H%M")
        save_path = os.path.join(f_path, f_name + ".pt")

    os.makedirs(os.path.dirname(save_path), exist_ok = True)
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_model_weights = None
    history = {"train_acc": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        # train 
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # calculate train accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({'loss': f'{running_loss/len(train_loader):.4f}'})

        train_acc = correct / total
        history["train_acc"].append(train_acc)

        # validation
        if val_loader is not None:
            val_acc = evaluate_model(model, val_loader, device)
            history["val_acc"].append(val_acc)
            print(f" Val Accuracy: {val_acc:.4f}", end = "")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_weights = copy.deepcopy(model.state_dict())
                torch.save(best_model_weights, save_path)
                print(f"     ✅ Best model saved (acc: {best_val_acc:.4f})")
            else:
                print()

    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        print(f"\nTraining completed. Best Val Acc: {best_val_acc:.4f}")
    else:
        print("\nTraining completed")

    return model, history


def evaluate_model(model, test_loader, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy

def plot_accuracy(history, mode):
    fig_name = f"{mode}_accuracy_plot.png"
    fig_path = os.path.join("./training_result/", fig_name)

    plt.figure(figsize = (8,  5))
    plt.plot([acc*100 for acc in history["train_acc"]], label = "Training Accuracy")

    if history["val_acc"]:
        plt.plot([acc*100 for acc in history["val_acc"]], label = "Validation Accuracy")
    
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    epochs = len(history["train_acc"])
    plt.xticks(ticks = range(epochs), labels = range(1, epochs + 1))

    plt.tight_layout()
    plt.savefig(fig_path)
    plt.show()