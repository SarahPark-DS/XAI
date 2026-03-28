import torch
import torch.nn as nn
import torch.optim as optim
import copy
import os
from datetime import datetime
from tqdm import tqdm

def train_model(model, train_loader, val_loader = None, epochs = 5, lr = 0.001, device = None, mode = "cifar", save_path = None):
    if save_path is None:
        f_path = "./training_result/"
        f_name = mode + "_" + datetime.now().strftime("%Y%m%d%H%M")
        save_path = os.path.join(f_path, f_name + ".pt")

    os.makedirs(os.path.dirname(save_path), exist_ok = True)
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_model_weights = None

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

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
            pbar.set_postfix({'loss': f'{running_loss/len(train_loader):.4f}'})

        # validation
        if val_loader is not None:
            val_acc = evaluate_model(model, val_loader, device)
            print(f" Val Accuracy: {val_acc:.2f}", end = "")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_weights = copy.deepcopy(model.state_dict())
            else:
                print()

    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        print(f"\nTraining completed. Best Val Acc: {best_val_acc:.2f}%")
    else:
        print("\nTraining completed")

    torch.save(best_model_weights, save_path)
    print(f"     ✅ Best model saved (acc: {best_val_acc:.2f}%)")

    return model

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

    accuracy = 100 * correct / total
    return accuracy