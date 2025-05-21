import torch
import torch.nn as nn
import torch.optim as optim
from utils.metrics import calculate_accuracy, calculate_auc
import csv
import copy

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=1e-4, device='cuda', class_weights=None, model_id="run1"):
    metrics = []
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Early stopping setup
    patience = 10
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_weights = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                val_loss = criterion(outputs, labels)
                val_running_loss += val_loss.item()

        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_running_loss / len(val_loader)
        val_accuracy = calculate_accuracy(model, val_loader, device)
        val_auc = calculate_auc(model, val_loader, device)

        metrics.append({
            "epoch": epoch + 1,
            "loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_accuracy": val_accuracy,
            "val_auc": val_auc
        })

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val AUC: {val_auc:.4f}")

        # Early stopping logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            print("✅ Validation loss improved, saving best model.")
        else:
            epochs_without_improvement += 1
            print(f"⚠️ No improvement in val loss for {epochs_without_improvement} epoch(s).")

        if epochs_without_improvement >= patience:
            print(f"⛔ Early stopping triggered after {epoch+1} epochs.")
            break

        # Save metrics
        with open(f"logs/{model_id}.csv", mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "loss", "val_loss", "val_accuracy", "val_auc"])
            writer.writeheader()
            writer.writerows(metrics)

    # Load best model weights
    model.load_state_dict(best_model_weights)
    return model
