import torch
import torch.nn as nn
import torch.optim as optim
from utils.metrics import calculate_accuracy, calculate_auc
import csv




def train_model(model, train_loader, val_loader, num_epochs=2, learning_rate=1e-4, device='cuda', class_weights=None):
    metrics = []
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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

        # After each epoch: evaluate on validation set
        val_accuracy = calculate_accuracy(model, val_loader, device)
        val_auc = calculate_auc(model, val_loader, device)

        metrics.append({
        "epoch": epoch + 1,
        "loss": running_loss / len(train_loader),
        "val_accuracy": val_accuracy,
        "val_auc": val_auc
        })

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, Val Accuracy: {val_accuracy:.4f}, Val AUC: {val_auc:.4f}" )

        with open("last_run_metrics.csv", mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "loss", "val_accuracy", "val_auc"])
            writer.writeheader()
            writer.writerows(metrics)


    return model
