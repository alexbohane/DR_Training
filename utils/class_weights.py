from sklearn.utils.class_weight import compute_class_weight
import torch
import numpy as np

def get_class_weights(train_dataset, num_classes=4, device='cuda'):
    # Extract integer labels from dataset (convert tensor → int)
    all_labels = [int(train_dataset[i][1]) for i in range(len(train_dataset))]

    class_labels = np.arange(num_classes)

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=class_labels,
        y=all_labels
    )

    return torch.tensor(class_weights, dtype=torch.float).to(device)
