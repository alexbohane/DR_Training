import torchvision.transforms as T

# Light transforms for common classes
light_transform = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(10),
])

# Heavy transforms for rare classes
heavy_transform = T.Compose([
    T.RandomHorizontalFlip(p=0.7),
    T.RandomRotation(30)
    # T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    # T.RandomResizedCrop(512, scale=(0.6, 1.0)),  # Stronger crop
])

test_transform = T.Compose([
    T.Resize((512, 512)),
])


