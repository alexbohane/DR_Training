import torchvision.transforms as T

# Light transforms for common classes
light_transform = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.2),
    T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=4),
    T.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.01),
    T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0))
])

heavy_transform = T.Compose([
    T.RandomHorizontalFlip(p=0.7),
    T.RandomVerticalFlip(p=0.3),
    T.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=8),
    T.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.08, hue=0.015),
    T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0))
])

test_transform = T.Compose([
    T.Resize((600, 600)),
])




# light_transform = T.Compose([
#     T.RandomHorizontalFlip(p=0.5),
#     T.RandomRotation(10),  # Small rotation
#     T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03),
#     T.RandomResizedCrop(600, scale=(0.9, 1.0)),  # Slight random crop/zoom
# ])