import os
from collections import Counter
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

def analyze_dataset(data_root):
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "valid")
    
    print(f"Analyzing dataset at {data_root}")
    
    if not os.path.exists(train_dir):
        print("Train dir not found!")
        return
        
    train_dataset = ImageFolder(train_dir)
    val_dataset = ImageFolder(val_dir)
    
    print(f"Total train images: {len(train_dataset)}")
    print(f"Total valid images: {len(val_dataset)}")
    
    train_classes = train_dataset.classes
    val_classes = val_dataset.classes
    print(f"Number of train classes: {len(train_classes)}")
    
    train_counts = Counter([label for _, label in train_dataset.samples])
    val_counts = Counter([label for _, label in val_dataset.samples])
    
    print("\nClass distribution (first 10 classes):")
    for i in range(min(10, len(train_classes))):
        print(f"Class {i} ({train_classes[i]}): Train={train_counts[i]}, Valid={val_counts[i]}")
        
    # Check for imbalance
    min_train = min(train_counts.values())
    max_train = max(train_counts.values())
    print(f"\nMin train samples per class: {min_train}")
    print(f"Max train samples per class: {max_train}")

    min_val = min(val_counts.values())
    max_val = max(val_counts.values())
    print(f"Min val samples per class: {min_val}")
    print(f"Max val samples per class: {max_val}")
    
    # Check image sizes
    print("\nChecking image sizes of first 10 train images:")
    for i in range(10):
        img, _ = train_dataset[i]
        print(f"Image {i} size: {img.size}")

if __name__ == "__main__":
    analyze_dataset(r"E:\AG_Dataset\Evaluation\Classification_Medicinal_Plant")
