import os
import shutil
import random

source_dir = "../BanglaLekha-Isolated/Images"
train_dir = "../BanglaLekha_dataset_split/train"
val_dir = "../BanglaLekha_dataset_split/validation"
test_dir = "../BanglaLekha_dataset_split/test"

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

train_split_ratio = 0.7
val_split_ratio = 0.1

for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)

    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    train_split_idx = int(len(images) * train_split_ratio)
    val_split_idx = train_split_idx + int(len(images) * val_split_ratio)

    train_images = images[:train_split_idx]
    val_images = images[train_split_idx:val_split_idx]
    test_images = images[val_split_idx:]

    # Create class folders
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

    # Move files
    for img in train_images:
        shutil.copy(os.path.join(class_path, img),
                    os.path.join(train_dir, class_name, img))
        
    for img in val_images:
        shutil.copy(os.path.join(class_path, img),
                    os.path.join(val_dir, class_name, img))

    for img in test_images:
        shutil.copy(os.path.join(class_path, img),
                    os.path.join(test_dir, class_name, img))

print("Done splitting dataset ✅")