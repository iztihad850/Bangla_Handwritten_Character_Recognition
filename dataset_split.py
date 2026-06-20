import os
import shutil
import random

source_dir = "../BanglaLekha-Isolated/Images"
output_dir = "../BanglaLekha_8fold"

num_folds = 8
test_ratio = 0.20

random.seed(42)

for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)

    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    # -----------------------------
    # Separate 20% Test Data
    # -----------------------------
    test_size = int(len(images) * test_ratio)

    test_images = images[:test_size]
    cv_images = images[test_size:]

    # -----------------------------
    # Save Test Data
    # -----------------------------
    test_class_dir = os.path.join(
        output_dir,
        "test",
        class_name
    )

    os.makedirs(test_class_dir, exist_ok=True)

    for img in test_images:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(test_class_dir, img)
        )

    # -----------------------------
    # Create 8 folds from remaining 80%
    # -----------------------------
    fold_size = len(cv_images) // num_folds
    folds = []

    for i in range(num_folds):
        start = i * fold_size

        if i == num_folds - 1:
            end = len(cv_images)
        else:
            end = (i + 1) * fold_size

        folds.append(cv_images[start:end])

    # -----------------------------
    # Generate fold datasets
    # -----------------------------
    for fold_idx in range(num_folds):

        val_images = folds[fold_idx]

        train_images = []
        for i in range(num_folds):
            if i != fold_idx:
                train_images.extend(folds[i])

        train_class_dir = os.path.join(
            output_dir,
            f"fold_{fold_idx + 1}",
            "train",
            class_name
        )

        val_class_dir = os.path.join(
            output_dir,
            f"fold_{fold_idx + 1}",
            "validation",
            class_name
        )

        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)

        # Copy training images
        for img in train_images:
            shutil.copy(
                os.path.join(class_path, img),
                os.path.join(train_class_dir, img)
            )

        # Copy validation images
        for img in val_images:
            shutil.copy(
                os.path.join(class_path, img),
                os.path.join(val_class_dir, img)
            )

print("20% Test Split + 8-Fold Cross Validation Created ✅")