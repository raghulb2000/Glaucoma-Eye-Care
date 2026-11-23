import os
import shutil
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SOURCE_DIR = os.path.join(BASE_DIR, "Dataset", "train")
OUTPUT_DIR = os.path.join(BASE_DIR, "Dataset_Split")

TRAIN_RATIO = 0.80
SEED = 42

random.seed(SEED)

classes = ["glaucoma", "normal"]

# Create output folders
for split in ["train", "val"]:
    for class_name in classes:
        os.makedirs(
            os.path.join(OUTPUT_DIR, split, class_name),
            exist_ok=True
        )

# Split each class separately
for class_name in classes:

    source_class = os.path.join(SOURCE_DIR, class_name)

    images = [
        f for f in os.listdir(source_class)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        )
    ]

    random.shuffle(images)

    split_index = int(len(images) * TRAIN_RATIO)

    train_images = images[:split_index]
    val_images = images[split_index:]

    for image in train_images:
        shutil.copy2(
            os.path.join(source_class, image),
            os.path.join(
                OUTPUT_DIR,
                "train",
                class_name,
                image
            )
        )

    for image in val_images:
        shutil.copy2(
            os.path.join(source_class, image),
            os.path.join(
                OUTPUT_DIR,
                "val",
                class_name,
                image
            )
        )

    print(f"\n{class_name}")
    print("Training:", len(train_images))
    print("Validation:", len(val_images))

print("\nDataset split completed.")
print("Saved to:", OUTPUT_DIR)