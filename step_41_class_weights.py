import tensorflow as tf
from pathlib import Path
import numpy as np


# 1. Find our project folder

project_folder = Path(
    __file__
).resolve().parent


# 2. Find training folder

train_folder = (
    project_folder
    / "data"
    / "raw"
    / "chest_xray"
    / "train"
)


# 3. Basic settings

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2


# 4. Load training dataset

train_dataset = (
    tf.keras.utils.image_dataset_from_directory(
        train_folder,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        color_mode="grayscale",
        label_mode="binary",
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=42,
        shuffle=True
    )
)


# 5. Load validation dataset

validation_dataset = (
    tf.keras.utils.image_dataset_from_directory(
        train_folder,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        color_mode="grayscale",
        label_mode="binary",
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=42,
        shuffle=False
    )
)


# 6. Show class names

print()
print("Class Names")
print("-----------")

print(
    train_dataset.class_names
)


# 7. Count images in each class

class_counts = np.zeros(
    2,
    dtype=int
)


for _, labels in train_dataset:

    labels = labels.numpy().flatten()

    for label in labels:

        class_counts[
            int(label)
        ] += 1


print()
print("Training Class Counts")
print("----------------------")

print(
    f"NORMAL: "
    f"{class_counts[0]}"
)

print(
    f"PNEUMONIA: "
    f"{class_counts[1]}"
)


# 8. Calculate class weights

total_samples = np.sum(
    class_counts
)

class_weights = {}


for class_index in range(2):

    class_weights[class_index] = (
        total_samples
        /
        (
            2
            * class_counts[class_index]
        )
    )


# 9. Display class weights

print()
print("Class Weights")
print("-------------")

print(
    f"NORMAL: "
    f"{class_weights[0]:.4f}"
)

print(
    f"PNEUMONIA: "
    f"{class_weights[1]:.4f}"
)


# 10. Step complete

print()
print("Step 41A complete!")