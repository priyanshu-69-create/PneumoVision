import tensorflow as tf
from pathlib import Path
import numpy as np

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# 1. Find our project folder

project_folder = Path(__file__).parent

train_folder = (
    project_folder
    / "data"
    / "raw"
    / "chest_xray"
    / "train"
)


# 2. Find NORMAL and PNEUMONIA folders

normal_folder = train_folder / "NORMAL"
pneumonia_folder = train_folder / "PNEUMONIA"


# 3. Basic settings

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
SEED = 42


# 4. Load NORMAL validation images

normal_validation = tf.keras.utils.image_dataset_from_directory(
    normal_folder,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    labels=None,
    validation_split=VALIDATION_SPLIT,
    subset="validation",
    seed=SEED,
    shuffle=False
)


# 5. Give NORMAL images label 0

normal_validation = normal_validation.map(
    lambda images: (
        images,
        tf.zeros(
            (tf.shape(images)[0], 1),
            dtype=tf.float32
        )
    )
)


# 6. Load PNEUMONIA validation images

pneumonia_validation = tf.keras.utils.image_dataset_from_directory(
    pneumonia_folder,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    labels=None,
    validation_split=VALIDATION_SPLIT,
    subset="validation",
    seed=SEED,
    shuffle=False
)


# 7. Give PNEUMONIA images label 1

pneumonia_validation = pneumonia_validation.map(
    lambda images: (
        images,
        tf.ones(
            (tf.shape(images)[0], 1),
            dtype=tf.float32
        )
    )
)


# 8. Combine both validation datasets

validation_dataset = normal_validation.concatenate(
    pneumonia_validation
)


# 9. Normalize validation images

normalization_layer = tf.keras.layers.Rescaling(
    1.0 / 255
)


validation_dataset = validation_dataset.map(
    lambda images, labels: (
        normalization_layer(images),
        labels
    )
)


# 10. Improve performance

AUTOTUNE = tf.data.AUTOTUNE

validation_dataset = validation_dataset.prefetch(
    AUTOTUNE
)


# 11. Load the validated model

model_path = (
    project_folder
    / "models"
    / "xray_classifier_validated.keras"
)

model = tf.keras.models.load_model(
    model_path
)


# 12. Store actual labels and probabilities

actual_labels = []
probabilities = []


# 13. Make predictions

for images, labels in validation_dataset:

    predictions = model.predict(
        images,
        verbose=0
    )

    actual_labels.extend(
        labels.numpy().flatten()
    )

    probabilities.extend(
        predictions.flatten()
    )


# 14. Convert to NumPy arrays

actual_labels = np.array(
    actual_labels
)

probabilities = np.array(
    probabilities
)


# 15. Check validation class distribution

normal_count = np.sum(
    actual_labels == 0
)

pneumonia_count = np.sum(
    actual_labels == 1
)


print()
print("Validation Dataset")
print("------------------")

print(
    f"NORMAL: {normal_count}"
)

print(
    f"PNEUMONIA: {pneumonia_count}"
)


# 16. Search for the best threshold

thresholds = np.arange(
    0.10,
    0.91,
    0.01
)

best_threshold = 0.5
best_f1 = 0.0


for threshold in thresholds:

    predicted_labels = (
        probabilities >= threshold
    ).astype(int)

    f1 = f1_score(
        actual_labels,
        predicted_labels
    )

    if f1 > best_f1:

        best_f1 = f1
        best_threshold = threshold


# 17. Display selected threshold

print()
print("Threshold Optimization")
print("----------------------")

print(
    f"Best Threshold: "
    f"{best_threshold:.2f}"
)

print(
    f"Validation F1-Score: "
    f"{best_f1:.4f}"
)


# 18. Evaluate selected threshold

validation_predictions = (
    probabilities >= best_threshold
).astype(int)


validation_precision = precision_score(
    actual_labels,
    validation_predictions
)

validation_recall = recall_score(
    actual_labels,
    validation_predictions
)


# 19. Display validation metrics

print()
print("Validation Results")
print("------------------")

print(
    f"Precision: "
    f"{validation_precision:.4f}"
)

print(
    f"Recall: "
    f"{validation_recall:.4f}"
)


# 20. Display confusion matrix

cm = confusion_matrix(
    actual_labels,
    validation_predictions
)


print()
print("Validation Confusion Matrix")
print("---------------------------")

print(cm)


# 21. Step complete

print()
print("Step 33 complete!")