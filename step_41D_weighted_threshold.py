import tensorflow as tf
from pathlib import Path
import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)


# 1. Find our project folder

project_folder = (
    Path(__file__).resolve().parent
)


# 2. Find the training folder

train_folders = list(
    project_folder.rglob("train")
)


train_folder = next(
    (
        folder
        for folder in train_folders
        if (
            folder.is_dir()
            and
            (folder / "NORMAL").exists()
            and
            (folder / "PNEUMONIA").exists()
        )
    ),
    None
)


if train_folder is None:

    raise FileNotFoundError(
        "Could not find the training dataset."
    )


# 3. Find the weighted model

model_files = list(
    project_folder.rglob(
        "xray_classifier_weighted.keras"
    )
)


if len(model_files) == 0:

    raise FileNotFoundError(
        "Could not find "
        "xray_classifier_weighted.keras"
    )


model_path = model_files[0]


# 4. Basic settings

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
SEED = 42


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

        seed=SEED,

        shuffle=True
    )
)


# 6. Verify validation classes

validation_class_names = (
    validation_dataset.class_names
)


print()
print("Validation Classes")
print("-------------------")

print(
    validation_class_names
)


# 7. Normalize validation images

normalization_layer = (
    tf.keras.layers.Rescaling(
        1.0 / 255
    )
)


validation_dataset = (
    validation_dataset.map(
        lambda images, labels: (
            normalization_layer(images),
            labels
        )
    )
)


# 8. Improve performance

AUTOTUNE = tf.data.AUTOTUNE

validation_dataset = (
    validation_dataset.prefetch(
        AUTOTUNE
    )
)


# 9. Load weighted model

model = tf.keras.models.load_model(
    model_path
)


# 10. Generate validation predictions

probabilities = []

true_labels = []


for images, labels in validation_dataset:

    predictions = model.predict(
        images,
        verbose=0
    )

    probabilities.extend(
        predictions[:, 0]
    )

    true_labels.extend(
        labels.numpy().flatten()
    )


probabilities = np.array(
    probabilities
)

true_labels = np.array(
    true_labels
)


# 11. Count validation classes

normal_count = np.sum(
    true_labels == 0
)

pneumonia_count = np.sum(
    true_labels == 1
)


print()
print("Weighted Model Validation Dataset")
print("---------------------------------")

print(
    f"NORMAL: "
    f"{normal_count}"
)

print(
    f"PNEUMONIA: "
    f"{pneumonia_count}"
)


# 12. Safety check

if normal_count == 0:

    raise ValueError(
        "Validation set contains "
        "zero NORMAL images."
    )


if pneumonia_count == 0:

    raise ValueError(
        "Validation set contains "
        "zero PNEUMONIA images."
    )


# 13. Search thresholds

thresholds = np.arange(
    0.10,
    0.91,
    0.01
)


best_threshold = 0.50

best_f1 = -1.0


for threshold in thresholds:

    predicted_labels = (
        probabilities >= threshold
    ).astype(int)


    f1 = f1_score(
        true_labels,
        predicted_labels,
        zero_division=0
    )


    if f1 > best_f1:

        best_f1 = f1

        best_threshold = threshold


# 14. Predictions using best threshold

final_predictions = (
    probabilities >= best_threshold
).astype(int)


# 15. Calculate validation metrics

precision = precision_score(
    true_labels,
    final_predictions,
    zero_division=0
)


recall = recall_score(
    true_labels,
    final_predictions,
    zero_division=0
)


f1 = f1_score(
    true_labels,
    final_predictions,
    zero_division=0
)


# 16. Confusion matrix

cm = confusion_matrix(
    true_labels,
    final_predictions
)


# 17. Display threshold results

print()
print("Threshold Optimization")
print("----------------------")

print(
    f"Best Threshold: "
    f"{best_threshold:.2f}"
)

print(
    f"Validation F1-Score: "
    f"{f1:.4f}"
)


# 18. Display validation results

print()
print("Validation Results")
print("------------------")

print(
    f"Precision: "
    f"{precision:.4f}"
)

print(
    f"Recall:    "
    f"{recall:.4f}"
)

print(
    f"F1-Score:  "
    f"{f1:.4f}"
)


# 19. Display confusion matrix

print()
print("Validation Confusion Matrix")
print("---------------------------")

print(cm)


# 20. Error analysis

print()
print("Validation Error Analysis")
print("-------------------------")

print(
    f"False Positives: "
    f"{cm[0][1]}"
)

print(
    f"False Negatives: "
    f"{cm[1][0]}"
)


# 21. Step complete

print()
print("Step 41D complete!")