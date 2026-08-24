import tensorflow as tf
from pathlib import Path
import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)


# ============================================================
# 1. Find project folder
# ============================================================

project_folder = (
    Path(__file__).resolve().parent
)


# ============================================================
# 2. Find MobileNetV2 model
# ============================================================

model_files = list(
    project_folder.rglob(
        "xray_classifier_mobilenet.keras"
    )
)


if len(model_files) == 0:

    raise FileNotFoundError(
        "Could not find "
        "xray_classifier_mobilenet.keras"
    )


model_path = model_files[0]


# ============================================================
# 3. Find training dataset
# ============================================================

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
        "Could not find training dataset."
    )


# ============================================================
# 4. Settings
# ============================================================

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 32

VALIDATION_SPLIT = 0.20

SEED = 42


# ============================================================
# 5. Load validation dataset
# ============================================================

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


# ============================================================
# 6. Convert grayscale → RGB
# ============================================================

def grayscale_to_rgb(
    images,
    labels
):

    images = tf.image.grayscale_to_rgb(
        images
    )

    return (
        images,
        labels
    )


validation_dataset = (
    validation_dataset.map(
        grayscale_to_rgb,
        num_parallel_calls=tf.data.AUTOTUNE
    )
)


# ============================================================
# 7. MobileNetV2 preprocessing
# ============================================================

def prepare_images(
    images,
    labels
):

    images = (
        tf.keras.applications
        .mobilenet_v2
        .preprocess_input(
            images
        )
    )

    return (
        images,
        labels
    )


validation_dataset = (
    validation_dataset.map(
        prepare_images,
        num_parallel_calls=tf.data.AUTOTUNE
    )
)


# ============================================================
# 8. Prefetch
# ============================================================

validation_dataset = (
    validation_dataset.prefetch(
        tf.data.AUTOTUNE
    )
)


# ============================================================
# 9. Load model
# ============================================================

model = tf.keras.models.load_model(
    model_path
)


# ============================================================
# 10. Generate validation predictions
# ============================================================

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


# ============================================================
# 11. Verify both classes exist
# ============================================================

normal_count = np.sum(
    true_labels == 0
)

pneumonia_count = np.sum(
    true_labels == 1
)


print()

print(
    "MobileNetV2 Validation Dataset"
)

print(
    "------------------------------"
)

print(
    f"NORMAL: "
    f"{normal_count}"
)

print(
    f"PNEUMONIA: "
    f"{pneumonia_count}"
)


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


# ============================================================
# 12. Search thresholds
# ============================================================

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


# ============================================================
# 13. Predictions using best threshold
# ============================================================

final_predictions = (
    probabilities >= best_threshold
).astype(int)


# ============================================================
# 14. Calculate metrics
# ============================================================

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


# ============================================================
# 15. Confusion matrix
# ============================================================

cm = confusion_matrix(
    true_labels,
    final_predictions
)


# ============================================================
# 16. Display optimization results
# ============================================================

print()

print(
    "Threshold Optimization"
)

print(
    "----------------------"
)

print(
    f"Best Threshold: "
    f"{best_threshold:.2f}"
)

print(
    f"Validation F1-Score: "
    f"{f1:.4f}"
)


# ============================================================
# 17. Validation results
# ============================================================

print()

print(
    "Validation Results"
)

print(
    "------------------"
)

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


# ============================================================
# 18. Confusion matrix
# ============================================================

print()

print(
    "Validation Confusion Matrix"
)

print(
    "---------------------------"
)

print(cm)


# ============================================================
# 19. Error analysis
# ============================================================

print()

print(
    "Validation Error Analysis"
)

print(
    "-------------------------"
)

print(
    f"False Positives: "
    f"{cm[0][1]}"
)

print(
    f"False Negatives: "
    f"{cm[1][0]}"
)


# ============================================================
# 20. Step complete
# ============================================================

print()

print(
    "Step 51 complete!"
)