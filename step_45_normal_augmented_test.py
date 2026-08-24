import tensorflow as tf
from pathlib import Path
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# 1. Find project folder
# ============================================================

project_folder = (
    Path(__file__).resolve().parent
)


# ============================================================
# 2. Find NORMAL-augmented model
# ============================================================

model_files = list(
    project_folder.rglob(
        "xray_classifier_normal_augmented.keras"
    )
)


if len(model_files) == 0:

    raise FileNotFoundError(
        "Could not find "
        "xray_classifier_normal_augmented.keras"
    )


model_path = model_files[0]


# ============================================================
# 3. Find test dataset
# ============================================================

test_folders = list(
    project_folder.rglob("test")
)


test_folder = next(
    (
        folder
        for folder in test_folders
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


if test_folder is None:

    raise FileNotFoundError(
        "Could not find test dataset."
    )


# ============================================================
# 4. Settings
# ============================================================

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 32

THRESHOLD = 0.50


# ============================================================
# 5. Load test dataset
# ============================================================

test_dataset = (
    tf.keras.utils.image_dataset_from_directory(

        test_folder,

        image_size=IMAGE_SIZE,

        batch_size=BATCH_SIZE,

        color_mode="grayscale",

        label_mode="binary",

        shuffle=False
    )
)


# ============================================================
# 6. Normalize test images
# ============================================================

normalization_layer = (
    tf.keras.layers.Rescaling(
        1.0 / 255
    )
)


test_dataset = (
    test_dataset.map(
        lambda images, labels: (
            normalization_layer(images),
            labels
        )
    )
)


test_dataset = (
    test_dataset.prefetch(
        tf.data.AUTOTUNE
    )
)


# ============================================================
# 7. Load model
# ============================================================

model = tf.keras.models.load_model(
    model_path
)


# ============================================================
# 8. Generate predictions
# ============================================================

probabilities = []

true_labels = []


for images, labels in test_dataset:

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
# 9. Apply threshold
# ============================================================

predicted_labels = (
    probabilities >= THRESHOLD
).astype(int)


# ============================================================
# 10. Calculate metrics
# ============================================================

accuracy = np.mean(
    predicted_labels == true_labels
)


precision = precision_score(
    true_labels,
    predicted_labels,
    zero_division=0
)


recall = recall_score(
    true_labels,
    predicted_labels,
    zero_division=0
)


f1 = f1_score(
    true_labels,
    predicted_labels,
    zero_division=0
)


# ============================================================
# 11. Confusion matrix
# ============================================================

cm = confusion_matrix(
    true_labels,
    predicted_labels
)


true_negative = cm[0][0]

false_positive = cm[0][1]

false_negative = cm[1][0]

true_positive = cm[1][1]


# ============================================================
# 12. Display results
# ============================================================

print()

print(
    "NORMAL-AUGMENTED MODEL TEST"
)

print(
    "==========================="
)

print()

print(
    f"Model: {model_path.name}"
)

print(
    f"Threshold: {THRESHOLD:.2f}"
)

print()

print(
    f"Accuracy:  {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall:    {recall:.4f}"
)

print(
    f"F1-Score:  {f1:.4f}"
)


# ============================================================
# 13. Confusion matrix
# ============================================================

print()

print(
    "Confusion Matrix"
)

print(
    "----------------"
)

print(cm)


# ============================================================
# 14. Error analysis
# ============================================================

print()

print(
    "Error Analysis"
)

print(
    "--------------"
)

print(
    f"True Negatives:  "
    f"{true_negative}"
)

print(
    f"False Positives: "
    f"{false_positive}"
)

print(
    f"False Negatives: "
    f"{false_negative}"
)

print(
    f"True Positives:  "
    f"{true_positive}"
)


# ============================================================
# 15. Compare against current best
# ============================================================

print()

print(
    "Comparison With Current Best"
)

print(
    "----------------------------"
)

print(
    "Weighted Model @ 0.50:"
)

print(
    "Accuracy:  0.8189"
)

print(
    "Precision: 0.7798"
)

print(
    "Recall:    0.9897"
)

print(
    "F1-Score:  0.8723"
)

print(
    "False Positives: 109"
)

print(
    "False Negatives: 4"
)


# ============================================================
# 16. Changes
# ============================================================

print()

print(
    "Changes"
)

print(
    "-------"
)

print(
    f"Accuracy Change: "
    f"{accuracy - 0.8189:+.4f}"
)

print(
    f"Precision Change: "
    f"{precision - 0.7798:+.4f}"
)

print(
    f"Recall Change: "
    f"{recall - 0.9897:+.4f}"
)

print(
    f"F1 Change: "
    f"{f1 - 0.8723:+.4f}"
)

print(
    f"False Positive Change: "
    f"{false_positive - 109:+d}"
)

print(
    f"False Negative Change: "
    f"{false_negative - 4:+d}"
)


# ============================================================
# 17. Step complete
# ============================================================

print()

print(
    "Step 48 complete!"
)