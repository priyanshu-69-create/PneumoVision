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

# Selected ONLY using validation data

THRESHOLD = 0.40


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


test_dataset = (
    test_dataset.map(
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


test_dataset = (
    test_dataset.map(
        prepare_images,
        num_parallel_calls=tf.data.AUTOTUNE
    )
)


# ============================================================
# 8. Prefetch
# ============================================================

test_dataset = (
    test_dataset.prefetch(
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
# 10. Generate test predictions
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
# 11. Apply locked threshold
# ============================================================

predicted_labels = (
    probabilities >= THRESHOLD
).astype(int)


# ============================================================
# 12. Calculate metrics
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
# 13. Confusion matrix
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
# 14. Display final results
# ============================================================

print()

print(
    "FINAL MOBILENETV2 TEST"
)

print(
    "======================"
)

print()

print(
    f"Model: {model_path.name}"
)

print(
    f"Locked Threshold: "
    f"{THRESHOLD:.2f}"
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
# 15. Confusion matrix
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
# 16. Error analysis
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
# 17. Compare MobileNetV2 @ 0.50
# ============================================================

print()

print(
    "MobileNetV2 @ 0.50"
)

print(
    "------------------"
)

print(
    "Accuracy:  0.8558"
)

print(
    "Precision: 0.8425"
)

print(
    "Recall:    0.9462"
)

print(
    "F1-Score:  0.8913"
)

print(
    "False Positives: 69"
)

print(
    "False Negatives: 21"
)


# ============================================================
# 18. Changes from 0.50
# ============================================================

print()

print(
    "Changes From Threshold 0.50"
)

print(
    "---------------------------"
)

print(
    f"Accuracy Change: "
    f"{accuracy - 0.8558:+.4f}"
)

print(
    f"Precision Change: "
    f"{precision - 0.8425:+.4f}"
)

print(
    f"Recall Change: "
    f"{recall - 0.9462:+.4f}"
)

print(
    f"F1 Change: "
    f"{f1 - 0.8913:+.4f}"
)

print(
    f"False Positive Change: "
    f"{false_positive - 69:+d}"
)

print(
    f"False Negative Change: "
    f"{false_negative - 21:+d}"
)


# ============================================================
# 19. Step complete
# ============================================================

print()

print(
    "Step 52 complete!"
)