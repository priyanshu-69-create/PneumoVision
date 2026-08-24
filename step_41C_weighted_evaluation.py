import tensorflow as tf
from pathlib import Path
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)


# 1. Find our project folder

project_folder = (
    Path(__file__).resolve().parent
)


# 2. Find the weighted model

model_files = list(
    project_folder.rglob(
        "xray_classifier_weighted.keras"
    )
)


if len(model_files) == 0:

    raise FileNotFoundError(
        "Could not find xray_classifier_weighted.keras"
    )


model_path = model_files[0]


# 3. Find test folder

test_folders = list(
    project_folder.rglob("test")
)


test_folder = None

for folder in test_folders:

    if (
        folder.is_dir()
        and
        (folder / "NORMAL").exists()
        and
        (folder / "PNEUMONIA").exists()
    ):

        test_folder = folder

        break


if test_folder is None:

    raise FileNotFoundError(
        "Could not find the test dataset."
    )


# 4. Basic settings

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


# 5. Load test dataset

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


# 6. Normalize test images

normalization_layer = (
    tf.keras.layers.Rescaling(
        1.0 / 255
    )
)


test_dataset = test_dataset.map(
    lambda images, labels: (
        normalization_layer(images),
        labels
    )
)


# 7. Improve performance

AUTOTUNE = tf.data.AUTOTUNE

test_dataset = (
    test_dataset.prefetch(
        AUTOTUNE
    )
)


# 8. Load weighted model

model = tf.keras.models.load_model(
    model_path
)


# 9. Make predictions

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


# 10. Decision threshold

threshold = 0.50


predicted_labels = (
    probabilities >= threshold
).astype(int)


# 11. Calculate metrics

accuracy = np.mean(
    predicted_labels
    == true_labels
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


# 12. Confusion matrix

cm = confusion_matrix(
    true_labels,
    predicted_labels
)


# 13. Extract confusion matrix values

true_negative = cm[0][0]

false_positive = cm[0][1]

false_negative = cm[1][0]

true_positive = cm[1][1]


# 14. Display results

print()

print(
    "WEIGHTED MODEL TEST RESULTS"
)

print(
    "============================"
)

print()

print(
    f"Model: "
    f"{model_path.name}"
)

print(
    f"Threshold: "
    f"{threshold:.2f}"
)

print()

print(
    "Accuracy:  "
    f"{accuracy:.4f}"
)

print(
    "Precision: "
    f"{precision:.4f}"
)

print(
    "Recall:    "
    f"{recall:.4f}"
)

print(
    "F1-Score:  "
    f"{f1:.4f}"
)

print()

print(
    "Confusion Matrix"
)

print(
    "----------------"
)

print(cm)

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


# 15. Classification report

print()

print(
    "Classification Report"
)

print(
    "---------------------"
)

print(
    classification_report(
        true_labels,
        predicted_labels,
        target_names=[
            "NORMAL",
            "PNEUMONIA"
        ],
        zero_division=0
    )
)


# 16. Step complete

print()

print(
    "Step 41C complete!"
)