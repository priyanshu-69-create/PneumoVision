import tensorflow as tf
from pathlib import Path
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)


# 1. Find our project folder

project_folder = Path(__file__).parent

test_folder = (
    project_folder
    / "data"
    / "raw"
    / "chest_xray"
    / "test"
)


# 2. Basic settings

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

FINAL_THRESHOLD = 0.50


# 3. Load the test dataset

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_folder,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    label_mode="binary",
    shuffle=False
)


# 4. Normalize test images

normalization_layer = tf.keras.layers.Rescaling(
    1.0 / 255
)


test_dataset = test_dataset.map(
    lambda images, labels: (
        normalization_layer(images),
        labels
    )
)


# 5. Improve performance

AUTOTUNE = tf.data.AUTOTUNE

test_dataset = test_dataset.prefetch(
    AUTOTUNE
)


# 6. Load the final validated model

model_path = (
    project_folder
    / "models"
    / "xray_classifier_validated.keras"
)

model = tf.keras.models.load_model(
    model_path
)


# 7. Store actual labels and probabilities

actual_labels = []
probabilities = []


# 8. Make predictions

for images, labels in test_dataset:

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


# 9. Convert to NumPy arrays

actual_labels = np.array(
    actual_labels
)

probabilities = np.array(
    probabilities
)


# 10. Apply final threshold

predicted_labels = (
    probabilities >= FINAL_THRESHOLD
).astype(int)


# 11. Calculate final metrics

accuracy = accuracy_score(
    actual_labels,
    predicted_labels
)

precision = precision_score(
    actual_labels,
    predicted_labels
)

recall = recall_score(
    actual_labels,
    predicted_labels
)

f1 = f1_score(
    actual_labels,
    predicted_labels
)

auc = roc_auc_score(
    actual_labels,
    probabilities
)

average_precision = (
    average_precision_score(
        actual_labels,
        probabilities
    )
)


# 12. Create final confusion matrix

cm = confusion_matrix(
    actual_labels,
    predicted_labels
)


# 13. Display final summary

print()
print("FINAL MODEL SUMMARY")
print("===================")

print()
print(
    "Model: "
    "xray_classifier_validated.keras"
)

print(
    f"Threshold: {FINAL_THRESHOLD:.2f}"
)

print()

print(
    f"Accuracy:           {accuracy:.4f}"
)

print(
    f"Precision:          {precision:.4f}"
)

print(
    f"Recall:             {recall:.4f}"
)

print(
    f"F1-Score:           {f1:.4f}"
)

print(
    f"ROC AUC:            {auc:.4f}"
)

print(
    f"Average Precision:  {average_precision:.4f}"
)

print()

print("Confusion Matrix")
print("----------------")

print(cm)


# 14. Save the summary

summary_path = (
    project_folder
    / "models"
    / "final_model_summary.txt"
)


with open(
    summary_path,
    "w"
) as file:

    file.write(
        "FINAL MODEL SUMMARY\n"
    )

    file.write(
        "===================\n\n"
    )

    file.write(
        "Model: "
        "xray_classifier_validated.keras\n"
    )

    file.write(
        f"Threshold: "
        f"{FINAL_THRESHOLD:.2f}\n\n"
    )

    file.write(
        f"Accuracy: "
        f"{accuracy:.4f}\n"
    )

    file.write(
        f"Precision: "
        f"{precision:.4f}\n"
    )

    file.write(
        f"Recall: "
        f"{recall:.4f}\n"
    )

    file.write(
        f"F1-Score: "
        f"{f1:.4f}\n"
    )

    file.write(
        f"ROC AUC: "
        f"{auc:.4f}\n"
    )

    file.write(
        f"Average Precision: "
        f"{average_precision:.4f}\n\n"
    )

    file.write(
        "Confusion Matrix:\n"
    )

    file.write(
        str(cm)
    )


# 15. Display save location

print()
print(
    "Summary saved to:"
)

print(
    summary_path
)


# 16. Step complete

print()
print("Step 35 complete!")