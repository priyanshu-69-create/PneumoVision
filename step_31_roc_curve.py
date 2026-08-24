import tensorflow as tf
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    roc_auc_score
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


# 6. Load the validated model

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


# 10. Calculate ROC values

false_positive_rate, true_positive_rate, thresholds = roc_curve(
    actual_labels,
    probabilities
)


# 11. Calculate AUC

auc_score = roc_auc_score(
    actual_labels,
    probabilities
)


# 12. Display AUC

print()
print("ROC Curve Analysis")
print("------------------")

print(
    f"AUC: {auc_score:.4f}"
)


# 13. Plot ROC curve

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    false_positive_rate,
    true_positive_rate,
    label=f"ROC Curve (AUC = {auc_score:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.title(
    "ROC Curve"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.legend()

plt.tight_layout()

plt.show()


# 14. Step complete

print()
print("Step 31 complete!")