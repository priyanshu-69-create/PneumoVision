import tensorflow as tf
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


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


# 7. Store images, actual labels, and predictions

all_images = []
actual_labels = []
predicted_labels = []
probabilities = []


# 8. Make predictions

for images, labels in test_dataset:

    predictions = model.predict(
        images,
        verbose=0
    )

    all_images.extend(
        images.numpy()
    )

    actual_labels.extend(
        labels.numpy().flatten()
    )

    probabilities.extend(
        predictions.flatten()
    )

    predicted_labels.extend(
        (predictions.flatten() >= 0.5).astype(int)
    )


# 9. Convert everything to NumPy arrays

all_images = np.array(
    all_images
)

actual_labels = np.array(
    actual_labels
)

predicted_labels = np.array(
    predicted_labels
)

probabilities = np.array(
    probabilities
)


# 10. Find the mistakes

false_positive_indices = np.where(
    (actual_labels == 0) &
    (predicted_labels == 1)
)[0]

false_negative_indices = np.where(
    (actual_labels == 1) &
    (predicted_labels == 0)
)[0]


# 11. Display error counts

print()
print("Step 30 - Error Analysis")
print("------------------------")

print(
    f"False Positives: "
    f"{len(false_positive_indices)}"
)

print(
    f"False Negatives: "
    f"{len(false_negative_indices)}"
)


# 12. Display false positives

plt.figure(
    figsize=(12, 8)
)

for position, index in enumerate(
    false_positive_indices[:6]
):

    plt.subplot(
        2,
        3,
        position + 1
    )

    plt.imshow(
        all_images[index].squeeze(),
        cmap="gray"
    )

    plt.title(
        f"Actual: NORMAL\n"
        f"Predicted: PNEUMONIA\n"
        f"Probability: "
        f"{probabilities[index]:.2f}"
    )

    plt.axis("off")


plt.suptitle(
    "False Positives"
)

plt.tight_layout()

plt.show()


# 13. Display false negatives

plt.figure(
    figsize=(6, 6)
)

if len(false_negative_indices) > 0:

    index = false_negative_indices[0]

    plt.imshow(
        all_images[index].squeeze(),
        cmap="gray"
    )

    plt.title(
        f"Actual: PNEUMONIA\n"
        f"Predicted: NORMAL\n"
        f"Probability: "
        f"{probabilities[index]:.2f}"
    )

    plt.axis("off")

else:

    plt.text(
        0.5,
        0.5,
        "No False Negatives",
        ha="center",
        va="center"
    )

    plt.axis("off")


plt.suptitle(
    "False Negative"
)

plt.tight_layout()
plt.show()

# 14. Step complete

print()
print("Step 30 complete!")