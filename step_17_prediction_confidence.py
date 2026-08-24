import tensorflow as tf
from pathlib import Path
import numpy as np


# 1. Find our project folder

project_folder = Path(__file__).parent

test_folder = (
    project_folder
    / "data"
    / "raw"
    / "chest_xray"
    / "test"
)


# 2. Basic image settings

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


# 4. Normalize the images

normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

test_dataset = test_dataset.map(
    lambda images, labels: (
        normalization_layer(images),
        labels
    )
)


# 5. Improve data loading performance

AUTOTUNE = tf.data.AUTOTUNE

test_dataset = test_dataset.prefetch(AUTOTUNE)


# 6. Load the original model

model_path = (
    project_folder
    / "models"
    / "xray_classifier.keras"
)

model = tf.keras.models.load_model(model_path)


# 7. Get predictions

predictions = model.predict(
    test_dataset,
    verbose=0
)


# 8. Get actual labels

actual_classes = np.concatenate([
    labels.numpy()
    for images, labels in test_dataset
]).astype(int)


# 9. Convert predictions into a simple array

probabilities = predictions.flatten()


# 10. Find the most confident predictions

most_confident_indices = np.argsort(
    np.abs(probabilities - 0.5)
)[::-1]


# 11. Display the 10 most confident predictions

print()
print("Most Confident Predictions")
print("---------------------------")

for index in most_confident_indices[:10]:

    probability = probabilities[index]

    actual = (
        "PNEUMONIA"
        if actual_classes[index] == 1
        else "NORMAL"
    )

    predicted = (
        "PNEUMONIA"
        if probability >= 0.5
        else "NORMAL"
    )

    confidence = (
        probability
        if probability >= 0.5
        else 1 - probability
    )

    print(
        f"Probability: {probability:.4f} | "
        f"Confidence: {confidence * 100:.2f}% | "
        f"Actual: {actual} | "
        f"Predicted: {predicted}"
    )


# 12. Find the most uncertain predictions

most_uncertain_indices = np.argsort(
    np.abs(probabilities - 0.5)
)


# 13. Display the 10 most uncertain predictions

print()
print("Most Uncertain Predictions")
print("---------------------------")

for index in most_uncertain_indices[:10]:

    probability = probabilities[index]

    actual = (
        "PNEUMONIA"
        if actual_classes[index] == 1
        else "NORMAL"
    )

    predicted = (
        "PNEUMONIA"
        if probability >= 0.5
        else "NORMAL"
    )

    confidence = (
        probability
        if probability >= 0.5
        else 1 - probability
    )

    print(
        f"Probability: {probability:.4f} | "
        f"Confidence: {confidence * 100:.2f}% | "
        f"Actual: {actual} | "
        f"Predicted: {predicted}"
    )


# 14. Step complete

print()
print("Step 17 complete!")