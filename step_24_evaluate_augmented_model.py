import tensorflow as tf
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report


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


# 5. Improve data loading performance

AUTOTUNE = tf.data.AUTOTUNE

test_dataset = test_dataset.prefetch(
    AUTOTUNE
)


# 6. Load augmented model

model_path = (
    project_folder
    / "models"
    / "xray_classifier_augmented.keras"
)

model = tf.keras.models.load_model(
    model_path
)


# 7. Evaluate the model

test_loss, test_accuracy = model.evaluate(
    test_dataset,
    verbose=1
)


print()
print("Augmented Model Results")
print("-----------------------")

print(
    f"Test Loss: {test_loss:.4f}"
)

print(
    f"Test Accuracy: {test_accuracy * 100:.2f}%"
)


# 8. Get predictions

predictions = model.predict(
    test_dataset,
    verbose=1
)


# 9. Convert probabilities to class predictions

predicted_labels = (
    predictions.flatten() >= 0.5
).astype(int)


# 10. Get actual labels

actual_labels = []

for images, labels in test_dataset:

    actual_labels.extend(
        labels.numpy().astype(int)
    )


actual_labels = tf.concat(
    [
        tf.convert_to_tensor(actual_labels)
    ],
    axis=0
).numpy()


# 11. Create confusion matrix

cm = confusion_matrix(
    actual_labels,
    predicted_labels
)

print()
print("Confusion Matrix:")
print(cm)

# 12. Classification report

print()
print("Classification Report:")

print(
    classification_report(
        actual_labels,
        predicted_labels,
        target_names=[
            "NORMAL",
            "PNEUMONIA"
        ]
    )
)
# 13. Step complete

print()
print("Step 24 complete!")

#Model jumped from 75% accuracy to 90%