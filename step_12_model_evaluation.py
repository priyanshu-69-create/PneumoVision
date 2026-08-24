import tensorflow as tf
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


# 1. Find our project folder

project_folder = Path(__file__).parent
test_folder = project_folder / "data" / "raw" / "chest_xray" / "test"

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

# 4. Normalize the test images

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

# 6. Load the trained model

model_path = project_folder / "models" / "xray_classifier.keras"
model = tf.keras.models.load_model(model_path)

# 7. Evaluate the model

loss, accuracy = model.evaluate(
    test_dataset,
    verbose=1
)


# 8. Print accuracy

print()
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")


# 9. Get predictions for the entire test dataset

predictions = model.predict(
    test_dataset,
    verbose=1
)


# 10. Convert probabilities into 0 or 1

predicted_classes = (
    predictions >= 0.5
).astype(int).flatten()


# 11. Get the actual labels

actual_classes = np.concatenate([
    labels.numpy()
    for images, labels in test_dataset
]).astype(int)

# 12. Create confusion matrix

cm = confusion_matrix(
    actual_classes,
    predicted_classes
)
print()
print("Confusion Matrix:")
print(cm)

# 13. Classification report

print()
print("Classification Report:")

print(
    classification_report(
        actual_classes,
        predicted_classes,
        target_names=["NORMAL", "PNEUMONIA"]
    )
)

# 14. Step complete

print()
print("Step 12 complete!")