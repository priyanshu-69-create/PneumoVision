import tensorflow as tf
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

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

model_path = (
    project_folder
    / "models"
    / "xray_classifier.keras"
)

model = tf.keras.models.load_model(model_path)


# 7. Get predictions for the entire test dataset

predictions = model.predict(
    test_dataset,
    verbose=1
)


# 8. Convert probabilities into classes

predicted_classes = (
    predictions >= 0.5
).astype(int).flatten()


# 9. Get the actual labels

actual_classes = np.concatenate([
    labels.numpy()
    for images, labels in test_dataset
]).astype(int)


# 10. Create the confusion matrix

cm = confusion_matrix(
    actual_classes,
    predicted_classes
)


# 11. Display the confusion matrix

plt.figure(figsize=(7, 6))

plt.imshow(cm)

plt.title("Confusion Matrix")

plt.colorbar()


# 12. Add class names

class_names = [
    "NORMAL",
    "PNEUMONIA"
]

plt.xticks(
    [0, 1],
    class_names
)

plt.yticks(
    [0, 1],
    class_names
)

# 13. Add numbers inside the matrix

for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

# 14. Add axis labels

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

# 15. Show the plot

plt.tight_layout()
plt.show()

# 16. Step complete

print()
print("Step 13 complete!")