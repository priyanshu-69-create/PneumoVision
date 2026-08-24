import tensorflow as tf
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


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


# 6. Load our trained model

model_path = (
    project_folder
    / "models"
    / "xray_classifier_validated.keras"
)

model = tf.keras.models.load_model(
    model_path
)


# 7. Store actual labels and predictions

actual_labels = []
predicted_labels = []


# 8. Make predictions

for images, labels in test_dataset:

    predictions = model.predict(
        images,
        verbose=0
    )

    predictions = (
        predictions >= 0.5
    ).astype(int)

    actual_labels.extend(
        labels.numpy().flatten()
    )

    predicted_labels.extend(
        predictions.flatten()
    )


# 9. Convert lists to NumPy arrays

actual_labels = np.array(
    actual_labels
)

predicted_labels = np.array(
    predicted_labels
)


# 10. Create confusion matrix

cm = confusion_matrix(
    actual_labels,
    predicted_labels
)


# 11. Display the numbers

print()
print("Confusion Matrix")
print("----------------")

print(cm)


# 12. Display the confusion matrix

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "NORMAL",
        "PNEUMONIA"
    ]
)

display.plot()

plt.title(
    "Confusion Matrix"
)

plt.tight_layout()

plt.show()


# 13. Step complete

print()
print("Step 28 complete!")