import tensorflow as tf
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


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


# 7. Get the real labels and model predictions

true_labels = []
predicted_labels = []

for images, labels in test_dataset:

    predictions = model.predict(images, verbose=0)

    true_labels.extend(labels.numpy().flatten())
    predicted_labels.extend((predictions >= 0.5).astype(int).flatten())


# 8. Convert lists into NumPy arrays

true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)


# 9. Create the confusion matrix

cm = confusion_matrix(
    true_labels,
    predicted_labels
)


# 10. Print the confusion matrix

print()
print("Confusion Matrix")
print("----------------")
print(cm)


# 11. Display the confusion matrix

class_names = test_dataset.class_names \
    if hasattr(test_dataset, "class_names") \
    else ["NORMAL", "PNEUMONIA"]

plt.figure(figsize=(6, 6))

plt.imshow(cm)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.xticks(
    [0, 1],
    class_names
)

plt.yticks(
    [0, 1],
    class_names
)

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()
plt.show()


# 12. Step complete

print()
print("Step 10 complete!")
# Get one batch of images and labels from the test dataset
images, labels = next(iter(test_dataset))