import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


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

# 7. Get one batch of test images

images, labels = next(iter(test_dataset))

# 8. Make predictions

predictions = model.predict(
    images,
    verbose=0
)

# 9. Display the first 6 images

plt.figure(figsize=(12, 8))

for i in range(6):

    plt.subplot(2, 3, i + 1)

    plt.imshow(
        images[i].numpy().squeeze(),
        cmap="gray"
    )

    actual = "PNEUMONIA" if labels[i].numpy() == 1 else "NORMAL"

    predicted = (
        "PNEUMONIA"
        if predictions[i][0] >= 0.5
        else "NORMAL"
    )

    plt.title(
        f"Actual: {actual}\nPredicted: {predicted}"
    )

    plt.axis("off")


plt.tight_layout()
plt.show()


# 10. Step complete

print()
print("Step 11 complete!")