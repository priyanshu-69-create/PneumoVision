import tensorflow as tf
from pathlib import Path

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
# 7. Evaluate the model

test_loss, test_accuracy = model.evaluate(
    test_dataset
)
# 8. Display the results

print()
print("Test Results")
print("------------")
print(
    f"Test Loss: {test_loss:.4f}"
)

print(
    f"Test Accuracy: {test_accuracy:.4f}"
)
# 9. Step complete
print()
print("Step 27 complete!")