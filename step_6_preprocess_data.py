import tensorflow as tf
from pathlib import Path

# 1. Find our dataset

project_folder = Path(__file__).parent

train_folder = project_folder / "data" / "raw" / "chest_xray" / "train"
val_folder = project_folder / "data" / "raw" / "chest_xray" / "val"
test_folder = project_folder / "data" / "raw" / "chest_xray" / "test"

# 2. Basic image settings

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# 3. Load the datasets

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_folder,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    label_mode="binary",
    shuffle=True
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    val_folder,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    label_mode="binary",
    shuffle=False
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_folder,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    label_mode="binary",
    shuffle=False
)

# 4. Create a normalization layer

normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

# 5. Apply normalization


train_dataset = train_dataset.map(
    lambda images, labels: (
        normalization_layer(images),
        labels
    )
)

val_dataset = val_dataset.map(
    lambda images, labels: (
        normalization_layer(images),
        labels
    )
)

test_dataset = test_dataset.map(
    lambda images, labels: (
        normalization_layer(images),
        labels
    )
)

# 6. Improve data loading performance

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(AUTOTUNE)
val_dataset = val_dataset.prefetch(AUTOTUNE)
test_dataset = test_dataset.prefetch(AUTOTUNE)

# 7. Check the normalized values

images, labels = next(iter(train_dataset))

print()
print("Preprocessing successful!")
print()

print("Images shape:", images.shape)
print("Labels shape:", labels.shape)

print()
print("Minimum pixel value:", tf.reduce_min(images).numpy())
print("Maximum pixel value:", tf.reduce_max(images).numpy())

print()
print("Class names:", train_dataset.class_names if hasattr(train_dataset, "class_names") else "Loaded successfully")

print()
print("Step 6 complete!")