import tensorflow as tf
from pathlib import Path

# 1. Set the locations of our dataset

project_folder = Path(__file__).parent

train_folder = project_folder / "data" / "raw" / "chest_xray" / "train"
test_folder = project_folder / "data" / "raw" / "chest_xray" / "test"
val_folder = project_folder / "data" / "raw" / "chest_xray" / "val"

# 2. Settings for our images

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# 3. Load the training dataset

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_folder,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    label_mode="binary",
    shuffle=True
)

# 4. Load the validation dataset

val_dataset = tf.keras.utils.image_dataset_from_directory(
    val_folder,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    label_mode="binary",
    shuffle=False
)

# 5. Load the test dataset

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_folder,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    label_mode="binary",
    shuffle=False
)

# 6. Display information about our datasets


print()
print("Dataset loading successful!")
print()

print("Class names:", train_dataset.class_names)
print()

print("Image size:", IMAGE_SIZE)
print("Batch size:", BATCH_SIZE)
print()

# 7. Look at one batch

images, labels = next(iter(train_dataset))

print("One batch of images:")
print("Images shape:", images.shape)
print("Labels shape:", labels.shape)
