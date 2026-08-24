import tensorflow as tf
from pathlib import Path

# 1. Find our project folder

project_folder = Path(__file__).parent

train_folder = project_folder / "data" / "raw" / "chest_xray" / "train"
val_folder = project_folder / "data" / "raw" / "chest_xray" / "val"

# 2. Basic image settings

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


# 5. Normalize the images

normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

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


# 6. Improve data loading performance

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(AUTOTUNE)
val_dataset = val_dataset.prefetch(AUTOTUNE)


# 7. Create the CNN model

model = tf.keras.Sequential([

    # Input image
    tf.keras.layers.Input(shape=(224, 224, 1)),

    # First convolution layer
    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    # Reduce the image size
    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    # Second convolution layer
    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    # Reduce the image size again
    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    # Third convolution layer
    tf.keras.layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    # Reduce the image size again
    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    # Turn the feature maps into one long vector
    tf.keras.layers.Flatten(),

    # Fully connected layer
    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    # Output layer
    tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )
])


# 8. Compile the model

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# 9. Train the model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10
)

model.save("models/xray_classifier.keras")

print()
print("Step 8 complete!")