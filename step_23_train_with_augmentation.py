import tensorflow as tf
from pathlib import Path


# 1. Find our project folder

project_folder = Path(__file__).parent

train_folder = (
    project_folder
    / "data"
    / "raw"
    / "chest_xray"
    / "train"
)


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


# 4. Create data augmentation

data_augmentation = tf.keras.Sequential([

    tf.keras.layers.RandomRotation(
        0.05
    ),

    tf.keras.layers.RandomZoom(
        0.10
    ),

    tf.keras.layers.RandomTranslation(
        height_factor=0.05,
        width_factor=0.05
    )

])


# 5. Normalize pixel values

normalization_layer = tf.keras.layers.Rescaling(
    1.0 / 255
)


# 6. Apply augmentation and normalization

train_dataset = train_dataset.map(
    lambda images, labels: (
        normalization_layer(
            data_augmentation(
                images,
                training=True
            )
        ),
        labels
    )
)


# 7. Improve data loading performance

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    AUTOTUNE
)


# 8. Create the CNN model

model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(224, 224, 1)
    ),

    # Data augmentation is already applied
    # in the dataset pipeline above.

    # First convolution block

    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),


    # Second convolution block

    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),


    # Third convolution block

    tf.keras.layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),


    # Convert feature maps into one vector

    tf.keras.layers.Flatten(),


    # Fully connected layer

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),


    # Output

    tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )

])


# 9. Compile the model

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# 10. Display model structure

model.summary()


# 11. Train the model

model.fit(
    train_dataset,
    epochs=10
)


# 12. Save the augmented model

model_path = (
    project_folder
    / "models"
    / "xray_classifier_augmented.keras"
)

model.save(model_path)


# 13. Step complete

print()
print("Step 23 complete!")