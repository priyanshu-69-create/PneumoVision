import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt


# 1. Find our project folder

project_folder = Path(__file__).parent

train_folder = (
    project_folder
    / "data"
    / "raw"
    / "chest_xray"
    / "train"
)


# 2. Basic settings

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2


# 3. Load training dataset

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_folder,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    label_mode="binary",
    validation_split=VALIDATION_SPLIT,
    subset="training",
    seed=42,
    shuffle=True
)


# 4. Load validation dataset

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    train_folder,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    label_mode="binary",
    validation_split=VALIDATION_SPLIT,
    subset="validation",
    seed=42,
    shuffle=False
)


# 5. Create data augmentation

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


# 6. Normalize training images
#    and apply augmentation

normalization_layer = tf.keras.layers.Rescaling(
    1.0 / 255
)


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


# 7. Normalize validation images
#    WITHOUT augmentation

validation_dataset = validation_dataset.map(
    lambda images, labels: (
        normalization_layer(images),
        labels
    )
)


# 8. Improve performance

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    AUTOTUNE
)


# 9. Create the CNN

model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(224, 224, 1)
    ),

    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    tf.keras.layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )

])


# 10. Compile the model

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# 11. Train and save the history

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10
)


# 12. Plot accuracy

plt.figure(figsize=(10, 6))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title(
    "Training vs Validation Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.tight_layout()

plt.show()


# 13. Plot loss

plt.figure(figsize=(10, 6))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title(
    "Training vs Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.tight_layout()

plt.show()


# 14. Save this model

model_path = (
    project_folder
    / "models"
    / "xray_classifier_validated.keras"
)

model.save(model_path)


# 15. Step complete

print()
print("Step 26 complete!")