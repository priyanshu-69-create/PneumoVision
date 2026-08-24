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


# 4. Normalize the images

normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

train_dataset = train_dataset.map(
    lambda images, labels: (
        normalization_layer(images),
        labels
    )
)


# 5. Improve data loading performance

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(AUTOTUNE)


# 6. Create class weights

normal_count = 1341
pneumonia_count = 3875

total_count = normal_count + pneumonia_count

normal_weight = (
    total_count / (2 * normal_count)
)

pneumonia_weight = (
    total_count / (2 * pneumonia_count)
)


class_weights = {
    0: normal_weight,
    1: pneumonia_weight
}


# 7. Display the weights

print("Class Weights")
print("----------------")

print(
    f"NORMAL: {normal_weight:.2f}"
)

print(
    f"PNEUMONIA: {pneumonia_weight:.2f}"
)


# 8. Create the CNN model

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

    tf.keras.layers.Dropout(
        0.5
    ),

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


# 10. Train the model using class weights

history = model.fit(
    train_dataset,
    epochs=10,
    class_weight=class_weights
)


# 11. Save the new model

model_path = (
    project_folder
    / "models"
    / "xray_classifier_weighted.keras"
)

model.save(model_path)


# 12. Step complete

print()
print("Step 15 complete!")