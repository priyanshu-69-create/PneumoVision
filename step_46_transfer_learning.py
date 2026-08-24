import tensorflow as tf
from pathlib import Path
import numpy as np


# ============================================================
# 1. Find project folder
# ============================================================

project_folder = (
    Path(__file__).resolve().parent
)


# ============================================================
# 2. Find training dataset
# ============================================================

train_folders = list(
    project_folder.rglob("train")
)


train_folder = next(
    (
        folder
        for folder in train_folders
        if (
            folder.is_dir()
            and
            (folder / "NORMAL").exists()
            and
            (folder / "PNEUMONIA").exists()
        )
    ),
    None
)


if train_folder is None:

    raise FileNotFoundError(
        "Could not find training dataset."
    )


# ============================================================
# 3. Settings
# ============================================================

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 32

VALIDATION_SPLIT = 0.20

SEED = 42

EPOCHS = 5


# ============================================================
# 4. Load training dataset
# ============================================================

train_dataset = (
    tf.keras.utils.image_dataset_from_directory(

        train_folder,

        image_size=IMAGE_SIZE,

        batch_size=BATCH_SIZE,

        color_mode="grayscale",

        label_mode="binary",

        validation_split=VALIDATION_SPLIT,

        subset="training",

        seed=SEED,

        shuffle=True
    )
)


# ============================================================
# 5. Load validation dataset
# ============================================================

validation_dataset = (
    tf.keras.utils.image_dataset_from_directory(

        train_folder,

        image_size=IMAGE_SIZE,

        batch_size=BATCH_SIZE,

        color_mode="grayscale",

        label_mode="binary",

        validation_split=VALIDATION_SPLIT,

        subset="validation",

        seed=SEED,

        shuffle=False
    )
)


# ============================================================
# 6. Convert grayscale → RGB
# ============================================================

def grayscale_to_rgb(
    images,
    labels
):

    images = tf.image.grayscale_to_rgb(
        images
    )

    return (
        images,
        labels
    )


train_dataset = (
    train_dataset.map(
        grayscale_to_rgb,
        num_parallel_calls=tf.data.AUTOTUNE
    )
)


validation_dataset = (
    validation_dataset.map(
        grayscale_to_rgb,
        num_parallel_calls=tf.data.AUTOTUNE
    )
)


# ============================================================
# 7. Prepare MobileNetV2 input
# ============================================================

def prepare_images(
    images,
    labels
):

    images = (
        tf.keras.applications
        .mobilenet_v2
        .preprocess_input(
            images
        )
    )

    return (
        images,
        labels
    )


train_dataset = (
    train_dataset.map(
        prepare_images,
        num_parallel_calls=tf.data.AUTOTUNE
    )
)


validation_dataset = (
    validation_dataset.map(
        prepare_images,
        num_parallel_calls=tf.data.AUTOTUNE
    )
)


# ============================================================
# 8. Prefetch
# ============================================================

train_dataset = (
    train_dataset.prefetch(
        tf.data.AUTOTUNE
    )
)


validation_dataset = (
    validation_dataset.prefetch(
        tf.data.AUTOTUNE
    )
)


# ============================================================
# 9. Load pretrained MobileNetV2
# ============================================================

base_model = (
    tf.keras.applications.MobileNetV2(

        input_shape=(
            224,
            224,
            3
        ),

        include_top=False,

        weights="imagenet"
    )
)


# ============================================================
# 10. Freeze pretrained layers
# ============================================================

base_model.trainable = False


# ============================================================
# 11. Build classifier
# ============================================================

inputs = tf.keras.Input(
    shape=(
        224,
        224,
        3
    )
)


x = base_model(
    inputs,
    training=False
)


x = tf.keras.layers.GlobalAveragePooling2D()(
    x
)


x = tf.keras.layers.Dropout(
    0.30
)(
    x
)


outputs = tf.keras.layers.Dense(
    1,
    activation="sigmoid"
)(
    x
)


model = tf.keras.Model(
    inputs,
    outputs
)


# ============================================================
# 12. Compile model
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-4
    ),

    loss="binary_crossentropy",

    metrics=[
        "accuracy"
    ]
)


# ============================================================
# 13. Class weights
# ============================================================

class_weights = {

    0: 1.9107,

    1: 0.6772

}


# ============================================================
# 14. Display model information
# ============================================================

print()

print(
    "TRANSFER LEARNING MODEL"
)

print(
    "======================="
)

print()

print(
    "Base Model: MobileNetV2"
)

print(
    "Pretrained Weights: ImageNet"
)

print(
    "Base Model Frozen: Yes"
)

print(
    f"Input Size: "
    f"{IMAGE_SIZE[0]} × "
    f"{IMAGE_SIZE[1]} × 3"
)

print(
    "Class Weights:"
)

print(
    f"NORMAL: "
    f"{class_weights[0]}"
)

print(
    f"PNEUMONIA: "
    f"{class_weights[1]}"
)


# ============================================================
# 15. Train classifier
# ============================================================

print()

print(
    "Starting transfer learning..."
)

print()


history = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=EPOCHS,

    class_weight=class_weights

)


# ============================================================
# 16. Save model
# ============================================================

output_path = (
    project_folder
    / "models"
    / "xray_classifier_mobilenet.keras"
)


output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)


model.save(
    output_path
)


# ============================================================
# 17. Final results
# ============================================================

print()

print(
    "TRANSFER LEARNING RESULTS"
)

print(
    "========================="
)

print()

print(
    f"Final Training Accuracy: "
    f"{history.history['accuracy'][-1]:.4f}"
)

print(
    f"Final Validation Accuracy: "
    f"{history.history['val_accuracy'][-1]:.4f}"
)

print(
    f"Final Training Loss: "
    f"{history.history['loss'][-1]:.4f}"
)

print(
    f"Final Validation Loss: "
    f"{history.history['val_loss'][-1]:.4f}"
)


# ============================================================
# 18. Save training history
# ============================================================

history_path = (
    project_folder
    / "models"
    / "mobilenet_training_history.npy"
)


np.save(
    history_path,
    history.history,
    allow_pickle=True
)


# ============================================================
# 19. Step complete
# ============================================================

print()

print(
    "Model saved to:"
)

print(
    output_path
)

print()

print(
    "Step 49 complete!"
)