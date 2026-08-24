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
# 2. Find weighted model
# ============================================================

model_files = list(
    project_folder.rglob(
        "xray_classifier_weighted.keras"
    )
)


if len(model_files) == 0:

    raise FileNotFoundError(
        "Could not find "
        "xray_classifier_weighted.keras"
    )


model_path = model_files[0]


# ============================================================
# 3. Find training dataset
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
# 4. Settings
# ============================================================

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 32

VALIDATION_SPLIT = 0.20

SEED = 42

EPOCHS = 5


# ============================================================
# 5. Load training dataset
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
# 6. Load validation dataset
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
# 7. Print class information
# ============================================================

print()

print(
    "Classes"
)

print(
    "-------"
)

print(
    train_dataset.class_names
)


# ============================================================
# 8. Normalize images
# ============================================================

normalization_layer = (
    tf.keras.layers.Rescaling(
        1.0 / 255
    )
)


# ============================================================
# 9. Create NORMAL-only augmentation
# ============================================================

normal_augmentation = tf.keras.Sequential(

    [

        tf.keras.layers.RandomRotation(
            0.03
        ),

        tf.keras.layers.RandomZoom(
            height_factor=0.05,
            width_factor=0.05
        ),

        tf.keras.layers.RandomTranslation(
            height_factor=0.03,
            width_factor=0.03
        ),

        tf.keras.layers.RandomContrast(
            0.10
        )

    ]
)


# ============================================================
# 10. Augment NORMAL images only
# ============================================================

def augment_normal_images(
    images,
    labels
):

    # Normalize first

    images = (
        normalization_layer(
            images
        )
    )


    # Create augmented version

    augmented_images = (
        normal_augmentation(
            images,
            training=True
        )
    )


    # NORMAL = 0
    #
    # PNEUMONIA = 1
    #
    # Apply augmentation only
    # when label == 0.

    normal_mask = tf.equal(
        labels,
        0
    )


    normal_mask = tf.reshape(
        normal_mask,
        [-1, 1, 1, 1]
    )


    images = tf.where(
        normal_mask,
        augmented_images,
        images
    )


    return (
        images,
        labels
    )


# ============================================================
# 11. Normalize validation images
# ============================================================

def normalize_validation(
    images,
    labels
):

    images = (
        normalization_layer(
            images
        )
    )


    return (
        images,
        labels
    )


# ============================================================
# 12. Apply preprocessing
# ============================================================

train_dataset = (
    train_dataset.map(
        augment_normal_images,
        num_parallel_calls=tf.data.AUTOTUNE
    )
)


validation_dataset = (
    validation_dataset.map(
        normalize_validation,
        num_parallel_calls=tf.data.AUTOTUNE
    )
)


# ============================================================
# 13. Prefetch
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
# 14. Load weighted model
# ============================================================

model = tf.keras.models.load_model(
    model_path
)


# ============================================================
# 15. Compile for gentle fine-tuning
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),

    loss="binary_crossentropy",

    metrics=[
        "accuracy"
    ]
)


# ============================================================
# 16. Class weights
# ============================================================

class_weights = {

    0: 1.9107,

    1: 0.6772

}


print()

print(
    "Fine-tuning weighted model..."
)

print(
    "NORMAL images receive "
    "targeted augmentation."
)

print(
    "Learning rate: 0.00001"
)

print(
    f"Epochs: {EPOCHS}"
)


# ============================================================
# 17. Fine-tune model
# ============================================================

history = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=EPOCHS,

    class_weight=class_weights

)


# ============================================================
# 18. Save improved model
# ============================================================

output_path = (
    project_folder
    / "models"
    / "xray_classifier_normal_augmented.keras"
)


output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)


model.save(
    output_path
)


# ============================================================
# 19. Print final training results
# ============================================================

print()

print(
    "NORMAL-AUGMENTED MODEL"
)

print(
    "======================"
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

print()

print(
    "Saved to:"
)

print(
    output_path
)

print()

print(
    "Step 47 complete!"
)