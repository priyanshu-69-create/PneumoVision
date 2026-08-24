import tensorflow as tf
from pathlib import Path


# ============================================================
# 1. Find project folder
# ============================================================

project_folder = Path(
    __file__
).resolve().parent


# ============================================================
# 2. Find MobileNetV2 model
# ============================================================

model_files = list(
    project_folder.rglob(
        "xray_classifier_mobilenet.keras"
    )
)


if len(model_files) == 0:
    raise FileNotFoundError(
        "Could not find "
        "xray_classifier_mobilenet.keras"
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
# 7. Convert grayscale → RGB
# ============================================================

def grayscale_to_rgb(images, labels):

    images = tf.image.grayscale_to_rgb(
        images
    )

    return images, labels


train_dataset = train_dataset.map(
    grayscale_to_rgb,
    num_parallel_calls=tf.data.AUTOTUNE
)


validation_dataset = validation_dataset.map(
    grayscale_to_rgb,
    num_parallel_calls=tf.data.AUTOTUNE
)


# ============================================================
# 8. MobileNetV2 preprocessing
# ============================================================

def prepare_images(images, labels):

    images = (
        tf.keras.applications
        .mobilenet_v2
        .preprocess_input(images)
    )

    return images, labels


train_dataset = train_dataset.map(
    prepare_images,
    num_parallel_calls=tf.data.AUTOTUNE
)


validation_dataset = validation_dataset.map(
    prepare_images,
    num_parallel_calls=tf.data.AUTOTUNE
)


# ============================================================
# 9. Prefetch
# ============================================================

train_dataset = train_dataset.prefetch(
    tf.data.AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    tf.data.AUTOTUNE
)


# ============================================================
# 10. Load trained MobileNetV2 model
# ============================================================

model = tf.keras.models.load_model(
    model_path
)


# ============================================================
# 11. Find MobileNetV2 base model
# ============================================================

base_model = None

for layer in model.layers:

    if isinstance(
        layer,
        tf.keras.Model
    ):

        base_model = layer

        break


if base_model is None:

    raise ValueError(
        "Could not find MobileNetV2 "
        "base model."
    )


# ============================================================
# 12. Freeze everything first
# ============================================================

base_model.trainable = True


# ============================================================
# 13. Freeze lower layers
# ============================================================

fine_tune_from = 100


for layer in base_model.layers[
    :fine_tune_from
]:

    layer.trainable = False


# ============================================================
# 14. Keep BatchNormalization frozen
# ============================================================

for layer in base_model.layers:

    if isinstance(
        layer,
        tf.keras.layers.BatchNormalization
    ):

        layer.trainable = False


# ============================================================
# 15. Count trainable layers
# ============================================================

trainable_layers = sum(
    layer.trainable
    for layer in base_model.layers
)


# ============================================================
# 16. Recompile with tiny learning rate
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
# 17. Class weights
# ============================================================

class_weights = {

    0: 1.9107,

    1: 0.6772

}


# ============================================================
# 18. Display information
# ============================================================

print()

print(
    "MOBILENETV2 FINE-TUNING"
)

print(
    "======================="
)

print()

print(
    "Base Model: MobileNetV2"
)

print(
    "Previously trained: YES"
)

print(
    f"Frozen lower layers: "
    f"{fine_tune_from}"
)

print(
    f"Trainable base layers: "
    f"{trainable_layers}"
)

print(
    "BatchNormalization: FROZEN"
)

print(
    "Learning Rate: 0.00001"
)

print(
    f"Epochs: {EPOCHS}"
)

print()

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
# 19. Fine-tune
# ============================================================

print()

print(
    "Starting fine-tuning..."
)

print()


history = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=EPOCHS,

    class_weight=class_weights
)


# ============================================================
# 20. Save fine-tuned model
# ============================================================

output_path = (
    project_folder
    / "models"
    / "xray_classifier_mobilenet_finetuned.keras"
)


output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)


model.save(
    output_path
)


# ============================================================
# 21. Display final results
# ============================================================

print()

print(
    "FINE-TUNED MOBILENETV2 RESULTS"
)

print(
    "=============================="
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
    "Step 53 complete!"
)