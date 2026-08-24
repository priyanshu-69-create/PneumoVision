import tensorflow as tf
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


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
# 3. Find test dataset
# ============================================================

test_folders = list(
    project_folder.rglob("test")
)


test_folder = next(
    (
        folder
        for folder in test_folders
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


if test_folder is None:

    raise FileNotFoundError(
        "Could not find test dataset."
    )


# ============================================================
# 4. Settings
# ============================================================

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 32

THRESHOLD = 0.50

NUM_EXAMPLES = 6


# ============================================================
# 5. Load model
# ============================================================

model = tf.keras.models.load_model(
    model_path
)


# ============================================================
# 6. Find last convolutional layer
# ============================================================

last_conv_layer = None

for layer in model.layers:

    if isinstance(
        layer,
        tf.keras.layers.Conv2D
    ):

        last_conv_layer = layer


if last_conv_layer is None:

    raise ValueError(
        "Could not find a Conv2D layer."
    )


print()

print(
    "Last Convolutional Layer:"
)

print(
    last_conv_layer.name
)


# ============================================================
# 7. Build Grad-CAM model
# ============================================================

input_tensor = tf.keras.Input(
    shape=(224, 224, 1)
)


x = input_tensor

last_conv_output = None


for layer in model.layers:

    x = layer(x)

    if layer == last_conv_layer:

        last_conv_output = x


grad_model = tf.keras.Model(
    inputs=input_tensor,
    outputs=[
        last_conv_output,
        x
    ]
)


# ============================================================
# 8. Grad-CAM function
# ============================================================

def create_grad_cam(image):

    # Add batch dimension

    image_batch = np.expand_dims(
        image,
        axis=0
    )


    # Track gradients

    with tf.GradientTape() as tape:

        conv_output, prediction = (
            grad_model(
                image_batch,
                training=False
            )
        )


        pneumonia_score = (
            prediction[:, 0]
        )


    # Calculate gradients

    gradients = tape.gradient(
        pneumonia_score,
        conv_output
    )


    # Average gradients

    pooled_gradients = (
        tf.reduce_mean(
            gradients,
            axis=(1, 2)
        )
    )


    # Remove batch dimension

    conv_output = conv_output[0]

    pooled_gradients = (
        pooled_gradients[0]
    )


    # Weight feature maps

    weighted_features = (
        conv_output
        * pooled_gradients
    )


    # Combine feature maps

    heatmap = tf.reduce_sum(
        weighted_features,
        axis=-1
    )


    # Keep positive influence

    heatmap = tf.maximum(
        heatmap,
        0
    )


    # Normalize

    max_value = tf.reduce_max(
        heatmap
    )


    heatmap = (
        heatmap
        / (max_value + 1e-8)
    )


    # Resize

    heatmap = tf.image.resize(
        heatmap[..., tf.newaxis],
        IMAGE_SIZE
    )


    return (
        heatmap
        .numpy()
        .squeeze()
    )


# ============================================================
# 9. Load test dataset
# ============================================================

test_dataset = (
    tf.keras.utils.image_dataset_from_directory(

        test_folder,

        image_size=IMAGE_SIZE,

        batch_size=BATCH_SIZE,

        color_mode="grayscale",

        label_mode="binary",

        shuffle=False
    )
)


# ============================================================
# 10. Normalize images
# ============================================================

normalization_layer = (
    tf.keras.layers.Rescaling(
        1.0 / 255
    )
)


test_dataset = (
    test_dataset.map(
        lambda images, labels: (
            normalization_layer(images),
            labels
        )
    )
)


# ============================================================
# 11. Collect predictions
# ============================================================

all_images = []

all_labels = []

all_probabilities = []


for images, labels in test_dataset:

    predictions = model.predict(
        images,
        verbose=0
    )


    all_images.extend(
        images.numpy()
    )


    all_labels.extend(
        labels.numpy().flatten()
    )


    all_probabilities.extend(
        predictions[:, 0]
    )


all_images = np.array(
    all_images
)

all_labels = np.array(
    all_labels
)

all_probabilities = np.array(
    all_probabilities
)


# ============================================================
# 12. Find false positives
# ============================================================

false_positive_indices = np.where(
    (all_labels == 0)
    &
    (all_probabilities >= THRESHOLD)
)[0]


if len(false_positive_indices) == 0:

    print()

    print(
        "No false positives found."
    )

    raise SystemExit


# ============================================================
# 13. Sort by confidence
# ============================================================

sorted_indices = (
    false_positive_indices[
        np.argsort(
            all_probabilities[
                false_positive_indices
            ]
        )[::-1]
    ]
)


selected_indices = (
    sorted_indices[
        :NUM_EXAMPLES
    ]
)


# ============================================================
# 14. Print selected examples
# ============================================================

print()

print(
    "High-Confidence False Positives"
)

print(
    "--------------------------------"
)


for position, index in enumerate(
    selected_indices,
    start=1
):

    print(
        f"{position}. "
        f"Test index: {index} | "
        f"Pneumonia probability: "
        f"{all_probabilities[index]:.4f}"
    )


# ============================================================
# 15. Create Grad-CAM comparison
# ============================================================

fig, axes = plt.subplots(
    NUM_EXAMPLES,
    2,
    figsize=(10, 4 * NUM_EXAMPLES)
)


if NUM_EXAMPLES == 1:

    axes = np.expand_dims(
        axes,
        axis=0
    )


for row, index in enumerate(
    selected_indices
):

    image = all_images[index]

    probability = (
        all_probabilities[index]
    )


    # Generate Grad-CAM

    heatmap = create_grad_cam(
        image
    )


    # --------------------------------------------------------
    # Original X-ray
    # --------------------------------------------------------

    axes[row, 0].imshow(
        image.squeeze(),
        cmap="gray"
    )


    axes[row, 0].set_title(
        "Original X-ray\n"
        f"Actual: NORMAL | "
        f"Predicted: PNEUMONIA\n"
        f"Probability: {probability:.3f}"
    )


    axes[row, 0].axis("off")


    # --------------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------------

    axes[row, 1].imshow(
        image.squeeze(),
        cmap="gray"
    )


    axes[row, 1].imshow(
        heatmap,
        cmap="jet",
        alpha=0.45
    )


    axes[row, 1].set_title(
        "Grad-CAM\n"
        "Where the model focused"
    )


    axes[row, 1].axis("off")


# ============================================================
# 16. Display figure
# ============================================================

plt.suptitle(
    "High-Confidence False Positives — Grad-CAM Analysis",
    fontsize=16
)


plt.tight_layout(
    rect=[
        0,
        0,
        1,
        0.98
    ]
)


plt.show()


# ============================================================
# 17. Step complete
# ============================================================

print()

print(
    "Step 46 complete!"
)