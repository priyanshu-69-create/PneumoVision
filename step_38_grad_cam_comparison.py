import tensorflow as tf
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# 1. Find our project folder

script_folder = Path(__file__).resolve().parent


# 2. Find the test dataset automatically

possible_test_folders = []

for base_folder in [
    script_folder,
    script_folder.parent
]:

    possible_test_folders.extend(
        base_folder.rglob("chest_xray/test")
    )


test_folder = None

for folder in possible_test_folders:

    if (
        (folder / "NORMAL").exists()
        and
        (folder / "PNEUMONIA").exists()
    ):

        test_folder = folder
        break


if test_folder is None:

    raise FileNotFoundError(
        "Could not find the chest_xray/test folder."
    )


print()
print("Test folder found:")
print(test_folder)


# 3. Find the validated model automatically

possible_model_files = []

for base_folder in [
    script_folder,
    script_folder.parent
]:

    possible_model_files.extend(
        base_folder.rglob(
            "xray_classifier_validated.keras"
        )
    )


model_path = None

if len(possible_model_files) > 0:

    model_path = possible_model_files[0]


if model_path is None:

    raise FileNotFoundError(
        "Could not find "
        "xray_classifier_validated.keras"
    )


print()
print("Model found:")
print(model_path)


# 4. Basic settings

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


# 5. Load the test dataset

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_folder,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    label_mode="binary",
    shuffle=False
)


# 6. Normalize test images

normalization_layer = tf.keras.layers.Rescaling(
    1.0 / 255
)


test_dataset = test_dataset.map(
    lambda images, labels: (
        normalization_layer(images),
        labels
    )
)


# 7. Store all test images and labels

all_images = []
all_labels = []


for images, labels in test_dataset:

    all_images.extend(
        images.numpy()
    )

    all_labels.extend(
        labels.numpy().flatten()
    )


all_images = np.array(
    all_images
)

all_labels = np.array(
    all_labels
)


# 8. Load the validated model

model = tf.keras.models.load_model(
    model_path
)


# 9. Make predictions for all test images

probabilities = model.predict(
    all_images,
    batch_size=BATCH_SIZE,
    verbose=0
).flatten()


predicted_labels = (
    probabilities >= 0.5
).astype(int)


# 10. Find prediction types

true_positive_indices = np.where(
    (all_labels == 1) &
    (predicted_labels == 1)
)[0]


false_positive_indices = np.where(
    (all_labels == 0) &
    (predicted_labels == 1)
)[0]


false_negative_indices = np.where(
    (all_labels == 1) &
    (predicted_labels == 0)
)[0]


# 11. Check that all three types exist

print()
print("Prediction Groups")
print("-----------------")

print(
    f"True Positives: "
    f"{len(true_positive_indices)}"
)

print(
    f"False Positives: "
    f"{len(false_positive_indices)}"
)

print(
    f"False Negatives: "
    f"{len(false_negative_indices)}"
)


if len(true_positive_indices) == 0:

    raise ValueError(
        "No True Positive examples found."
    )


if len(false_positive_indices) == 0:

    raise ValueError(
        "No False Positive examples found."
    )


if len(false_negative_indices) == 0:

    raise ValueError(
        "No False Negative examples found."
    )


# 12. Select one example of each

selected_indices = [

    true_positive_indices[0],

    false_positive_indices[0],

    false_negative_indices[0]

]


# 13. Create input tensor

input_tensor = tf.keras.Input(
    shape=(224, 224, 1)
)


# 14. Find the last convolutional layer

last_conv_layer = None

for layer in model.layers:

    if isinstance(
        layer,
        tf.keras.layers.Conv2D
    ):

        last_conv_layer = layer


print()
print(
    "Last Conv Layer:",
    last_conv_layer.name
)


# 15. Rebuild the forward pass

x = input_tensor

last_conv_output = None


for layer in model.layers:

    x = layer(x)

    if layer == last_conv_layer:

        last_conv_output = x


# 16. Create Grad-CAM model

grad_model = tf.keras.Model(
    inputs=input_tensor,
    outputs=[
        last_conv_output,
        x
    ]
)


# 17. Function to create Grad-CAM

def create_grad_cam(image):

    image = image[
        np.newaxis,
        ...
    ]


    with tf.GradientTape() as tape:

        conv_output, prediction = (
            grad_model(
                image,
                training=False
            )
        )

        pneumonia_score = (
            prediction[:, 0]
        )


    # Get gradients

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


    # Resize heatmap

    heatmap = tf.image.resize(
        heatmap[..., tf.newaxis],
        IMAGE_SIZE
    )


    return heatmap.numpy().squeeze()


# 18. Names for our examples

names = [

    "TRUE POSITIVE",

    "FALSE POSITIVE",

    "FALSE NEGATIVE"

]


# 19. Create comparison figure

plt.figure(
    figsize=(15, 12)
)


# 20. Generate Grad-CAM for each example

for position, index in enumerate(
    selected_indices
):

    image = all_images[index]


    heatmap = create_grad_cam(
        image
    )


    actual = (

        "PNEUMONIA"

        if all_labels[index] == 1

        else "NORMAL"

    )


    predicted = (

        "PNEUMONIA"

        if predicted_labels[index] == 1

        else "NORMAL"

    )


    probability = probabilities[index]


    # Original image

    plt.subplot(
        3,
        2,
        position * 2 + 1
    )


    plt.imshow(
        image.squeeze(),
        cmap="gray"
    )


    plt.title(
        f"{names[position]}\n"
        f"Actual: {actual}\n"
        f"Predicted: {predicted}\n"
        f"Pneumonia Probability: "
        f"{probability:.2f}"
    )


    plt.axis("off")


    # Grad-CAM

    plt.subplot(
        3,
        2,
        position * 2 + 2
    )


    plt.imshow(
        image.squeeze(),
        cmap="gray"
    )


    plt.imshow(
        heatmap,
        cmap="jet",
        alpha=0.45
    )


    plt.title(
        "Grad-CAM"
    )


    plt.axis("off")


# 21. Overall title

plt.suptitle(
    "Grad-CAM: Correct vs Incorrect Predictions",
    fontsize=16
)

plt.tight_layout()
plt.show()
# 22. Step complete

print()
print("Step 38 complete!")