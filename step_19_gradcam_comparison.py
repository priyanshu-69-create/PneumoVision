import tensorflow as tf
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# 1. Find our project folder

project_folder = Path(__file__).parent

test_folder = (
    project_folder
    / "data"
    / "raw"
    / "chest_xray"
    / "test"
)


# 2. Basic image settings

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


# 3. Load the trained model

model_path = (
    project_folder
    / "models"
    / "xray_classifier.keras"
)

model = tf.keras.models.load_model(model_path)


# 4. Find the last convolutional layer

last_conv_layer = None

for layer in reversed(model.layers):

    if isinstance(
        layer,
        tf.keras.layers.Conv2D
    ):

        last_conv_layer = layer
        break


print(
    "Last convolutional layer:",
    last_conv_layer.name
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


# 6. Normalize the dataset

normalization_layer = tf.keras.layers.Rescaling(
    1.0 / 255
)

test_dataset = test_dataset.map(
    lambda images, labels: (
        normalization_layer(images),
        labels
    )
)


# 7. Get all images and labels

all_images = []
all_labels = []

for images, labels in test_dataset:

    all_images.append(images.numpy())
    all_labels.append(labels.numpy())


all_images = np.concatenate(all_images)

# IMPORTANT:
# Flatten labels so they have shape (624,)
# instead of (624, 1)

all_labels = np.concatenate(
    all_labels
).astype(int).flatten()


# 8. Get predictions

predictions = model.predict(
    all_images,
    verbose=1
).flatten()


# 9. Convert predictions to classes

predicted_classes = (
    predictions >= 0.5
).astype(int)


# 10. Find four different types of examples

# Correct NORMAL
correct_normal_indices = np.where(
    (all_labels == 0) &
    (predicted_classes == 0)
)[0]

correct_normal = correct_normal_indices[0]


# Wrong NORMAL
wrong_normal_indices = np.where(
    (all_labels == 0) &
    (predicted_classes == 1)
)[0]

wrong_normal = wrong_normal_indices[0]


# Correct PNEUMONIA
correct_pneumonia_indices = np.where(
    (all_labels == 1) &
    (predicted_classes == 1)
)[0]

correct_pneumonia = correct_pneumonia_indices[0]


# Wrong PNEUMONIA
wrong_pneumonia_indices = np.where(
    (all_labels == 1) &
    (predicted_classes == 0)
)[0]

wrong_pneumonia = wrong_pneumonia_indices[0]


# 11. Store the selected examples

selected_indices = [
    correct_normal,
    wrong_normal,
    correct_pneumonia,
    wrong_pneumonia
]


selected_titles = [
    "Correct NORMAL",
    "Wrong NORMAL",
    "Correct PNEUMONIA",
    "Wrong PNEUMONIA"
]


# 12. Create a Grad-CAM model

grad_input = tf.keras.Input(
    shape=(224, 224, 1)
)


x = grad_input

conv_output = None


for layer in model.layers:

    x = layer(x)

    if layer.name == last_conv_layer.name:

        conv_output = x


grad_model = tf.keras.models.Model(
    inputs=grad_input,
    outputs=[
        conv_output,
        x
    ]
)


# 13. Function to create Grad-CAM

def create_gradcam(image):

    image_batch = np.expand_dims(
        image,
        axis=0
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            image_batch
        )

        prediction = predictions[:, 0]


    # Calculate gradients

    gradients = tape.gradient(
        prediction,
        conv_outputs
    )


    # Average gradients

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )


    # Remove batch dimension

    conv_outputs = conv_outputs[0]


    # Weight feature maps

    heatmap = conv_outputs @ (
        pooled_gradients[..., tf.newaxis]
    )

    heatmap = tf.squeeze(
        heatmap
    )


    # Keep positive values

    heatmap = tf.maximum(
        heatmap,
        0
    )


    # Normalize

    heatmap = heatmap / (
        tf.reduce_max(heatmap) + 1e-8
    )


    return (
        heatmap.numpy(),
        prediction.numpy()[0]
    )


# 14. Create comparison figure

plt.figure(
    figsize=(12, 12)
)


# 15. Process all four examples

for position, index in enumerate(
    selected_indices
):

    image = all_images[index]

    actual = all_labels[index]

    predicted = predicted_classes[index]

    probability = predictions[index]


    # Create Grad-CAM

    heatmap, prediction_value = create_gradcam(
        image
    )


    # Original X-ray

    plt.subplot(
        4,
        2,
        position * 2 + 1
    )

    plt.imshow(
        image.squeeze(),
        cmap="gray"
    )

    plt.title(
        f"{selected_titles[position]}\n"
        f"Probability: {probability:.4f}"
    )

    plt.axis("off")


    # Grad-CAM

    plt.subplot(
        4,
        2,
        position * 2 + 2
    )

    plt.imshow(
        image.squeeze(),
        cmap="gray"
    )

    plt.imshow(
        heatmap,
        alpha=0.5,
        cmap="jet"
    )

    plt.title(
        "Grad-CAM"
    )

    plt.axis("off")


# 16. Improve layout

plt.tight_layout()

plt.show()


# 17. Step complete

print()
print("Step 19 complete!")