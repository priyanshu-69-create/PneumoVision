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


# 2. Basic settings

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


# 3. Load the test dataset

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_folder,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    label_mode="binary",
    shuffle=False
)


# 4. Normalize test images

normalization_layer = tf.keras.layers.Rescaling(
    1.0 / 255
)


test_dataset = test_dataset.map(
    lambda images, labels: (
        normalization_layer(images),
        labels
    )
)


# 5. Get one batch

images, labels = next(
    iter(test_dataset)
)


# 6. Load the validated model

model_path = (
    project_folder
    / "models"
    / "xray_classifier_validated.keras"
)

model = tf.keras.models.load_model(
    model_path
)


# 7. Select one X-ray

sample_image = images[0:1]

sample_label = labels[0].numpy()


# 8. Create a new input tensor

input_tensor = tf.keras.Input(
    shape=(224, 224, 1)
)


# 9. Pass the input through
#    the original model layers

x = input_tensor

conv_outputs = []

for layer in model.layers:

    x = layer(x)

    if isinstance(
        layer,
        tf.keras.layers.Conv2D
    ):

        conv_outputs.append(x)


# 10. Create feature extraction model

feature_model = tf.keras.Model(
    inputs=input_tensor,
    outputs=conv_outputs
)


# 11. Generate feature maps

feature_maps = feature_model.predict(
    sample_image,
    verbose=0
)


# 12. Display original X-ray

plt.figure(
    figsize=(5, 5)
)

plt.imshow(
    sample_image[0].numpy().squeeze(),
    cmap="gray"
)

actual = (
    "PNEUMONIA"
    if sample_label == 1
    else "NORMAL"
)

plt.title(
    f"Original X-ray\nActual: {actual}"
)

plt.axis("off")

plt.tight_layout()

plt.show()


# 13. Display feature maps

for layer_number, feature_map in enumerate(
    feature_maps
):

    number_of_features = (
        feature_map.shape[-1]
    )

    number_to_display = min(
        number_of_features,
        8
    )

    plt.figure(
        figsize=(12, 6)
    )

    for feature_number in range(
        number_to_display
    ):

        plt.subplot(
            2,
            4,
            feature_number + 1
        )

        plt.imshow(
            feature_map[
                0,
                :,
                :,
                feature_number
            ],
            cmap="viridis"
        )

        plt.title(
            f"Feature {feature_number + 1}"
        )

        plt.axis("off")


    plt.suptitle(
        f"Conv Layer {layer_number + 1}"
    )

    plt.tight_layout()

    plt.show()


# 14. Step complete

print()
print("Step 36 complete!")