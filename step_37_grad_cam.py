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


# 8. Find the last convolutional layer

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


# 9. Create a new input tensor

input_tensor = tf.keras.Input(
    shape=(224, 224, 1)
)


# 10. Rebuild the forward pass

x = input_tensor

last_conv_output = None

for layer in model.layers:

    x = layer(x)

    if layer == last_conv_layer:

        last_conv_output = x


# 11. Create Grad-CAM model

grad_model = tf.keras.Model(
    inputs=input_tensor,
    outputs=[
        last_conv_output,
        x
    ]
)


# 12. Calculate gradients

with tf.GradientTape() as tape:

    conv_output, prediction = (
        grad_model(
            sample_image,
            training=False
        )
    )

    pneumonia_score = prediction[:, 0]


# 13. Get gradients

gradients = tape.gradient(
    pneumonia_score,
    conv_output
)


# 14. Average gradients across
#     the spatial dimensions

pooled_gradients = tf.reduce_mean(
    gradients,
    axis=(1, 2)
)


# 15. Remove batch dimension

conv_output = conv_output[0]

pooled_gradients = pooled_gradients[0]


# 16. Weight each feature map
#     using its gradient

weighted_features = (
    conv_output
    * pooled_gradients
)


# 17. Combine the feature maps

heatmap = tf.reduce_sum(
    weighted_features,
    axis=-1
)


# 18. Keep only positive influence

heatmap = tf.maximum(
    heatmap,
    0
)


# 19. Normalize the heatmap

max_value = tf.reduce_max(
    heatmap
)

heatmap = heatmap / (
    max_value + 1e-8
)


heatmap = heatmap.numpy()


# 20. Get prediction probability

probability = (
    prediction[0, 0].numpy()
)


# 21. Determine labels

actual_label = (
    "PNEUMONIA"
    if sample_label == 1
    else "NORMAL"
)

predicted_label = (
    "PNEUMONIA"
    if probability >= 0.5
    else "NORMAL"
)


# 22. Resize heatmap

heatmap = tf.image.resize(
    heatmap[..., np.newaxis],
    IMAGE_SIZE
).numpy().squeeze()


# 23. Display original X-ray

plt.figure(
    figsize=(6, 6)
)

plt.imshow(
    sample_image[0].numpy().squeeze(),
    cmap="gray"
)

plt.title(
    f"Original X-ray\n"
    f"Actual: {actual_label}\n"
    f"Predicted: {predicted_label}\n"
    f"Pneumonia Probability: "
    f"{probability:.2f}"
)

plt.axis("off")

plt.tight_layout()

plt.show()


# 24. Display Grad-CAM heatmap

plt.figure(
    figsize=(6, 6)
)

plt.imshow(
    sample_image[0].numpy().squeeze(),
    cmap="gray"
)

plt.imshow(
    heatmap,
    cmap="jet",
    alpha=0.45
)

plt.title(
    "Grad-CAM Heatmap"
)

plt.axis("off")
plt.tight_layout()
plt.show()

# 25. Step complete

print()
print("Step 37 complete!")

#Note:
#Bright regions = stronger influence
#Dark regions   = weaker influence
#Important: the heatmap tells us where the model's prediction was influenced,
#not that those regions are medically diagnostic or that the model has identified pneumonia there.