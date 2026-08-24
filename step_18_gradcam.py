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


# 5. Load one NORMAL image

normal_images = list(
    (test_folder / "NORMAL").glob("*")
)

image_path = normal_images[0]


# 6. Load and prepare the image

image = tf.keras.utils.load_img(
    image_path,
    target_size=IMAGE_SIZE,
    color_mode="grayscale"
)

image_array = tf.keras.utils.img_to_array(
    image
)

image_array = image_array / 255.0

image_batch = np.expand_dims(
    image_array,
    axis=0
)


# 7. Create a symbolic input

grad_input = tf.keras.Input(
    shape=(224, 224, 1)
)


# 8. Pass the symbolic input through
#    every layer of the original model

x = grad_input

conv_output = None

for layer in model.layers:

    x = layer(x)

    if layer.name == last_conv_layer.name:

        conv_output = x


# 9. Create the Grad-CAM model

grad_model = tf.keras.models.Model(
    inputs=grad_input,
    outputs=[
        conv_output,
        x
    ]
)


# 10. Record the gradients

with tf.GradientTape() as tape:

    conv_outputs, predictions = grad_model(
        image_batch
    )

    prediction = predictions[:, 0]


# 11. Calculate gradients

gradients = tape.gradient(
    prediction,
    conv_outputs
)


# 12. Average gradients across
#     the height and width

pooled_gradients = tf.reduce_mean(
    gradients,
    axis=(0, 1, 2)
)


# 13. Remove the batch dimension

conv_outputs = conv_outputs[0]


# 14. Weight each feature map

heatmap = conv_outputs @ (
    pooled_gradients[..., tf.newaxis]
)

heatmap = tf.squeeze(
    heatmap
)


# 15. Keep only positive values

heatmap = tf.maximum(
    heatmap,
    0
)


# 16. Normalize the heatmap

heatmap = heatmap / (
    tf.reduce_max(heatmap) + 1e-8
)


# 17. Get prediction probability

prediction_value = prediction.numpy()[0]


# 18. Display the original X-ray

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

plt.imshow(
    image_array.squeeze(),
    cmap="gray"
)

plt.title(
    f"Original X-ray\n"
    f"Pneumonia probability: "
    f"{prediction_value:.4f}"
)

plt.axis("off")


# 19. Display the Grad-CAM heatmap

plt.subplot(1, 2, 2)

plt.imshow(
    image_array.squeeze(),
    cmap="gray"
)

plt.imshow(
    heatmap.numpy(),
    alpha=0.5,
    cmap="jet"
)

plt.title(
    "Grad-CAM"
)

plt.axis("off")


plt.tight_layout()

plt.show()


# 20. Step complete

print()
print("Step 18 complete!")

#Note:
# Dark / blue
 #    ↓
#Less influence

#Brighter / warmer
#     ↓
#More influence

# Dark / blue
#     ↓
#Less influence

#Brighter / warmer
#     ↓
#More influence