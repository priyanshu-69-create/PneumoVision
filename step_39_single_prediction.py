import tensorflow as tf
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# 1. Find our project folder

project_folder = Path(__file__).resolve().parent


# 2. Find the test dataset automatically

test_folders = list(
    project_folder.rglob("chest_xray/test")
)


test_folder = None

for folder in test_folders:

    if (
        (folder / "NORMAL").exists()
        and
        (folder / "PNEUMONIA").exists()
    ):

        test_folder = folder
        break


if test_folder is None:

    raise FileNotFoundError(
        "Could not find chest_xray/test"
    )


# 3. Choose which class to test

class_to_test ="PNEUMONIA"


class_folder = (
    test_folder
    / class_to_test
)


# 4. Find images inside the class folder

image_files = []

for extension in [
    "*.jpeg",
    "*.jpg",
    "*.png"
]:

    image_files.extend(
        class_folder.glob(extension)
    )


if len(image_files) == 0:

    raise FileNotFoundError(
        f"No images found in {class_folder}"
    )


# 5. Select the first image

image_path = image_files[0]


print()
print("Selected X-ray:")
print(image_path)


# 6. Find the trained model

model_files = list(
    project_folder.rglob(
        "xray_classifier_validated.keras"
    )
)


if len(model_files) == 0:

    raise FileNotFoundError(
        "Could not find "
        "xray_classifier_validated.keras"
    )


model_path = model_files[0]


# 7. Load the trained model

model = tf.keras.models.load_model(
    model_path
)


# 8. Load the X-ray

image = tf.keras.utils.load_img(
    image_path,
    color_mode="grayscale",
    target_size=(224, 224)
)


# 9. Convert image to NumPy array

image_array = (
    tf.keras.utils.img_to_array(
        image
    )
)


# 10. Normalize pixel values

image_array = (
    image_array / 255.0
)


# 11. Add batch dimension

image_array = np.expand_dims(
    image_array,
    axis=0
)


# 12. Make prediction

probability = model.predict(
    image_array,
    verbose=0
)[0][0]


# 13. Apply final threshold

threshold = 0.50


if probability >= threshold:

    prediction = "PNEUMONIA"

else:

    prediction = "NORMAL"


# 14. Display result

print()
print("X-ray Prediction")
print("----------------")

print(
    f"Actual Class: {class_to_test}"
)

print(
    f"Prediction: {prediction}"
)

print(
    f"Pneumonia Probability: "
    f"{probability:.4f}"
)

print(
    f"Threshold: {threshold:.2f}"
)


# 15. Display the X-ray

plt.figure(
    figsize=(6, 6)
)

plt.imshow(
    image_array[0].squeeze(),
    cmap="gray"
)

plt.title(
    f"Actual: {class_to_test}\n"
    f"Prediction: {prediction}\n"
    f"Pneumonia Probability: "
    f"{probability:.2f}"
)

plt.axis("off")

plt.tight_layout()

plt.show()


# 16. Step complete

print()
print("Step 39 complete!")