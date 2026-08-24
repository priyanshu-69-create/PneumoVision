import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path

# 1. Find our training dataset

project_folder = Path(__file__).parent

train_folder = project_folder / "data" / "raw" / "chest_xray" / "train"


# 2. Load the training dataset

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_folder,
    image_size=(224, 224),
    batch_size=9,
    color_mode="grayscale",
    label_mode="binary",
    shuffle=True
)

# 3. Get one batch of images

images, labels = next(iter(train_dataset))

# 4. Get the class names

class_names = train_dataset.class_names

print("Class names:", class_names)

# 5. Display the images

plt.figure(figsize=(10, 10))

for i in range(9):
    plt.subplot(3, 3, i + 1)

    # Display the X-ray
    plt.imshow(images[i].numpy().squeeze(), cmap="gray")

    # Convert the label from a TensorFlow value to an integer
    label = int(labels[i].numpy()[0])

    # Put the class name above the image
    plt.title(class_names[label])

    # Remove axis numbers
    plt.axis("off")

# 6. Show everything

plt.tight_layout()
plt.show()