import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt


# 1. Find our project folder

project_folder = Path(__file__).parent

train_folder = (
    project_folder
    / "data"
    / "raw"
    / "chest_xray"
    / "train"
)


# 2. Basic image settings

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


# 3. Load the training dataset

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_folder,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    label_mode="binary",
    shuffle=True
)


# 4. Create the data augmentation pipeline

data_augmentation = tf.keras.Sequential([

    tf.keras.layers.RandomRotation(
        0.05
    ),

    tf.keras.layers.RandomZoom(
        0.10
    ),

    tf.keras.layers.RandomTranslation(
        height_factor=0.05,
        width_factor=0.05
    ),

])


# 5. Get one batch of images

images, labels = next(
    iter(train_dataset)
)


# 6. Display original and augmented images

plt.figure(
    figsize=(12, 8)
)


for i in range(6):

    # Original image

    plt.subplot(
        2,
        6,
        i + 1
    )

    plt.imshow(
        images[i].numpy().squeeze(),
        cmap="gray"
    )

    plt.title("Original")

    plt.axis("off")


    # Augmented image

    augmented_image = data_augmentation(
        images[i:i + 1],
        training=True
    )


    plt.subplot(
        2,
        6,
        i + 7
    )

    plt.imshow(
        augmented_image[0].numpy().squeeze(),
        cmap="gray"
    )

    plt.title("Augmented")

    plt.axis("off")


# 7. Improve layout

plt.tight_layout()

plt.show()


# 8. Step complete

print()
print("Step 22 complete!")