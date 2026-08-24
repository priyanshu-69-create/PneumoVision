import tensorflow as tf
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# 1. Find project folder

project_folder = (
    Path(__file__).resolve().parent
)


# 2. Find weighted model

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


# 3. Find test dataset

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


# 4. Settings

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 32

THRESHOLD = 0.50


# 5. Load test dataset

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


# 6. Normalize

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


# 7. Load model

model = tf.keras.models.load_model(
    model_path
)


# 8. Collect images and labels

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


# 9. Create predictions

all_predictions = (
    all_probabilities >= THRESHOLD
).astype(int)


# 10. Find error types

false_positive_indices = np.where(
    (all_labels == 0)
    &
    (all_predictions == 1)
)[0]


false_negative_indices = np.where(
    (all_labels == 1)
    &
    (all_predictions == 0)
)[0]


true_negative_indices = np.where(
    (all_labels == 0)
    &
    (all_predictions == 0)
)[0]


true_positive_indices = np.where(
    (all_labels == 1)
    &
    (all_predictions == 1)
)[0]


# 11. Print summary

print()

print(
    "WEIGHTED MODEL ERROR ANALYSIS"
)

print(
    "============================="
)

print()

print(
    f"False Positives: "
    f"{len(false_positive_indices)}"
)

print(
    f"False Negatives: "
    f"{len(false_negative_indices)}"
)

print(
    f"True Negatives: "
    f"{len(true_negative_indices)}"
)

print(
    f"True Positives: "
    f"{len(true_positive_indices)}"
)


# 12. Show false positives

print()

print(
    "False Positive Examples"
)

print(
    "------------------------"
)


num_examples = min(
    12,
    len(false_positive_indices)
)


if num_examples > 0:

    fig, axes = plt.subplots(
        3,
        4,
        figsize=(12, 9)
    )


    axes = axes.flatten()


    for i in range(num_examples):

        index = (
            false_positive_indices[i]
        )


        axes[i].imshow(
            all_images[index].squeeze(),
            cmap="gray"
        )


        axes[i].set_title(
            f"Pneumonia Score: "
            f"{all_probabilities[index]:.3f}"
        )


        axes[i].axis("off")


    for i in range(
        num_examples,
        len(axes)
    ):

        axes[i].axis("off")


    plt.suptitle(
        "False Positives: NORMAL → PNEUMONIA"
    )

    plt.tight_layout()

    plt.show()


# 13. Show false negatives

print()

print(
    "False Negative Examples"
)

print(
    "-----------------------"
)


num_examples = min(
    12,
    len(false_negative_indices)
)


if num_examples > 0:

    fig, axes = plt.subplots(
        3,
        4,
        figsize=(12, 9)
    )


    axes = axes.flatten()


    for i in range(num_examples):

        index = (
            false_negative_indices[i]
        )


        axes[i].imshow(
            all_images[index].squeeze(),
            cmap="gray"
        )


        axes[i].set_title(
            f"Pneumonia Score: "
            f"{all_probabilities[index]:.3f}"
        )


        axes[i].axis("off")


    for i in range(
        num_examples,
        len(axes)
    ):

        axes[i].axis("off")


    plt.suptitle(
        "False Negatives: PNEUMONIA → NORMAL"
    )

    plt.tight_layout()

    plt.show()


# 14. Score statistics

print()

print(
    "False Positive Score Statistics"
)

print(
    "-------------------------------"
)


if len(false_positive_indices) > 0:

    fp_scores = (
        all_probabilities[
            false_positive_indices
        ]
    )


    print(
        f"Minimum: "
        f"{fp_scores.min():.4f}"
    )

    print(
        f"Maximum: "
        f"{fp_scores.max():.4f}"
    )

    print(
        f"Average: "
        f"{fp_scores.mean():.4f}"
    )


# 15. Step complete

print()

print(
    "Step 42 complete!"
)