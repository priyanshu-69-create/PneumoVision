from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# 1. Find our project folder

project_folder = Path(__file__).parent

test_folder = (
    project_folder
    / "data"
    / "raw"
    / "chest_xray"
    / "test"
)


# 2. Find image folders

normal_folder = test_folder / "NORMAL"
pneumonia_folder = test_folder / "PNEUMONIA"


normal_images = list(normal_folder.glob("*"))
pneumonia_images = list(pneumonia_folder.glob("*"))


# 3. Function to collect image statistics

def collect_stats(image_paths):

    widths = []
    heights = []
    brightness = []
    contrast = []


    for image_path in image_paths:

        try:

            image = Image.open(image_path).convert("L")

            image_array = np.array(image)


            # Dimensions

            width, height = image.size

            widths.append(width)
            heights.append(height)


            # Brightness

            brightness.append(
                image_array.mean()
            )


            # Contrast

            contrast.append(
                image_array.std()
            )


        except Exception:

            continue


    return (
        np.array(widths),
        np.array(heights),
        np.array(brightness),
        np.array(contrast)
    )


# 4. Collect statistics

(
    normal_widths,
    normal_heights,
    normal_brightness,
    normal_contrast
) = collect_stats(normal_images)


(
    pneumonia_widths,
    pneumonia_heights,
    pneumonia_brightness,
    pneumonia_contrast
) = collect_stats(pneumonia_images)


# 5. Display basic information

print()
print("Step 21 - Dataset Bias Visualization")
print("-------------------------------------")

print()
print(
    f"NORMAL images: {len(normal_brightness)}"
)

print(
    f"PNEUMONIA images: {len(pneumonia_brightness)}"
)


# =========================================================
# 6. Brightness distribution
# =========================================================

plt.figure(figsize=(10, 6))

plt.hist(
    normal_brightness,
    bins=30,
    alpha=0.6,
    label="NORMAL"
)

plt.hist(
    pneumonia_brightness,
    bins=30,
    alpha=0.6,
    label="PNEUMONIA"
)

plt.title("Brightness Distribution")

plt.xlabel("Average Pixel Brightness")

plt.ylabel("Number of Images")

plt.legend()

plt.tight_layout()

plt.show()


# =========================================================
# 7. Contrast distribution
# =========================================================

plt.figure(figsize=(10, 6))

plt.hist(
    normal_contrast,
    bins=30,
    alpha=0.6,
    label="NORMAL"
)

plt.hist(
    pneumonia_contrast,
    bins=30,
    alpha=0.6,
    label="PNEUMONIA"
)

plt.title("Contrast Distribution")

plt.xlabel("Pixel Contrast")

plt.ylabel("Number of Images")

plt.legend()

plt.tight_layout()

plt.show()


# =========================================================
# 8. Image width distribution
# =========================================================

plt.figure(figsize=(10, 6))

plt.hist(
    normal_widths,
    bins=30,
    alpha=0.6,
    label="NORMAL"
)

plt.hist(
    pneumonia_widths,
    bins=30,
    alpha=0.6,
    label="PNEUMONIA"
)

plt.title("Original Image Width Distribution")

plt.xlabel("Image Width (pixels)")

plt.ylabel("Number of Images")

plt.legend()

plt.tight_layout()

plt.show()


# =========================================================
# 9. Image height distribution
# =========================================================

plt.figure(figsize=(10, 6))

plt.hist(
    normal_heights,
    bins=30,
    alpha=0.6,
    label="NORMAL"
)

plt.hist(
    pneumonia_heights,
    bins=30,
    alpha=0.6,
    label="PNEUMONIA"
)

plt.title("Original Image Height Distribution")

plt.xlabel("Image Height (pixels)")

plt.ylabel("Number of Images")

plt.legend()

plt.tight_layout()

plt.show()


# 10. Step complete

print()
print("Step 21 complete!")