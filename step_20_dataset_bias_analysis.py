from pathlib import Path
import numpy as np
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


# 2. Find NORMAL and PNEUMONIA folders

normal_folder = test_folder / "NORMAL"
pneumonia_folder = test_folder / "PNEUMONIA"


normal_images = list(normal_folder.glob("*"))
pneumonia_images = list(pneumonia_folder.glob("*"))


# 3. Function to analyze a class

def analyze_images(image_paths):

    widths = []
    heights = []
    brightness = []
    contrast = []


    for image_path in image_paths:

        try:

            image = Image.open(image_path).convert("L")

            image_array = np.array(image)


            # Image dimensions

            width, height = image.size

            widths.append(width)
            heights.append(height)


            # Average brightness

            brightness.append(
                image_array.mean()
            )


            # Contrast

            contrast.append(
                image_array.std()
            )


        except Exception:

            continue


    return {
        "widths": np.array(widths),
        "heights": np.array(heights),
        "brightness": np.array(brightness),
        "contrast": np.array(contrast)
    }


# 4. Analyze both classes

normal_stats = analyze_images(
    normal_images
)

pneumonia_stats = analyze_images(
    pneumonia_images
)


# 5. Display image counts

print()
print("Dataset Bias Analysis")
print("---------------------")

print()
print(
    f"NORMAL images analyzed: "
    f"{len(normal_stats['widths'])}"
)

print(
    f"PNEUMONIA images analyzed: "
    f"{len(pneumonia_stats['widths'])}"
)


# 6. Compare image dimensions

print()
print("Image Dimensions")
print("----------------")

print(
    f"NORMAL average width: "
    f"{normal_stats['widths'].mean():.2f}"
)

print(
    f"PNEUMONIA average width: "
    f"{pneumonia_stats['widths'].mean():.2f}"
)

print(
    f"NORMAL average height: "
    f"{normal_stats['heights'].mean():.2f}"
)

print(
    f"PNEUMONIA average height: "
    f"{pneumonia_stats['heights'].mean():.2f}"
)


# 7. Compare brightness

print()
print("Brightness")
print("----------")

print(
    f"NORMAL average brightness: "
    f"{normal_stats['brightness'].mean():.2f}"
)

print(
    f"PNEUMONIA average brightness: "
    f"{pneumonia_stats['brightness'].mean():.2f}"
)


# 8. Compare contrast

print()
print("Contrast")
print("--------")

print(
    f"NORMAL average contrast: "
    f"{normal_stats['contrast'].mean():.2f}"
)

print(
    f"PNEUMONIA average contrast: "
    f"{pneumonia_stats['contrast'].mean():.2f}"
)


# 9. Calculate differences

brightness_difference = abs(
    normal_stats["brightness"].mean()
    - pneumonia_stats["brightness"].mean()
)

contrast_difference = abs(
    normal_stats["contrast"].mean()
    - pneumonia_stats["contrast"].mean()
)


# 10. Display differences

print()
print("Differences")
print("-----------")

print(
    f"Brightness difference: "
    f"{brightness_difference:.2f}"
)

print(
    f"Contrast difference: "
    f"{contrast_difference:.2f}"
)


# 11. Step complete

print()
print("Step 20 complete!")