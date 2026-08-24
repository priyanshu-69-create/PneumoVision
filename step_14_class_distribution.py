from pathlib import Path

# 1. Find our project folder

project_folder = Path(__file__).parent

# 2. Find the training dataset

train_folder = (
    project_folder
    / "data"
    / "raw"
    / "chest_xray"
    / "train"
)

# 3. Find the two class folders

normal_folder = train_folder / "NORMAL"
pneumonia_folder = train_folder / "PNEUMONIA"

# 4. Count the images

normal_images = list(normal_folder.glob("*"))
pneumonia_images = list(pneumonia_folder.glob("*"))


normal_count = len(normal_images)
pneumonia_count = len(pneumonia_images)

# 5. Calculate total images

total_images = normal_count + pneumonia_count

# 6. Calculate percentages

normal_percentage = (
    normal_count / total_images
) * 100

pneumonia_percentage = (
    pneumonia_count / total_images
) * 100

# 7. Display the results

print("Training Dataset Distribution")
print("--------------------------------")

print(f"NORMAL images: {normal_count}")
print(f"PNEUMONIA images: {pneumonia_count}")

print()

print(f"Total images: {total_images}")

print()

print(
    f"NORMAL: {normal_percentage:.2f}%"
)

print(
    f"PNEUMONIA: {pneumonia_percentage:.2f}%"
)

# 8. Step complete
print()
print("Step 14 complete!")