from pathlib import Path

# Location of our dataset
dataset_folder = Path(__file__).parent / "data" / "raw" / "chest_xray"

# Check if the dataset folder exists
if not dataset_folder.exists():
    print("Dataset folder not found!")
    print("Expected location:", dataset_folder)
    exit()

print("Dataset found!")
print()

# Find NORMAL and PNEUMONIA folders
normal_folders = list(dataset_folder.rglob("NORMAL"))
pneumonia_folders = list(dataset_folder.rglob("PNEUMONIA"))

print("NORMAL folders found:", len(normal_folders))
print("PNEUMONIA folders found:", len(pneumonia_folders))
print()

# Count images inside each class
image_extensions = {".jpg", ".jpeg", ".png"}

normal_count = 0

for folder in normal_folders:
    for file in folder.iterdir():
        if file.suffix.lower() in image_extensions:
            normal_count += 1

pneumonia_count = 0

for folder in pneumonia_folders:
    for file in folder.iterdir():
        if file.suffix.lower() in image_extensions:
            pneumonia_count += 1

print("NORMAL images:", normal_count)
print("PNEUMONIA images:", pneumonia_count)
print()

print("Total images:", normal_count + pneumonia_count)

print()
print("Dataset check complete!")
