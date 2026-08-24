from pathlib import Path
# Get the main project folder
project_folder = Path(__file__).parent

# Create required folders
folders = [
    "data/raw",
    "data/processed",
    "models",
    "notebooks",
    "src"
]

for folder in folders:
    folder_path = project_folder / folder
    folder_path.mkdir(parents=True, exist_ok=True)
    print(f"Created: {folder}")

print()
print("Project structure setup successful!")

#This will work with folders and file locations instead of manually writing paths