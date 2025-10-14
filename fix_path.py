import pickle
import os
from pathlib import Path

"""
Run this script BEFORE deploying to convert absolute paths to relative paths.
This ensures the app works both locally and in deployment.
"""

# Path to your Celeb_Classifier directory
BASE_DIR = Path("/home/anas/Celeb_Classifier")
data_dir = BASE_DIR / "data"

# Load the existing filenames.pkl
filenames_path = BASE_DIR / "filenames.pkl"

if not filenames_path.exists():
    raise FileNotFoundError(f"{filenames_path} not found!")

print("📂 Loading filenames.pkl...")
with open(filenames_path, "rb") as f:
    absolute_filenames = pickle.load(f)

print(f"Found {len(absolute_filenames)} files")

# Convert absolute paths to relative paths (relative to BASE_DIR)
relative_filenames = []
for abs_path in absolute_filenames:
    path = Path(abs_path)
    try:
        # Get path relative to BASE_DIR
        rel_path = path.relative_to(BASE_DIR)
        relative_filenames.append(str(rel_path))
    except ValueError:
        # If path is not relative to BASE_DIR, try to extract just data/actor/file
        if "data" in path.parts:
            idx = path.parts.index("data")
            rel_path = Path(*path.parts[idx:])
            relative_filenames.append(str(rel_path))
        else:
            print(f"⚠️ Warning: Could not convert {abs_path}")
            relative_filenames.append(str(path))

# Save the updated filenames.pkl with relative paths
output_path = BASE_DIR / "filenames.pkl"
with open(output_path, "wb") as f:
    pickle.dump(relative_filenames, f)

print(f"✅ Updated filenames.pkl with {len(relative_filenames)} relative paths")
print(f"📝 Sample paths:")
for i, path in enumerate(relative_filenames[:3]):
    print(f"  {i+1}. {path}")

print("\n💡 Now your filenames.pkl is ready for deployment!")