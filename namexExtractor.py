import os
import pickle
from pathlib import Path

# Define dataset paths
BASE_DIR = Path(__file__).resolve().parent
data_dir = f'{BASE_DIR}/data'
data_dir_main = BASE_DIR
# Check directory existence
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Directory '{data_dir}' does not exist. Please check the path.")

# Collect all image file paths
filenames = []
actors = sorted(os.listdir(data_dir))  # sort for consistency

for actor in actors:
    actor_path = os.path.join(data_dir, actor)
    if os.path.isdir(actor_path):
        for file in os.listdir(actor_path):
            # Skip hidden files (like .DS_Store or thumbnails)
            if file.startswith('.'):
                continue
            file_path = os.path.join(actor_path, file)
            filenames.append(file_path)
    else:
        print(f"Skipping non-directory item: {actor_path}")

# Save as pickle
pickle_file = os.path.join(data_dir_main, 'filenames.pkl')
with open(pickle_file, 'wb') as f:
    pickle.dump(filenames, f)

print(f"✅ Filenames successfully saved to {pickle_file}")
print(f"📁 Total images found: {len(filenames)}")
