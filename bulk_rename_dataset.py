import os
from pathlib import Path

# Base directory where all actor folders live
BASE_DIR = Path("/home/anas/Celeb_Classifier/data")

if not BASE_DIR.exists():
    raise FileNotFoundError(f"{BASE_DIR} does not exist")

total_files = 0

for actor_folder in sorted(BASE_DIR.iterdir()):
    if not actor_folder.is_dir():
        continue

    actor_name = actor_folder.name.strip().replace(" ", "_")
    images = sorted([f for f in actor_folder.iterdir() if f.is_file()])

    # Create backup folder if needed
    if len(images) == 0:
        print(f"⚠️ No images found in {actor_name}")
        continue

    print(f"📸 Renaming {len(images)} files in {actor_name}...")

    for i, img_path in enumerate(images, start=1):
        ext = img_path.suffix.lower()  # keep original extension (.jpg, .png, etc.)
        new_name = f"{actor_name}_{i:03d}{ext}"  # Actor_Name_001.jpg
        new_path = img_path.with_name(new_name)

        try:
            os.rename(img_path, new_path)
            total_files += 1
        except Exception as e:
            print(f"❌ Failed to rename {img_path.name}: {e}")

print(f"\n✅ Done! Renamed {total_files} files total.")
