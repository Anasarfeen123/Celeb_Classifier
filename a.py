import pickle
from pathlib import Path

pkl = Path("/home/anas/Celeb_Classifier/filenames.pkl")
files = pickle.load(open(pkl, "rb"))

# Print any broken paths
print(files)
broken = [f for f in files if "Samuel/" in f]
print(f"Found {len(broken)} wrong paths:")
for b in broken[:5]:
    print(b)
