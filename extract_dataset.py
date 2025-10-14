
import os
from icrawler.builtin import BingImageCrawler
from PIL import Image
from time import sleep
from pathlib import Path # Added for robust path handling

# === CELEB LISTS ===
hollywood_celebrities = celebrities = [
    "Johnny Depp", "Arnold Schwarzenegger", "Jim Carrey", "Leonardo DiCaprio",
    "Tom Cruise", "Robert Downey Jr", "Emma Watson", "Daniel Radcliffe",
    "Chris Evans", "Brad Pitt", "Charles Chaplin", "Morgan Freeman",
    "Tom Hanks", "Hugh Jackman", "Matt Damon", "Sylvester Stallone",
    "Will Smith", "Clint Eastwood", "Cameron Diaz", "George Clooney",
    "Steven Spielberg", "Harrison Ford", "Robert De Niro", "Al Pacino",
    "Russell Crowe", "Liam Neeson", "Kate Winslet", "Sean Connery",
    "Mark Wahlberg", "Natalie Portman", "Pierce Brosnan", "Keanu Reeves",
    "Orlando Bloom", "Dwayne Johnson", "Jackie Chan", "Angelina Jolie",
    "Adam Sandler", "Scarlett Johansson", "Heath Ledger", "Anne Hathaway",
    "Daniel Craig", "Jessica Alba", "Ryan Reynolds", "Edward Norton",
    "Keira Knightley", "Christopher Nolan", "Bradley Cooper", "Will Ferrell",
    "Julia Roberts", "Nicolas Cage", "Ian McKellen", "Halle Berry",
    "Bruce Willis", "Samuel L. Jackson", "Ben Stiller", "Tommy Lee Jones",
    "Jack Black", "Antonio Banderas", "Denzel Washington", "Steve Carell",
    "Selena Gomez", "Shia LaBeouf", "Megan Fox", "James Franco"
]
kollywood_celebrities = [
    "Vijay", "Ajith Kumar", "Suriya", "Dhanush", "Sivakarthikeyan",
    "Kamal Haasan", "Rajinikanth", "Vikram", "Nayanthara", "Trisha Krishnan",
    "Samantha Ruth Prabhu", "Anirudh Ravichander", "Jayam Ravi", "Vijay Sethupathi"
]

# Combine all lists
all_celebrities = hollywood_celebrities + kollywood_celebrities


# === Config ===
# output_root will be relative to where this script is run
output_root = Path("data") 
images_per_celeb = 100
target_size = (64, 64)   # pixel size
output_root.mkdir(exist_ok=True) # Use Path.mkdir for safety

def resize_all_images(folder, size):
    """Resize all images in folder to given size in-place."""
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        try:
            img = Image.open(file_path).convert("RGB")
            img = img.resize(size, Image.LANCZOS) # type: ignore
            img.save(file_path, quality=85)
        except Exception:
            # Remove corrupt or non-image files
            os.remove(file_path)

for name in all_celebrities:
    folder = output_root / name.replace(" ", "_") # Use Path objects for joining
    folder.mkdir(exist_ok=True) # Create folder

    print(f"\n🎥 Downloading images for {name}...")
    crawler = BingImageCrawler(storage={"root_dir": str(folder)}) # Convert Path to string for crawler
    crawler.crawl(keyword=f"{name} face portrait", filters={"type":"photo"}, max_num=images_per_celeb)
    print(f"🪄 Resizing images for {name}...")
    resize_all_images(str(folder), target_size) # Convert Path to string for resize_all_images
    print(f"✅ Done: {name}")
    sleep(3)  # avoid server spam    
print("\n✅ All downloads complete. 64×64 dataset ready in 'data/'")