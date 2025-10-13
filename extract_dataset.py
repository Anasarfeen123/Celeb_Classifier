import os
from icrawler.builtin import BingImageCrawler
from PIL import Image
from time import sleep
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
    "Selena Gomez", "Shia LaBeouf", "Megan Fox", "James Franco",
    "Mel Gibson", "Vin Diesel", "Tim Allen", "Robin Williams", "Jason Biggs",
    "Seann William Scott", "Jean-Claude Van Damme", "Owen Wilson",
    "Christian Bale", "Peter Jackson", "Sandra Bullock", "Bruce Lee",
    "Zendaya", "Drew Barrymore", "Tom Holland", "Macaulay Culkin",
    "Jack Nicholson", "Bill Murray", "Sigourney Weaver", "Jake Gyllenhaal",
    "Jason Statham", "Jet Li", "Kate Beckinsale", "Rowan Atkinson",
    "Marlon Brando", "Jennifer Lopez", "John Travolta", "Ben Affleck", "Chris Hemsworth", "James McAvoy", "Tom Hiddleston", "Daisy Ridley",
    "Chris Pratt"
]
kollywood_celebrities = [
    "Vijay", "Ajith Kumar", "Suriya", "Dhanush", "Sivakarthikeyan",
    "Kamal Haasan", "Rajinikanth", "Vikram", "Nayanthara", "Trisha Krishnan",
    "Samantha Ruth Prabhu", "Anirudh Ravichander", "Jayam Ravi", "Vijay Sethupathi"
]

# Combine all lists
all_celebrities = hollywood_celebrities + kollywood_celebrities


# === Config ===
output_root = "data"
images_per_celeb = 100
target_size = (64, 64)   # pixel size
os.makedirs(output_root, exist_ok=True)

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
    folder = os.path.join(output_root, name.replace(" ", "_"))
    os.makedirs(folder, exist_ok=True)

    print(f"\n🎥 Downloading images for {name}...")
    crawler = BingImageCrawler(storage={"root_dir": folder})
    crawler.crawl(
        keyword=f"{name} face portrait",
        filters={"type": "photo"},
        max_num=images_per_celeb
    )

    print(f"🪄 Resizing images for {name}...")
    resize_all_images(folder, target_size)
    print(f"✅ Done: {name}")

    sleep(3)  # avoid server spam

print("\n✅ All downloads complete. 64×64 dataset ready in 'data_64/'")