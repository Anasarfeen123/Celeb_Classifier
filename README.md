# 🌟 Celebrity Lookalike Finder

An AI-powered web application that finds your **celebrity lookalike** using deep learning and face analysis.

Upload or capture a photo, and the model analyzes your facial features to find the closest matching celebrity from the dataset.

Built with **Streamlit, TensorFlow, and ResNet50**.

---

# 🚀 Features

### 🎥 AI Face Matching

Uses **ResNet50 deep neural network embeddings** to compare facial features and identify similar celebrities.

### 🎯 Face Detection

Automatically detects and crops the face using **OpenCV Haar Cascades** for more accurate results.

### ✨ Image Enhancement

Improves photo quality using:

* CLAHE contrast enhancement
* Noise reduction
* LAB color-space preprocessing

### 🏆 Top Celebrity Matches

Displays the **top matches with similarity scores**.

### 📸 Camera Integration

Capture a photo directly from your webcam inside the app.

### ⚙️ Advanced Settings

Users can toggle:

* Face detection
* Image enhancement
* Multiple match results

---

# 🚀 Live Demo

👉 Try it here:
https://celebritylookalike.streamlit.app/

No installation required — just open the link, allow camera access, and see which celebrity you resemble!

---

# 🧠 How It Works

The system follows this pipeline:

```
User Photo
     ↓
Face Detection (OpenCV)
     ↓
Image Enhancement
     ↓
Feature Extraction (ResNet50)
     ↓
Cosine Similarity Comparison
     ↓
Top Celebrity Matches
```

Each face is converted into a **2048-dimension feature vector** using ResNet50 embeddings.

Similarity between faces is computed using **cosine similarity**.

---

# 📦 Installation

## 1️⃣ Clone the repository

```
git clone https://github.com/anasarfeen123/Celeb_Classifier.git
cd Celeb_Classifier
```

---

## 2️⃣ Install dependencies

```
pip install -r requirements.txt
```

Dependencies include: 

* Streamlit
* TensorFlow
* OpenCV
* scikit-learn
* NumPy
* Pillow
* tqdm

---

# ▶️ Running the App

Start the Streamlit server:

```
streamlit run app.py
```

Then open:

```
http://localhost:8501
```

Allow camera access and capture your photo.

---

# 📂 Project Structure

```
Celeb_Classifier
│
├── app.py                    # Streamlit web application
├── generate_features.py      # Feature extraction pipeline
├── extract_dataset.py        # Download celebrity images
├── bulk_rename_dataset.py    # Rename dataset images
├── namexExtractor.py         # Generate filenames.pkl
├── fix_path.py               # Convert absolute paths to relative
│
├── features.pkl              # Extracted feature embeddings
├── filenames.pkl             # Image paths for dataset
│
├── data/                     # Celebrity image dataset
│   ├── Tom_Cruise/
│   ├── Leonardo_DiCaprio/
│   └── ...
│
├── uploads/                  # User captured images
├── requirements.txt
└── README.md
```

---

# 🧠 Dataset Creation

The dataset is automatically downloaded using Bing Image Search. 

Run:

```
python extract_dataset.py
```

This downloads celebrity images and resizes them.

---

# 🧬 Feature Generation

After downloading the dataset, generate feature embeddings.

Basic command:

```
python generate_features.py --use-face-detection --filter-quality
```

Recommended command:

```
python generate_features.py \
  --use-face-detection \
  --use-augmentation \
  --filter-quality \
  --balance \
  --workers 4
```

This creates:

```
features.pkl
filenames_processed.pkl
```

---

# 📊 Model Details

| Component         | Technology         |
| ----------------- | ------------------ |
| Feature Extractor | ResNet50           |
| Image Processing  | OpenCV             |
| Similarity Metric | Cosine Similarity  |
| Framework         | TensorFlow / Keras |
| Web App           | Streamlit          |

---

# 📸 Tips for Best Results

For the most accurate matches:

• Use **good lighting**
• Face the camera directly
• Avoid sunglasses or hats
• Keep a **neutral expression**

---

# 🧪 Accuracy

Typical results:

| Metric        | Accuracy |
| ------------- | -------- |
| Top-1 Match   | ~70-85%  |
| Top-3 Matches | ~85-95%  |

Accuracy improves with:

* More dataset images
* Face detection enabled
* Data augmentation

---

# 🤝 Contributing

Contributions are welcome!

Steps:

1. Fork the repo
2. Create a feature branch
3. Submit a pull request

---

# 📜 License

MIT License

---

# 👨‍💻 Author

**Anas Arfeen**

Built using deep learning and computer vision to explore **facial similarity detection**.

---

⭐ If you like this project, consider starring the repository.
