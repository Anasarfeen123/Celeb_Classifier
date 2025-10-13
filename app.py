import streamlit as st
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import numpy as np
import glob
import pickle
import os

# --- Page setup ---
st.set_page_config(page_title="Celebrity Lookalike", page_icon="🌟", layout="centered")

# --- Styling ---
st.markdown("""
<style>
body {
    background: radial-gradient(circle at top left, #1e1f26, #111);
    color: #eaeaea;
    font-family: 'Poppins', sans-serif;
}
h1 {
    text-align: center;
    font-weight: 600;
    background: linear-gradient(90deg, #ff4b4b, #ffb347);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
h3 {
    text-align: center;
    color: #fafafa;
    font-weight: 500;
}
.stButton>button {
    border-radius: 10px;
    padding: 0.6rem 1.2rem;
    font-weight: 600;
    color: white;
    background: linear-gradient(90deg, #ff4b4b, #ff7f50);
    border: none;
    transition: 0.3s ease;
}
.stButton>button:hover {
    transform: scale(1.05);
    background: linear-gradient(90deg, #ff7f50, #ff4b4b);
    box-shadow: 0 0 15px rgba(255, 123, 123, 0.4);
}
.result-box {
    background: rgba(255,255,255,0.05);
    border-radius: 14px;
    padding: 20px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.3);
    margin-top: 25px;
}
.actor-name {
    font-size: 1.5rem;
    font-weight: 600;
    color: #ffb347;
    text-align: center;
    margin-top: 10px;
}
.caption {
    font-size: 0.9rem;
    color: #bbb;
    text-align: center;
}
.stCameraInput {
    text-align: center !important;
}
</style>
""", unsafe_allow_html=True)

# --- Cache model ---
@st.cache_resource
def load_model():
    return VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

model = load_model()

# --- Load features ---
@st.cache_data
def load_features():
    features = pickle.load(open("features.pkl", "rb"))
    filenames = pickle.load(open("filenames.pkl", "rb"))

    base_dir = "Celeb_Classifier/data"
    fixed_filenames = []

    for f in filenames:
        f = f.replace("\\", "/")
        base = os.path.basename(f)  # e.g., Amrita_Rao.109.jpg

        # Try to infer the folder name from the file
        folder_guess = base.split("_")[0]  # e.g., "Amrita" from "Amrita_Rao.109.jpg"
        expected_path = os.path.join(base_dir, folder_guess, base)

        if os.path.exists(expected_path):
            fixed_filenames.append(expected_path)
        else:
            # If folder_guess path doesn’t exist, search anywhere in /data/
            matches = glob.glob(os.path.join(base_dir, "**", base), recursive=True)
            if matches:
                fixed_filenames.append(matches[0])  # pick the first found file
            else:
                print("Missing:", base)
                fixed_filenames.append(expected_path)  # fallback for debugging

    return features, fixed_filenames


feature_list, filenames = load_features()

# --- Helper funcs ---
def feature_extractor(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = np.expand_dims(image.img_to_array(img), axis=0)
    preprocessed = preprocess_input(img_array)
    return model.predict(preprocessed).flatten()

def save_img(captured_image):
    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", "captured.jpg")
    with open(file_path, "wb") as f:
        f.write(captured_image.getvalue())
    return file_path

# --- App Header ---
st.markdown("<h1>Celebrity Lookalike Finder</h1>", unsafe_allow_html=True)
st.markdown("<h3>Find out which celebrity mirrors your vibe 🎥</h3>", unsafe_allow_html=True)
st.divider()

# --- Camera Input ---
captured_image = st.camera_input("📸 Capture your photo", key="camera")

if captured_image:
    img_path = save_img(captured_image)
    img = Image.open(captured_image)
    
    with st.spinner("Analyzing your features..."):
        uploaded_features = feature_extractor(img_path, model)
        similarity = cosine_similarity([uploaded_features], feature_list)
        index = np.argmax(similarity)
        celeb_name = " ".join(os.path.basename(filenames[index]).split("_")).title()
    
    # --- Display results ---
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.image(img, use_container_width=True, caption="Your Photo")

    with col2:
        st.image(filenames[index], use_container_width=True, caption="Your Celebrity Match")
        st.markdown(f'<div class="actor-name">{celeb_name}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    st.caption("Tip: Try different lighting or angles for better matches!")
else:
    st.info("🎬 Ready when you are! Allow camera access and capture a clear photo for best results.")
