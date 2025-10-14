import streamlit as st
from tensorflow.keras.applications import ResNet50 # type: ignore
from tensorflow.keras.applications.resnet50 import preprocess_input # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import numpy as np
import pickle
import os
from pathlib import Path

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
</style>
""", unsafe_allow_html=True)

# Determine the base directory of the app
BASE_DIR = Path(__file__).parent

# --- Cache model ---
@st.cache_resource
def load_model():
    return ResNet50(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))

model = load_model()

# --- Load features ---
@st.cache_data
def load_features():
    features_path = BASE_DIR / "features.pkl"
    filenames_path = BASE_DIR / "filenames.pkl"

    if not features_path.exists():
        st.error(f"Error: {features_path.name} not found. Please ensure features.pkl is in the app's directory.")
        st.stop()
    if not filenames_path.exists():
        st.error(f"Error: {filenames_path.name} not found. Please ensure filenames.pkl is in the app's directory.")
        st.stop()

    features = pickle.load(open(features_path, "rb"))
    filenames = pickle.load(open(filenames_path, "rb"))

    # Convert all filenames to absolute paths relative to BASE_DIR
    # This handles both relative paths (from deployment) and absolute paths (from local)
    fixed_filenames = []
    for f in filenames:
        path = Path(f)
        if path.is_absolute():
            # If it's already absolute, use as is (for local development)
            if path.exists():
                fixed_filenames.append(str(path))
            else:
                # Try to find it relative to BASE_DIR
                rel_path = BASE_DIR / path.name
                if rel_path.exists():
                    fixed_filenames.append(str(rel_path))
                else:
                    # Try data/actor/file structure
                    try:
                        parts = path.parts
                        if "data" in parts:
                            idx = parts.index("data")
                            rel_path = BASE_DIR / Path(*parts[idx:])
                            fixed_filenames.append(str(rel_path))
                        else:
                            fixed_filenames.append(str(path))
                    except:
                        fixed_filenames.append(str(path))
        else:
            # If relative, resolve relative to BASE_DIR
            full_path = BASE_DIR / path
            fixed_filenames.append(str(full_path))

    return features, fixed_filenames

feature_list, filenames = load_features()

# --- Helper funcs ---
def feature_extractor(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = np.expand_dims(image.img_to_array(img), axis=0)
    preprocessed = preprocess_input(img_array)
    return model.predict(preprocessed).flatten()

def save_img(captured_image):
    uploads_dir = BASE_DIR / "uploads"
    uploads_dir.mkdir(exist_ok=True)
    file_path = uploads_dir / "captured.jpg"
    with open(file_path, "wb") as f:
        f.write(captured_image.getvalue())
    return str(file_path)

# --- Additional Styling ---
st.markdown("""
<style>
body {
    background: radial-gradient(circle at 20% 20%, #1f1f2e, #0c0c10);
    color: #eaeaea;
    font-family: 'Poppins', sans-serif;
    overflow-x: hidden;
}
h1, h3 {
    text-align: center;
}
.stButton>button {
    border-radius: 12px;
    padding: 0.65rem 1.3rem;
    font-weight: 600;
    color: white;
    background: linear-gradient(90deg, #ff4b4b, #ff7f50);
    border: none;
    transition: 0.3s ease;
}
.stButton>button:hover {
    transform: scale(1.05);
    background: linear-gradient(90deg, #ff7f50, #ff4b4b);
    box-shadow: 0 0 18px rgba(255, 123, 123, 0.4);
}
.result-box {
    background: rgba(255,255,255,0.05);
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    margin-top: 35px;
    animation: fadeIn 1.2s ease-in-out;
}
.actor-name {
    font-size: 1.6rem;
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
[data-testid="stCameraInput"] {
    background: rgba(255,255,255,0.03);
    border-radius: 18px;
    box-shadow: 0 0 20px rgba(255, 100, 100, 0.15);
    padding: 12px;
    transition: all 0.3s ease;
}
[data-testid="stCameraInput"]:hover {
    box-shadow: 0 0 35px rgba(255, 120, 120, 0.25);
}
[data-testid="stImage"] img {
    border-radius: 14px;
    box-shadow: 0 0 12px rgba(255, 75, 75, 0.4);
    transition: all 0.3s ease;
}
[data-testid="stImage"] img:hover {
    transform: scale(1.03);
    box-shadow: 0 0 20px rgba(255, 123, 123, 0.6);
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)

# --- Hero Section ---
st.markdown("""
    <style>
    .hero {
        text-align: center;
        margin-top: 20px;
        animation: fadeIn 1.2s ease-in-out;
    }
    .hero img {
        width: 130px;
        border-radius: 22px;
        box-shadow: 0 0 28px rgba(255, 75, 75, 0.35);
        margin-bottom: 18px;
        transition: all 0.4s ease;
    }
    .hero img:hover {
        transform: scale(1.08);
        box-shadow: 0 0 40px rgba(255, 123, 123, 0.65);
    }
    .hero h1 {
        font-weight: 700;
        font-size: 2.6rem;
        background: linear-gradient(90deg, #ff4b4b, #ffb347, #ff4b4b);
        background-size: 200% auto;
        animation: gradientMove 5s ease infinite;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .hero h3 {
        color: #e5e5e5;
        font-weight: 400;
        font-size: 1.1rem;
        margin-top: 10px;
        margin-bottom: 4px;
    }
    .hero p {
        color: #ff7f50;
        font-size: 0.9rem;
        margin-top: 8px;
        letter-spacing: 0.6px;
    }
    @keyframes gradientMove {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    </style>

    <div class="hero">
        <img src="https://raw.githubusercontent.com/Anasarfeen123/Celeb_Classifier/main/hc.jpg" alt="Hack Club Logo"/>
        <h1>Celebrity Lookalike Finder</h1>
        <h3>Find out which celebrity mirrors your vibe 🎥</h3>
        <p>Powered by Hack Club</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
<hr style='border: none; border-top: 1px solid rgba(255,255,255,0.15);
            margin: 25px 0; box-shadow: 0 0 10px rgba(255, 100, 100, 0.15);'>
""", unsafe_allow_html=True)

# --- Camera Input ---
captured_image = st.camera_input("📸 Capture your photo", key="camera")

if captured_image:
    img_path = save_img(captured_image)
    img = Image.open(captured_image)
    
    with st.spinner("Analyzing your features..."):
        uploaded_features = feature_extractor(img_path, model)
        similarity = cosine_similarity([uploaded_features], feature_list)
        index = np.argmax(similarity)
        
        # Extract celebrity name from the filename
        celeb_filepath = Path(filenames[index])
        celeb_filename = celeb_filepath.name
        celeb_name_parts = celeb_filename.split('_')
        
        if len(celeb_name_parts) >= 2:
            # Join parts until the numeric index
            celeb_name = " ".join(celeb_name_parts[:-1]).title()
        else:
            # Fallback if name format is unexpected
            celeb_name = celeb_filepath.stem.replace("_", " ").title()

    # --- Display results ---
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.image(img, use_container_width=True, caption="Your Photo")

    with col2:
        # Check if celebrity image exists before displaying
        celeb_img_path = filenames[index]
        if Path(celeb_img_path).exists():
            st.image(celeb_img_path, use_container_width=True, caption="Your Celebrity Match")
        else:
            st.warning(f"Celebrity image not found: {Path(celeb_img_path).name}")
            st.info("The match was found, but the image file is missing from the deployment.")
        
        st.markdown(f'<div class="actor-name">{celeb_name}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    st.caption("Tip: Try different lighting or angles for better matches!")
else:
    st.info("🎬 Ready when you are! Allow camera access and capture a clear photo for best results.")