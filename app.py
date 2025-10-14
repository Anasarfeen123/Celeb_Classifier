import streamlit as st
from tensorflow.keras.applications import ResNet50 # type: ignore
from tensorflow.keras.applications.resnet50 import preprocess_input # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import numpy as np
import pickle
import cv2
from pathlib import Path

# --- Page setup ---
st.set_page_config(page_title="Celebrity Lookalike", page_icon="🌟", layout="centered")

# Determine the base directory of the app
BASE_DIR = Path(__file__).parent

# --- Cache model and face detector ---
@st.cache_resource
def load_model():
    return ResNet50(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))

@st.cache_resource
def load_face_cascade():
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    return cv2.CascadeClassifier(cascade_path)

model = load_model()
face_cascade = load_face_cascade()

# --- Load features ---
@st.cache_data
def load_features():
    features_path = BASE_DIR / "features.pkl"
    filenames_path = BASE_DIR / "filenames.pkl"

    # Try processed filenames first (from improved feature extraction)
    filenames_processed_path = BASE_DIR / "filenames_processed.pkl"
    if filenames_processed_path.exists():
        filenames_path = filenames_processed_path

    if not features_path.exists():
        st.error(f"Error: {features_path.name} not found. Please ensure features.pkl is in the app's directory.")
        st.stop()
    if not filenames_path.exists():
        st.error(f"Error: {filenames_path.name} not found. Please ensure filenames.pkl is in the app's directory.")
        st.stop()

    features = pickle.load(open(features_path, "rb"))
    filenames = pickle.load(open(filenames_path, "rb"))

    # Convert all filenames to absolute paths relative to BASE_DIR
    fixed_filenames = []
    for f in filenames:
        path = Path(f)
        if path.is_absolute():
            if path.exists():
                fixed_filenames.append(str(path))
            else:
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
            full_path = BASE_DIR / path
            fixed_filenames.append(str(full_path))

    return features, fixed_filenames

feature_list, filenames = load_features()

# --- Helper functions ---
def detect_and_crop_face(img_array):
    """Detect face and return cropped face region."""
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    if len(faces) > 0:
        # Get the largest face
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face
        
        # Add padding (15%)
        padding = int(0.15 * max(w, h))
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img_array.shape[1], x + w + padding)
        y2 = min(img_array.shape[0], y + h + padding)
        
        face_img = img_array[y1:y2, x1:x2]
        return face_img, True
    
    return img_array, False

def enhance_image_quality(img_array):
    """Apply preprocessing to improve image quality."""
    # Convert to LAB color space for better processing
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Merge channels
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    # Reduce noise
    enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
    
    return enhanced

def feature_extractor(img_path, model, use_face_detection=True, use_enhancement=True):
    """Extract features with optional face detection and enhancement."""
    # Load image using OpenCV
    img_cv = cv2.imread(img_path)
    if img_cv is None:
        # Fallback to PIL
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = np.expand_dims(image.img_to_array(img), axis=0)
        preprocessed = preprocess_input(img_array)
        return model.predict(preprocessed).flatten()
    
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    
    # Face detection
    face_detected = False
    if use_face_detection:
        img_cv, face_detected = detect_and_crop_face(img_cv)
    
    # Enhance image quality
    if use_enhancement:
        img_cv = enhance_image_quality(img_cv)
    
    # Resize to target size
    img_resized = cv2.resize(img_cv, (224, 224), interpolation=cv2.INTER_LANCZOS4)
    
    # Preprocess and extract features
    img_array = np.expand_dims(img_resized, axis=0)
    preprocessed = preprocess_input(img_array.astype(np.float32))
    features = model.predict(preprocessed).flatten()
    
    # L2 normalization
    norm = np.linalg.norm(features)
    if norm > 0:
        features /= norm
    
    return features

def save_img(captured_image):
    uploads_dir = BASE_DIR / "uploads"
    uploads_dir.mkdir(exist_ok=True)
    file_path = uploads_dir / "captured.jpg"
    with open(file_path, "wb") as f:
        f.write(captured_image.getvalue())
    return str(file_path)

def get_top_matches(features, feature_list, filenames, top_k=5):
    """Get top K celebrity matches with similarity scores."""
    similarity = cosine_similarity([features], feature_list)[0]
    top_indices = np.argsort(similarity)[-top_k:][::-1]
    
    matches = []
    for idx in top_indices:
        celeb_filepath = Path(filenames[idx])
        celeb_filename = celeb_filepath.name
        celeb_name_parts = celeb_filename.split('_')
        
        if len(celeb_name_parts) >= 2:
            celeb_name = " ".join(celeb_name_parts[:-1]).title()
        else:
            celeb_name = celeb_filepath.stem.replace("_", " ").title()
        
        matches.append({
            'name': celeb_name,
            'path': filenames[idx],
            'similarity': similarity[idx] * 100
        })
    
    return matches

# --- Styling ---
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
.similarity-score {
    font-size: 1.2rem;
    color: #4ade80;
    text-align: center;
    font-weight: 500;
    margin-top: 5px;
}
.caption {
    font-size: 0.9rem;
    color: #bbb;
    text-align: center;
}
.match-card {
    background: rgba(255,255,255,0.03);
    border-radius: 12px;
    padding: 15px;
    margin: 10px 0;
    border: 1px solid rgba(255,255,255,0.1);
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
        <p>Powered by AI with Face Detection</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
<hr style='border: none; border-top: 1px solid rgba(255,255,255,0.15);
            margin: 25px 0; box-shadow: 0 0 10px rgba(255, 100, 100, 0.15);'>
""", unsafe_allow_html=True)

# Settings expander
with st.expander("⚙️ Advanced Settings"):
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        use_face_detection = st.checkbox("🎯 Face Detection", value=True, help="Crop to face region for better accuracy")
        use_enhancement = st.checkbox("✨ Image Enhancement", value=True, help="Apply preprocessing to improve quality")
    with col_s2:
        show_top_matches = st.checkbox("📊 Show Top 3 Matches", value=False, help="Display multiple celebrity matches")
        top_k = 3 if show_top_matches else 1

# --- Camera Input ---
captured_image = st.camera_input("📸 Capture your photo", key="camera")

if captured_image:
    img_path = save_img(captured_image)
    img = Image.open(captured_image)
    
    with st.spinner("🔍 Analyzing your features with AI..."):
        uploaded_features = feature_extractor(
            img_path, 
            model, 
            use_face_detection=use_face_detection,
            use_enhancement=use_enhancement
        )
        
        # Get top matches
        matches = get_top_matches(uploaded_features, feature_list, filenames, top_k=top_k)
        
        # Check if face was detected
        img_cv = cv2.imread(img_path)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        _, face_found = detect_and_crop_face(img_cv)

    # --- Display results ---
    if not face_found and use_face_detection:
        st.warning("⚠️ No face detected! Results may be less accurate. Try better lighting or a clearer angle.")
    
    if show_top_matches:
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.markdown("### Your Photo")
        st.image(img, use_container_width=True)
        st.markdown("---")
        st.markdown("### 🏆 Top Celebrity Matches")
        
        for i, match in enumerate(matches):
            st.markdown(f'<div class="match-card">', unsafe_allow_html=True)
            col1, col2 = st.columns([1, 2])
            
            with col1:
                if Path(match['path']).exists():
                    st.image(match['path'], use_container_width=True)
                else:
                    st.warning("Image not found")
            
            with col2:
                medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                st.markdown(f"#### {medal} Match #{i+1}")
                st.markdown(f'<div class="actor-name">{match["name"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="similarity-score">{match["similarity"]:.1f}% Match</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Single best match display
        best_match = matches[0]
        
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.image(img, use_container_width=True, caption="Your Photo")

        with col2:
            if Path(best_match['path']).exists():
                st.image(best_match['path'], use_container_width=True, caption="Your Celebrity Match")
            else:
                st.warning("Celebrity image not found")
                st.info("The match was found, but the image file is missing.")
            
            st.markdown(f'<div class="actor-name">{best_match["name"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="similarity-score">{best_match["similarity"]:.1f}% Match</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
    
    st.caption("💡 Tip: Try different lighting or angles for better matches!")
    
    # Show confidence message
    if matches[0]['similarity'] > 80:
        st.success("🌟 High confidence match!")
    elif matches[0]['similarity'] > 60:
        st.info("✅ Good match found!")
    else:
        st.warning("⚠️ Low confidence. Try a clearer photo for better results.")
else:
    st.info("🎬 Ready when you are! Allow camera access and capture a clear photo for best results.")
    
    # Tips section
    with st.expander("📸 Tips for Best Results"):
        st.markdown("""
        - **Lighting**: Use good, even lighting on your face
        - **Angle**: Face the camera directly, looking straight ahead
        - **Background**: Use a plain background if possible
        - **Expression**: Neutral expression works best
        - **Distance**: Position yourself at arm's length from the camera
        - **Quality**: Ensure your camera lens is clean
        """)

# Footer
st.markdown("""
<hr style='border: none; border-top: 1px solid rgba(255,255,255,0.15); margin: 35px 0;'>
<p style='text-align: center; color: #888; font-size: 0.85rem;'>
    Made with ❤️ using ResNet50 Deep Learning • Face Detection • Image Enhancement
</p>
""", unsafe_allow_html=True)