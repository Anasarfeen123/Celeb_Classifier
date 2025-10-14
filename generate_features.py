import os
import pickle
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import argparse
import warnings
from collections import defaultdict

# Silence TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Shared globals for multiprocessing
MODEL = None
FACE_CASCADE = None
VECTOR_SIZE = None

def init_worker():
    """Initializer for worker processes: loads ResNet50 and face detector in each process."""
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable GPU in subprocesses to prevent OOM
    global MODEL, FACE_CASCADE, VECTOR_SIZE
    from tensorflow.keras.applications import ResNet50
    import cv2
    
    MODEL = ResNet50(
        weights="imagenet",
        include_top=False,
        pooling="avg",         # global average pooling
        input_shape=(224, 224, 3)
    )
    VECTOR_SIZE = MODEL.output_shape[-1]  # typically 2048
    
    # Load Haar Cascade for face detection
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    FACE_CASCADE = cv2.CascadeClassifier(cascade_path)


def detect_and_crop_face(img_array):
    """
    Detect face in image and return cropped face region.
    Returns original image if no face detected.
    """
    global FACE_CASCADE
    import cv2
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    # Detect faces with multiple scale factors for better accuracy
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    if len(faces) > 0:
        # Get the largest face (most likely the main subject)
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face
        
        # Add padding around face (15% on each side)
        padding = int(0.15 * max(w, h))
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img_array.shape[1], x + w + padding)
        y2 = min(img_array.shape[0], y + h + padding)
        
        # Crop to face region
        face_img = img_array[y1:y2, x1:x2]
        return face_img, True
    
    return img_array, False


def augment_image(img_array):
    """
    Create augmented versions of the image for better feature extraction.
    Returns list of augmented images.
    """
    import cv2
    
    augmented = [img_array]  # Original
    
    # Horizontal flip
    augmented.append(cv2.flip(img_array, 1))
    
    # Slight rotations (-5, +5 degrees)
    h, w = img_array.shape[:2]
    center = (w // 2, h // 2)
    
    for angle in [-5, 5]:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img_array, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        augmented.append(rotated)
    
    # Brightness adjustments
    for factor in [0.9, 1.1]:
        brightened = np.clip(img_array * factor, 0, 255).astype(np.uint8)
        augmented.append(brightened)
    
    return augmented


def extract_feature(path_str, use_augmentation=False, use_face_detection=True):
    """
    Load image, detect face, preprocess, and extract ResNet50 feature vector.
    Can optionally use augmentation for more robust features.
    """
    global MODEL, VECTOR_SIZE
    from tensorflow.keras.preprocessing import image as keras_image
    from tensorflow.keras.applications.resnet50 import preprocess_input
    import cv2
    
    path = Path(path_str)
    if not path.exists():
        return np.zeros((VECTOR_SIZE,), dtype=np.float32)

    try:
        # Load image using OpenCV for better preprocessing
        img_cv = cv2.imread(str(path))
        if img_cv is None:
            return np.zeros((VECTOR_SIZE,), dtype=np.float32)
        
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        
        # Face detection and cropping
        face_detected = False
        if use_face_detection:
            img_cv, face_detected = detect_and_crop_face(img_cv)
        
        # Resize to target size
        img_resized = cv2.resize(img_cv, (224, 224), interpolation=cv2.INTER_LANCZOS4)
        
        features_list = []
        
        if use_augmentation:
            # Generate augmented versions
            augmented_images = augment_image(img_resized)
            
            for aug_img in augmented_images:
                arr = np.expand_dims(aug_img, axis=0)
                arr = preprocess_input(arr.astype(np.float32))
                feat = MODEL.predict(arr, verbose=0).flatten().astype(np.float32)
                features_list.append(feat)
            
            # Average all augmented features
            feat = np.mean(features_list, axis=0).astype(np.float32)
        else:
            # Single feature extraction
            arr = np.expand_dims(img_resized, axis=0)
            arr = preprocess_input(arr.astype(np.float32))
            feat = MODEL.predict(arr, verbose=0).flatten().astype(np.float32)
        
        # L2 normalization for cosine similarity
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat /= norm

        return feat

    except Exception as e:
        print(f"⚠️ Error processing {path.name}: {e}")
        return np.zeros((VECTOR_SIZE,), dtype=np.float32)


def extract_feature_wrapper(args):
    """Wrapper to unpack arguments for multiprocessing."""
    return extract_feature(*args)


def filter_low_quality_images(filenames, data_dir):
    """
    Filter out corrupted, too small, or invalid images.
    Returns list of valid filenames.
    """
    from PIL import Image
    import cv2
    
    valid_files = []
    invalid_count = 0
    
    print("🔍 Filtering low-quality images...")
    
    for filepath in tqdm(filenames, desc="Validating images"):
        try:
            path = Path(filepath)
            if not path.exists():
                invalid_count += 1
                continue
            
            # Try opening with PIL
            with Image.open(filepath) as img:
                w, h = img.size
                # Filter out very small images
                if w < 50 or h < 50:
                    invalid_count += 1
                    continue
                
                # Check if image is too dark/bright
                img_array = np.array(img.convert('RGB'))
                mean_brightness = np.mean(img_array)
                if mean_brightness < 10 or mean_brightness > 245:
                    invalid_count += 1
                    continue
            
            valid_files.append(filepath)
            
        except Exception as e:
            invalid_count += 1
            continue
    
    print(f"✅ Valid images: {len(valid_files)}")
    print(f"❌ Filtered out: {invalid_count} low-quality images")
    
    return valid_files


def balance_dataset(filenames, min_images_per_celeb=5, max_images_per_celeb=100):
    """
    Balance the dataset so each celebrity has similar number of images.
    This prevents bias towards celebrities with more images.
    """
    from collections import defaultdict
    import random
    
    # Group by celebrity (based on parent folder or filename pattern)
    celeb_images = defaultdict(list)
    
    for filepath in filenames:
        path = Path(filepath)
        # Extract celebrity name from path
        if 'data' in path.parts:
            idx = path.parts.index('data')
            if idx + 1 < len(path.parts):
                celeb_name = path.parts[idx + 1]
                celeb_images[celeb_name].append(filepath)
    
    balanced_files = []
    
    print(f"\n📊 Balancing dataset (min: {min_images_per_celeb}, max: {max_images_per_celeb} per celebrity)...")
    
    for celeb, images in celeb_images.items():
        num_images = len(images)
        
        if num_images < min_images_per_celeb:
            print(f"⚠️ {celeb}: Only {num_images} images (< {min_images_per_celeb}), skipping")
            continue
        
        if num_images > max_images_per_celeb:
            # Randomly sample max_images_per_celeb
            selected = random.sample(images, max_images_per_celeb)
            balanced_files.extend(selected)
            print(f"📉 {celeb}: Reduced from {num_images} to {max_images_per_celeb}")
        else:
            balanced_files.extend(images)
            print(f"✓ {celeb}: {num_images} images")
    
    print(f"\n✅ Balanced dataset: {len(balanced_files)} images from {len(celeb_images)} celebrities")
    
    return balanced_files


def main(args):
    data_dir_main = Path(args.base_dir).resolve()
    filenames_pkl = data_dir_main / "filenames.pkl"

    if not filenames_pkl.exists():
        raise FileNotFoundError(f"Could not find {filenames_pkl}. Run namexExtractor.py first.")

    # Load filenames
    raw_filenames = pickle.load(open(filenames_pkl, "rb"))
    filenames = [str(Path(f).resolve()) for f in raw_filenames]

    print(f"📁 Found {len(filenames)} images initially.")
    
    # Filter low-quality images
    if args.filter_quality:
        filenames = filter_low_quality_images(filenames, data_dir_main)
    
    # Balance dataset
    if args.balance:
        filenames = balance_dataset(
            filenames,
            min_images_per_celeb=args.min_images,
            max_images_per_celeb=args.max_images
        )
    
    print(f"\n🧠 Using ResNet50 with {args.workers} worker(s).")
    print(f"🔧 Face detection: {'ON' if args.use_face_detection else 'OFF'}")
    print(f"🔄 Augmentation: {'ON' if args.use_augmentation else 'OFF'}")

    # Warn if GPU with multiprocessing
    if args.workers > 1 and args.use_gpu:
        warnings.warn(
            "Multiple TensorFlow processes with GPU can cause crashes. "
            "Use --workers 1 if running on GPU."
        )

    # Prepare arguments for parallel processing
    process_args = [
        (fn, args.use_augmentation, args.use_face_detection) 
        for fn in filenames
    ]

    # Run parallel processing
    with Pool(processes=args.workers, initializer=init_worker) as pool:
        results_iter = pool.imap_unordered(
            extract_feature_wrapper, 
            process_args, 
            chunksize=args.chunksize
        )
        features = []
        valid_filenames = []
        
        for i, feat in enumerate(tqdm(results_iter, total=len(filenames), desc="🔍 Extracting features")):
            # Skip zero vectors (failed extractions)
            if not np.allclose(feat, 0):
                features.append(feat)
                valid_filenames.append(filenames[i])

    print(f"\n✅ Successfully extracted {len(features)} feature vectors")
    print(f"❌ Failed extractions: {len(filenames) - len(features)}")

    # Stack all features
    features_arr = np.stack(features, axis=0).astype(np.float32)

    # Save features
    out_path = data_dir_main / args.output
    with open(out_path, "wb") as f:
        pickle.dump(features_arr, f)

    # Save updated filenames (only for successfully processed images)
    filenames_out_path = data_dir_main / "filenames_processed.pkl"
    with open(filenames_out_path, "wb") as f:
        pickle.dump(valid_filenames, f)

    print(f"\n✅ Saved {features_arr.shape[0]} embeddings to {out_path}")
    print(f"✅ Saved valid filenames to {filenames_out_path}")
    print(f"🧩 Feature vector size: {features_arr.shape[1]} (L2-normalized)")
    
    # Print statistics
    print(f"\n📊 Feature Statistics:")
    print(f"   Mean: {np.mean(features_arr):.4f}")
    print(f"   Std: {np.std(features_arr):.4f}")
    print(f"   Min: {np.min(features_arr):.4f}")
    print(f"   Max: {np.max(features_arr):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enhanced ResNet50 feature extractor with face detection and augmentation"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="/home/anas/Celeb_Classifier",
        help="Path to Celeb_Classifier (contains filenames.pkl and where output will be saved).",
    )
    parser.add_argument(
        "--workers", 
        type=int, 
        default=4, 
        help="Number of worker processes to use."
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=4,
        help="Chunksize for multiprocessing (adjust based on memory)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="features.pkl",
        help="Output pickle file name (under base-dir).",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Set if you want to attempt using GPU (CAUTION: multi-process + GPU unstable).",
    )
    parser.add_argument(
        "--use-augmentation",
        action="store_true",
        help="Enable data augmentation for more robust features (slower but better accuracy).",
    )
    parser.add_argument(
        "--use-face-detection",
        action="store_true",
        default=True,
        help="Enable face detection and cropping (recommended for better accuracy).",
    )
    parser.add_argument(
        "--filter-quality",
        action="store_true",
        default=True,
        help="Filter out low-quality images before processing.",
    )
    parser.add_argument(
        "--balance",
        action="store_true",
        help="Balance dataset to prevent bias towards celebrities with more images.",
    )
    parser.add_argument(
        "--min-images",
        type=int,
        default=5,
        help="Minimum images per celebrity (used with --balance).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=100,
        help="Maximum images per celebrity (used with --balance).",
    )

    args = parser.parse_args()
    main(args)