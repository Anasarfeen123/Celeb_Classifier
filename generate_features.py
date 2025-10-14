import os
import pickle
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import argparse
import warnings

# Silence TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Shared globals for multiprocessing
MODEL = None
VECTOR_SIZE = None

def init_worker():
    """Initializer for worker processes: loads ResNet50 in each process."""
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable GPU in subprocesses to prevent OOM
    global MODEL, VECTOR_SIZE
    from tensorflow.keras.applications import ResNet50

    MODEL = ResNet50(
        weights="imagenet",
        include_top=False,
        pooling="avg",         # global average pooling
        input_shape=(224, 224, 3)
    )
    VECTOR_SIZE = MODEL.output_shape[-1]  # typically 2048


def extract_feature(path_str):
    """Load image, preprocess, and extract ResNet50 feature vector."""
    global MODEL, VECTOR_SIZE
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.applications.resnet50 import preprocess_input
    path = Path(path_str)
    if not path.exists():
        return np.zeros((VECTOR_SIZE,), dtype=np.float32)

    try:
        img = image.load_img(path, target_size=(224, 224))
        arr = image.img_to_array(img)
        arr = np.expand_dims(arr, axis=0)
        arr = preprocess_input(arr)

        feat = MODEL.predict(arr, verbose=0).flatten().astype(np.float32)

        # L2 normalization for cosine similarity
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat /= norm

        return feat

    except Exception as e:
        print(f"⚠️ Error processing {path.name}: {e}")
        return np.zeros((VECTOR_SIZE,), dtype=np.float32)


def main(args):
    data_dir_main = Path(args.base_dir).resolve()
    filenames_pkl = data_dir_main / "filenames.pkl"

    if not filenames_pkl.exists():
        raise FileNotFoundError(f"Could not find {filenames_pkl}. Run generate_filenames.py first.")

    # Load filenames
    raw_filenames = pickle.load(open(filenames_pkl, "rb"))
    filenames = [str(Path(f).resolve()) for f in raw_filenames]

    print(f"📁 Found {len(filenames)} images to process.")
    print(f"🧠 Using ResNet50 with {args.workers} worker(s).")

    # Warn if GPU with multiprocessing
    if args.workers > 1 and args.use_gpu:
        warnings.warn(
            "Multiple TensorFlow processes with GPU can cause crashes. "
            "Use --workers 1 if running on GPU."
        )

    # Run parallel processing
    with Pool(processes=args.workers, initializer=init_worker) as pool:
        results_iter = pool.imap_unordered(extract_feature, filenames, chunksize=4)
        features = []
        for feat in tqdm(results_iter, total=len(filenames), desc="🔍 Extracting features"):
            features.append(feat)

    # Stack all features
    features_arr = np.stack(features, axis=0).astype(np.float32)

    out_path = data_dir_main / args.output
    with open(out_path, "wb") as f:
        pickle.dump(features_arr, f)

    print(f"\n✅ Saved {features_arr.shape[0]} embeddings to {out_path}")
    print(f"🧩 Feature vector size: {features_arr.shape[1]} (L2-normalized)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel ResNet50 feature extractor")
    parser.add_argument(
        "--base-dir",
        type=str,
        default="/home/anas/Celeb_Classifier",
        help="Path to Celeb_Classifier (contains filenames.pkl and where output will be saved).",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of worker processes to use."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="feature_embedding.pkl",
        help="Output pickle file name (under base-dir).",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Set if you want to attempt using GPU (CAUTION: multi-process + GPU unstable).",
    )

    args = parser.parse_args()
    main(args)
