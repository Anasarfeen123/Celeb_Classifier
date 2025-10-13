# generate_features_parallel.py
import os
import pickle
from pathlib import Path
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
import numpy as np
import argparse
import warnings

# quiet TF logs a bit
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# TensorFlow imports inside initializer to avoid forking issues
MODEL = None
VECTOR_SIZE = None

def init_worker():
    """Initializer for each worker process: loads the TF model into that process."""
    global MODEL, VECTOR_SIZE
    # import inside function so it happens in worker process
    from tensorflow.keras.applications.vgg16 import VGG16
    # load model (this happens once per worker)
    MODEL = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    # compute flattened vector length (e.g., 7*7*512 = 25088)
    out_shape = MODEL.output_shape  # (None, H, W, C)
    VECTOR_SIZE = int(np.prod(out_shape[1:]))

def extract_feature(path_str):
    """Worker function: loads image, preprocesses and returns flattened feature vector."""
    global MODEL, VECTOR_SIZE
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.applications.vgg16 import preprocess_input
    import numpy as np

    path = Path(path_str)
    if not path.exists():
        # return zero vector but keep dtype float32
        return np.zeros((VECTOR_SIZE,), dtype=np.float32)

    try:
        img = image.load_img(path, target_size=(224, 224))
        arr = image.img_to_array(img)
        arr = np.expand_dims(arr, axis=0)
        arr = preprocess_input(arr)
        feat = MODEL.predict(arr, verbose=0)
        return feat.flatten().astype(np.float32)
    except Exception as e:
        # safe fallback: print once and return zeros
        print(f"⚠️ Error processing {path}: {e}")
        return np.zeros((VECTOR_SIZE,), dtype=np.float32)

def chunked_iterator(it, chunk_size=256):
    # yields lists of items (useful if you want to chunk mapping)
    it = list(it)
    for i in range(0, len(it), chunk_size):
        yield it[i:i+chunk_size]

def main(args):
    data_dir_main = Path(args.base_dir).resolve()
    filenames_pkl = data_dir_main / "filenames.pkl"

    if not filenames_pkl.exists():
        raise FileNotFoundError(f"Could not find {filenames_pkl}. Run generate_filenames.py first.")

    # load filenames
    raw_filenames = pickle.load(open(filenames_pkl, "rb"))
    # normalize to string paths, ensure absolute
    filenames = [str(Path(f).resolve()) for f in raw_filenames]

    print(f"Found {len(filenames)} files to process. Workers: {args.workers}")

    # safety: GPU warning
    if args.workers > 1 and args.use_gpu:
        warnings.warn(
            "You requested multiple workers with GPU enabled. "
            "Multiple TF processes sharing a single GPU often crash with OOMs. "
            "Consider running with --use-gpu False or --workers 1."
        )

    # create pool and process
    with Pool(processes=args.workers, initializer=init_worker) as pool:
        # imap yields results as they finish, but we want progress bar
        results_iter = pool.imap_unordered(extract_feature, filenames, chunksize=4)

        features = []
        for feat in tqdm(results_iter, total=len(filenames), desc="Extracting features"):
            features.append(feat)

    # convert to numpy array (N, vector_len) for smaller disk + easier later use
    features_arr = np.stack(features, axis=0).astype(np.float32)

    out_path = data_dir_main / args.output
    with open(out_path, "wb") as f:
        pickle.dump(features_arr, f)

    print(f"\n✅ Saved features to {out_path} (shape {features_arr.shape})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel VGG16 feature extraction")
    parser.add_argument(
        "--base-dir",
        type=str,
        default="/home/anas/Celeb_Classifier",
        help="Path to Celeb_Classifier (contains filenames.pkl and where output will be saved).",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of worker processes to spawn."
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
