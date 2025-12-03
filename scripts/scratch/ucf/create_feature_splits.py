import os
import shutil
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURATION ---
FEATURES_DIR = "/work3/s225224/ucf-crime/features/motion"
ANNOTATION_FILE = "/dtu/blackhole/10/187952/ucf-crime-blackhole/Temporal_Anomaly_Annotation_for_Testing_Videos.txt"
# ---------------------


def load_test_set_names(annotation_path):
    test_videos = set()
    if not os.path.exists(annotation_path):
        print("❌ Error: Annotation file not found.")
        return test_videos

    with open(annotation_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 1:
                vid_name = parts[0].replace(".mp4", "")
                test_videos.add(vid_name)
    return test_videos


def organize_features():
    print(f"📦 Organizing features in: {FEATURES_DIR}")

    # 1. Create Subdirectories
    train_dir = os.path.join(FEATURES_DIR, "Train")
    test_dir = os.path.join(FEATURES_DIR, "Test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 2. Load Key
    test_set_names = load_test_set_names(ANNOTATION_FILE)

    # 3. Find Files (Non-recursive, so we don't grab ones we already moved)
    feature_files = list(Path(FEATURES_DIR).glob("*.npy"))
    print(f"📂 Found {len(feature_files)} files to move.")

    count_moved = 0

    for f_path in tqdm(feature_files):
        vid_name = f_path.stem

        # Determine Destination
        if vid_name in test_set_names:
            dest = os.path.join(test_dir, f_path.name)
        else:
            dest = os.path.join(train_dir, f_path.name)

        # Move
        try:
            shutil.move(str(f_path), dest)
            count_moved += 1
        except Exception as e:
            print(f"Error moving {f_path.name}: {e}")

    print(f"✅ Moved {count_moved} files.")
    print(f"   Train: {len(os.listdir(train_dir))}")
    print(f"   Test:  {len(os.listdir(test_dir))}")


if __name__ == "__main__":
    organize_features()
