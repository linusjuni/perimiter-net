import os
import glob
import cv2
import concurrent.futures
from tqdm import tqdm
from pathlib import Path
from functools import partial

# --- CONFIGURATION ---
SOURCE_ROOT = "/dtu/blackhole/10/187952/ucf-crime-blackhole"
DEST_ROOT = "/dtu/blackhole/10/187952/ucf-crime-blackhole/Frames"
ANNOTATION_FILE = os.path.join(SOURCE_ROOT, "Temporal_Anomaly_Annotation_for_Testing_Videos.txt")
# ---------------------

def parse_test_video_names(annotation_path):
    test_videos = set()
    if not os.path.exists(annotation_path):
        return test_videos
    with open(annotation_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 1:
                test_videos.add(parts[0].replace('.mp4', ''))
    return test_videos

def process_single_video(video_path, output_root, test_video_set):
    """Worker function for parallel processing."""
    try:
        vid_name = Path(video_path).stem
        parent_folder = Path(video_path).parent.name
        
        # --- Identify Class ---
        CLASSES = {'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion', 
                   'Fighting', 'RoadAccidents', 'Robbery', 'Shooting', 
                   'Shoplifting', 'Stealing', 'Vandalism', 'NormalVideos'}
        
        class_name = "Unknown"
        if parent_folder in CLASSES:
            class_name = parent_folder
        elif parent_folder in ["Testing_Normal_Videos_Anomaly", "Normal_Videos_for_Event_Recognition"]:
            class_name = "NormalVideos"
        else:
            for part in Path(video_path).parts:
                if part in CLASSES:
                    class_name = part
                    break
        
        if class_name == "Unknown":
            if "Normal_Videos_event" in str(video_path):
                class_name = "NormalVideos"
            else:
                return # Skip

        # --- Determine Split ---
        is_test_video = (vid_name in test_video_set) or (parent_folder == "Testing_Normal_Videos_Anomaly")
        split = "Test" if is_test_video else "Train"

        # --- Check Existing ---
        save_dir = os.path.join(output_root, split, class_name)
        # We can't safely mkdir in parallel without exist_ok=True, which is thread-safe mostly,
        # but to be safe we usually rely on OS handling.
        os.makedirs(save_dir, exist_ok=True)
        
        if os.path.exists(os.path.join(save_dir, f"{vid_name}_0.png")):
            return # Skip

        # --- Extract ---
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return

        count = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            out_name = f"{vid_name}_{count}.png"
            # Quality 1 for speed/size balance
            cv2.imwrite(os.path.join(save_dir, out_name), frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            count += 1
        cap.release()
        return 1 # Success count
    except Exception as e:
        return 0

if __name__ == "__main__":
    print(f"cpu_count: {os.cpu_count()}")
    
    # 1. Setup
    test_set = parse_test_video_names(ANNOTATION_FILE)
    all_videos = glob.glob(os.path.join(SOURCE_ROOT, "**", "*.mp4"), recursive=True)
    print(f"Found {len(all_videos)} videos. Starting parallel extraction...")

    # 2. Parallel Processing
    # We use all available CPUs passed by the scheduler
    num_workers = int(os.environ.get('LSB_DJOB_NUMPROC', 4)) 
    print(f"Spawning {num_workers} workers.")

    # Create partial function with fixed arguments
    worker_func = partial(process_single_video, output_root=DEST_ROOT, test_video_set=test_set)

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Map returns an iterator, converting to list triggers execution
        # tqdm wraps the iterator for progress bar
        results = list(tqdm(executor.map(worker_func, all_videos), total=len(all_videos)))

    print("Done!")