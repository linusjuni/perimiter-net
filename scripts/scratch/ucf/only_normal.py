import os
import glob
import cv2
from pathlib import Path

# --- CONFIGURATION ---
TARGET_SOURCE = (
    "/dtu/blackhole/10/187952/ucf-crime-blackhole/Normal_Videos_for_Event_Recognition"
)
DEST_ROOT = "/dtu/blackhole/10/187952/ucf-crime-blackhole/Frames"
# ---------------------


def process_video_loud(video_path, output_root):
    vid_name = Path(video_path).stem
    print(f"[{vid_name}]", end=" ")

    # 1. Setup Path
    class_name = "NormalVideos"
    split = "Train"  # Forcing Train for these normal videos
    save_dir = os.path.join(output_root, split, class_name)
    os.makedirs(save_dir, exist_ok=True)

    # 2. Check Skip Logic
    first_frame = os.path.join(save_dir, f"{vid_name}_0.png")
    if os.path.exists(first_frame):
        print(f"⏩ SKIPPED (File exists: {first_frame})")
        return

    # 3. Open Video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ FAILED (Could not open video)")
        return

    # 4. Read First Frame (The critical test)
    ret, frame = cap.read()
    if not ret:
        print("❌ FAILED (Opened, but frame 0 read returned False)")
        return

    # 5. Extract Loop
    print("✅ Extracting...", end=" ", flush=True)

    # Save the first frame we already read
    cv2.imwrite(first_frame, frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    count = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        out_name = f"{vid_name}_{count}.png"
        cv2.imwrite(
            os.path.join(save_dir, out_name), frame, [cv2.IMWRITE_PNG_COMPRESSION, 1]
        )
        count += 1

        # Simple progress indicator for long videos
        if count % 1000 == 0:
            print(".", end="", flush=True)

    cap.release()
    print(f" Done! ({count} frames)")


if __name__ == "__main__":
    print(f"🔍 Scanning {TARGET_SOURCE}...")
    all_videos = glob.glob(os.path.join(TARGET_SOURCE, "**", "*.mp4"), recursive=True)
    all_videos.sort()

    print(f"📹 Found {len(all_videos)} Normal videos.")
    print("-" * 60)

    for video in all_videos:
        process_video_loud(video, DEST_ROOT)
