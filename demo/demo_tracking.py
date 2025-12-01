import cv2
import numpy as np
import sys
import os

# Add parent directory to path so we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.detection.detector import PersonDetector
from src.tracking.tracker import PersonTracker


def get_color_for_id(track_id):
    """Generate a unique color for each track ID."""
    # Convert track_id to int if it's not already
    if isinstance(track_id, str):
        track_id = int(track_id)
    elif not isinstance(track_id, int):
        track_id = hash(track_id) % 1000  # Convert any type to int

    # Use golden ratio for well-distributed colors
    golden_ratio = 0.618033988749895
    hue = (track_id * golden_ratio) % 1.0

    # Convert HSV to RGB (OpenCV uses BGR)
    hsv = np.array([[[hue * 180, 255, 255]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return tuple(map(int, bgr[0][0]))


def main():
    # Initialize detector and tracker
    print("Loading YOLOv8 model...")
    detector = PersonDetector(conf_threshold=0.4)

    print("Loading DeepSORT tracker...")
    tracker = PersonTracker(max_age=30)

    # Video is now in demo directory
    video_path = os.path.join(os.path.dirname(__file__), "example_video.mp4")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    print("Processing video... Press 'q' to quit")

    frame_count = 0
    unique_ids = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Detect persons
        detections = detector.detect(frame)

        # Update tracker
        tracks = tracker.update(detections, frame)

        # Draw tracked persons with unique colors
        confirmed_tracks = [t for t in tracks if t.is_confirmed()]

        for track in confirmed_tracks:
            track_id = track.track_id
            unique_ids.add(track_id)

            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])

            # Get unique color for this ID
            color = get_color_for_id(track_id)

            # Draw rectangle with unique color
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw ID label with matching color
            label = f"ID: {track_id}"

            # Add background rectangle for text
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1
            )

            # Draw text in white
            cv2.putText(
                frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        # Show stats
        stats_text = f"Frame: {frame_count} | Active: {len(confirmed_tracks)} | Total IDs: {len(unique_ids)}"
        cv2.putText(
            frame,
            stats_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Display frame
        cv2.imshow("Person Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\nDone!")
    print(f"Total unique people tracked: {len(unique_ids)}")
    print(f"IDs: {sorted(unique_ids)}")


if __name__ == "__main__":
    main()
