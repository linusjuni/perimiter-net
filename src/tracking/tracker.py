from deep_sort_realtime.deepsort_tracker import DeepSort


class PersonTracker:
    """Simple wrapper for DeepSORT tracking."""
    
    def __init__(self, max_age=30):
        """
        Initialize tracker.
        
        Args:
            max_age: Number of frames to keep track alive without detection
        """
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=3,  # Confirm track after 3 consecutive detections
            max_iou_distance=0.7,
            embedder="mobilenet",  # Feature extractor for re-identification
            embedder_gpu=False
        )
    
    def update(self, detections, frame):
        """
        Update tracks with new detections.
        
        Args:
            detections: List of [x1, y1, x2, y2, confidence]
            frame: Current frame (numpy array)
        
        Returns:
            List of tracks with track_id and bounding box
        """
        # Convert to DeepSORT format: [[x1, y1, w, h, conf], ...]
        deep_sort_detections = []
        for det in detections:
            x1, y1, x2, y2, conf = det
            w = x2 - x1
            h = y2 - y1
            deep_sort_detections.append(([x1, y1, w, h], conf, 'person'))
        
        # Update tracker
        tracks = self.tracker.update_tracks(deep_sort_detections, frame=frame)
        
        return tracks