from deep_sort_realtime.deepsort_tracker import DeepSort


class PersonTracker:
    """DeepSORT person tracker wrapper."""

    def __init__(self, max_age=30):
        """Initialize DeepSORT tracker."""
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=3,
            max_iou_distance=0.7,
            embedder="mobilenet",
            embedder_gpu=False,
        )

    def update(self, detections, frame):
        """Update tracks with new detections."""
        deep_sort_detections = []
        for det in detections:
            x1, y1, x2, y2, conf = det
            w = x2 - x1
            h = y2 - y1
            deep_sort_detections.append(([x1, y1, w, h], conf, "person"))
        tracks = self.tracker.update_tracks(deep_sort_detections, frame=frame)
        return tracks
