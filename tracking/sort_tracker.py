# tracking/sort_tracker.py
"""
Wrapper for the original SORT tracker (tracking/sort.py).

Usage
-----
>>> from tracking.sort_tracker import init_tracker
>>> tracker = init_tracker()
>>> tracks = tracker.update(detections)   # detections: np.ndarray (N, 5)
"""

import sys
from pathlib import Path

# ------------------------------------------------------------------
# Ensure the folder containing sort.py is importable
# ------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent          # .../tracking
if str(_THIS_DIR) not in sys.path:
    sys.path.append(str(_THIS_DIR))

try:
    from sort import Sort
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "❌  Could not import 'sort'. Make sure you have downloaded\n"
        "    https://github.com/abewley/sort/blob/master/sort.py\n"
        f"and placed it inside: {_THIS_DIR}\n\nOriginal error: {e}"
    )

# ------------------------------------------------------------------
# Factory function – call this from your pipeline
# ------------------------------------------------------------------
def init_tracker(max_age: int = 30,
                 min_hits: int = 3,
                 iou_threshold: float = 0.3) -> Sort:
    """
    Create and return a SORT tracker instance.

    Parameters
    ----------
    max_age : int
        Frames to keep a track alive without detections.
    min_hits : int
        Min detections before a track is confirmed.
    iou_threshold : float
        Minimum IOU for matching detections to tracks.

    Returns
    -------
    Sort
        Ready‑to‑use SORT tracker object.
    """
    return Sort(max_age=max_age,
                min_hits=min_hits,
                iou_threshold=iou_threshold)


# ------------------------------------------------------------------
# Self‑test (run `python tracking/sort_tracker.py` to verify)
# ------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np

    tracker = init_tracker()
    dummy_dets = np.array([[100, 100, 200, 200, 1.0]])  # x1,y1,x2,y2,score
    print("Input detections:\n", dummy_dets)
    print("Tracker output:\n", tracker.update(dummy_dets))
