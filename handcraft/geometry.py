"""Hand landmark geometry helpers.

Pure math over MediaPipe's 21 hand landmarks. Landmarks are plain
``(x, y, z)`` tuples of normalized floats, exactly as produced by
``mediapipe.tasks.python.vision.HandLandmarker`` (x/y in image space,
z relative depth toward the camera).
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Sequence, Tuple

Point = Tuple[float, float, float]

# MediaPipe hand landmark indices.
WRIST = 0
THUMB_TIP = 4
THUMB_IP = 3
INDEX_MCP = 5
MIDDLE_MCP = 9
PINKY_MCP = 17

# (mcp, pip, tip) for the four non-thumb fingers.
FINGERS = ((5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20))


class HandClass(Enum):
    FIST = "fist"
    OPEN_PALM = "open_palm"
    OTHER = "other"


def dist(a: Point, b: Point) -> float:
    """Euclidean distance between two 3D points."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def dist2d(a: Point, b: Point) -> float:
    """Distance in the image plane (z ignored)."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def palm_size(landmarks: Sequence[Point]) -> float:
    """Reference hand size: wrist -> middle-finger MCP distance.

    Used to normalize velocities and finger metrics so behaviour is
    independent of how far the hand is from the camera. Returns 0.0 for
    degenerate input.
    """
    if len(landmarks) < 21:
        return 0.0
    return dist2d(landmarks[WRIST], landmarks[MIDDLE_MCP])


def finger_extension(landmarks: Sequence[Point], mcp: int, pip: int, tip: int,
                     palm: float) -> float:
    """Signed extension of one finger, normalized by palm size.

    Positive means the tip is further from the wrist than the PIP joint
    (extended); negative means the tip is tucked back toward the wrist
    (curled).
    """
    wrist = landmarks[WRIST]
    return (dist(wrist, landmarks[tip]) - dist(wrist, landmarks[pip])) / palm


# Modern shared thumb-gap thresholds: thumb tip -> index-MCP distance
# normalized by palm size. Deliberately relaxed versus the legacy .8/.65
# pair so moderate thumb extensions register consistently on both hands.
THUMB_GAP_EXTENDED = 0.65
THUMB_GAP_FOLDED = 0.55


def thumb_extended(landmarks: Sequence[Point], palm: float,
                   threshold: float = 0.8) -> bool:
    """True when the thumb tip sits clearly away from the palm.

    Legacy strict classifier kept for the classic engine; modern code
    uses thumb_gap_ratio / thumb_hysteresis with THUMB_GAP_* instead.
    """
    return dist(landmarks[THUMB_TIP], landmarks[INDEX_MCP]) / palm > threshold


def thumb_gap_ratio(landmarks: Sequence[Point],
                    palm: float | None = None) -> float:
    """Thumb tip -> index MCP distance, normalized by palm size.

    Returns NaN for degenerate input (too few landmarks, non-finite
    values or a near-zero palm) so stateful callers hold their current
    hysteresis state instead of spuriously folding or extending.
    """
    if len(landmarks) <= INDEX_MCP:
        return float("nan")
    if palm is None:
        palm = palm_size(landmarks)
    gap = dist(landmarks[THUMB_TIP], landmarks[INDEX_MCP])
    if not (math.isfinite(gap) and math.isfinite(palm)) or palm < 1e-4:
        return float("nan")
    return gap / palm


def thumb_hysteresis(gap: float, out: bool,
                     extended: float = THUMB_GAP_EXTENDED,
                     folded: float = THUMB_GAP_FOLDED):
    """Advance a thumb-gap hysteresis state by one frame.

    ``gap`` is a thumb_gap_ratio measurement, ``out`` the current state.
    Returns ``(out, rose, fell)``. A NaN gap or one inside the
    folded..extended band keeps the state with no edges, so jitter and
    degenerate frames neither arm nor release.
    """
    if not math.isfinite(gap):
        return out, False, False
    if not out and gap > extended:
        return True, True, False
    if out and gap < folded:
        return False, False, True
    return out, False, False


def classify_hand(landmarks: Sequence[Point],
                  extended_margin: float = 0.15,
                  curled_margin: float = 0.05) -> HandClass:
    """Classify a 21-landmark hand as FIST, OPEN_PALM or OTHER.

    All metrics are normalized by palm size, so the classification is
    scale invariant. Ambiguous hands return OTHER so callers can require
    a stable dwell on a definite class before acting.
    """
    palm = palm_size(landmarks)
    if palm < 1e-4:
        return HandClass.OTHER

    extensions = [
        finger_extension(landmarks, mcp, pip, tip, palm)
        for mcp, pip, tip in FINGERS
    ]
    extended = sum(1 for e in extensions if e > extended_margin)
    curled = sum(1 for e in extensions if e < curled_margin)

    if curled == 4 and not thumb_extended(landmarks, palm):
        return HandClass.FIST
    if extended == 4:
        return HandClass.OPEN_PALM
    return HandClass.OTHER
