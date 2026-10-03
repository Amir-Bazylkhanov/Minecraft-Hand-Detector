"""Pure-Python palm-contact gesture recognition and key-binding contract.

This module is intentionally independent of camera, keyboard, Minecraft, and
third-party packages.  A camera/tracking layer supplies already-detected hand
landmarks; an input layer decides what to do with the returned key sets.

Integration contract
--------------------
Create :class:`PalmContactRecognizer`, then call ``update_hands(hands)`` once
per camera frame::

    recognizer = PalmContactRecognizer()
    result = recognizer.update_hands({"Left": landmarks})

``hands`` maps a handedness label (``"Left"``/``"Right"``, case-insensitive)
to at least 21 landmarks in MediaPipe order.  A landmark may be an object with
``x`` and ``y`` attributes, a ``(x, y)`` tuple/list (extra coordinates are
ignored), or a mapping with ``"x"`` and ``"y"`` entries.  Coordinates may use
any consistent scale and origin; contact distances are normalized by palm size.

``update_hands`` returns :class:`RecognitionResult`:

``held_keys``
    A ``frozenset`` of keys that should be down after this frame.  Send key-up
    for previously held keys absent from this set.
``pulse_keys``
    A tuple of keys that entered contact on this frame only.  A pulse is emitted
    once per contact, not repeatedly while the finger remains on the palm.
``contact_states``
    Gesture-id to current contact boolean after hysteresis.
``confidences``
    Gesture-id to a state confidence in the inclusive range 0.0 to 1.0.
``distances``
    Gesture-id to the current palm-normalized tip-to-palm distance, or ``None``
    when the required hand is not tracked.
``tracked_hands``
    Normalized handedness labels with valid landmarks in this frame.

Default gestures are deliberately separated by handedness and landmark:
left index tip (8) to palm holds ``W``; left thumb tip (4) to palm pulses
``SPACE`` once per contact.  Tracking loss releases every hold immediately.
Custom bindings can target either hand, any named finger tip, ``hold`` or
``pulse`` actions, and per-gesture hysteresis thresholds.  This module never
opens a webcam and never sends keyboard events.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Set, Tuple

WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20
PALM_POINTS = (0, 5, 9, 13, 17)

FINGER_TIPS = {
    "thumb": THUMB_TIP,
    "index": INDEX_TIP,
    "middle": MIDDLE_TIP,
    "ring": RING_TIP,
    "pinky": PINKY_TIP,
}

DEFAULT_BINDINGS: Dict[str, Dict[str, Any]] = {
    "left_index_palm": {
        "hand": "left",
        "finger": "index",
        "action": "hold",
        "key": "W",
        "description": "Hold W (walk forward) while the left index tip touches the palm.",
    },
    "left_thumb_palm": {
        "hand": "left",
        "finger": "thumb",
        "action": "pulse",
        "key": "SPACE",
        "description": "Pulse Space (jump) once when the left thumb tip touches the palm.",
    },
}


@dataclass(frozen=True)
class RecognitionResult:
    """Frame result returned by :meth:`PalmContactRecognizer.update_hands`."""

    held_keys: frozenset
    pulse_keys: tuple
    contact_states: Dict[str, bool]
    confidences: Dict[str, float]
    distances: Dict[str, Optional[float]]
    tracked_hands: frozenset

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-compatible snapshot for logging or IPC."""

        return {
            "held_keys": sorted(self.held_keys),
            "pulse_keys": list(self.pulse_keys),
            "contact_states": dict(self.contact_states),
            "confidences": dict(self.confidences),
            "distances": dict(self.distances),
            "tracked_hands": sorted(self.tracked_hands),
        }


@dataclass(frozen=True)
class GestureBinding:
    """Validated contact-to-key binding used internally by the recognizer."""

    gesture_id: str
    hand: str
    finger: str
    landmark: int
    action: str
    key: str
    enter_distance: float
    exit_distance: float
    ambiguity_margin: float


@dataclass(frozen=True)
class _Observation:
    palm: Tuple[float, float]
    scale: float
    points: Tuple[Tuple[float, float], ...]

    def normalized_distance(self, landmark: int) -> float:
        x, y = self.points[landmark]
        return hypot(x - self.palm[0], y - self.palm[1]) / self.scale

    def folded_index_palm_distance(self) -> float:
        """Accept an isolated bent index at the palm edge, not an open hand/fist.

        The image cannot prove physical contact. Require a clearly curled
        index and at least two extended other fingers before using the palm
        polygon instead of the small center-contact target.
        """
        def wrist_distance(index: int) -> float:
            x, y = self.points[index]
            wx, wy = self.points[WRIST]
            return hypot(x - wx, y - wy)

        if wrist_distance(8) >= wrist_distance(6) - 0.15 * self.scale:
            return float("inf")
        extended = sum(wrist_distance(tip) > wrist_distance(pip) + 0.15 * self.scale
                       for pip, tip in ((10, 12), (14, 16), (18, 20)))
        if extended < 2:
            return float("inf")

        # Palm contour follows wrist -> index -> middle -> ring -> pinky.
        polygon = [self.points[index] for index in PALM_POINTS]
        x, y = self.points[INDEX_TIP]
        inside = False
        closest = float("inf")
        for a, b in zip(polygon, polygon[1:] + polygon[:1]):
            ax, ay = a
            bx, by = b
            dx, dy = bx - ax, by - ay
            length_sq = dx * dx + dy * dy
            t = 0.0 if length_sq == 0 else max(0.0, min(1.0,
                ((x - ax) * dx + (y - ay) * dy) / length_sq))
            closest = min(closest, hypot(x - ax - t * dx, y - ay - t * dy))
            if (ay > y) != (by > y):
                crossing_x = ax + (y - ay) * dx / dy
                if x < crossing_x:
                    inside = not inside
        return 0.0 if inside else closest / self.scale


@dataclass
class _GestureState:
    contact: bool = False


class PalmContactRecognizer:
    """Recognize palm contacts with hysteresis, without performing I/O.

    Parameters
    ----------
    bindings:
        Optional mapping of gesture id to binding dictionaries.  Each binding
        accepts ``hand`` (``left``/``right``), ``finger`` (``thumb``, ``index``,
        ``middle``, ``ring``, or ``pinky``), ``action`` (``hold``/``pulse``),
        ``key``, and optional ``enter_distance``, ``exit_distance``, and
        ``ambiguity_margin``.  When omitted, :data:`DEFAULT_BINDINGS` is used.
    enter_distance, exit_distance:
        Default palm-normalized thresholds.  Contact starts at or below
        ``enter_distance`` and ends at or above ``exit_distance``.  Values
        between them preserve the previous state (hysteresis).
    ambiguity_margin:
        Default minimum normalized separation from the other finger tips.  On
        contact entry the target tip must be at least this much closer than the
        nearest other tip, preventing an index contact from being interpreted
        as a thumb contact (or vice versa) when the hand is ambiguous.
    """

    def __init__(
        self,
        bindings: Optional[Mapping[str, Mapping[str, Any]]] = None,
        *,
        enter_distance: float = 0.32,
        exit_distance: float = 0.48,
        ambiguity_margin: float = 0.03,
    ) -> None:
        self._validate_thresholds(enter_distance, exit_distance)
        if ambiguity_margin < 0:
            raise ValueError("ambiguity_margin must be non-negative")
        self.enter_distance = float(enter_distance)
        self.exit_distance = float(exit_distance)
        self.ambiguity_margin = float(ambiguity_margin)
        source = DEFAULT_BINDINGS if bindings is None else bindings
        self.bindings = tuple(self._coerce_binding(gid, spec) for gid, spec in source.items())
        self._states = {binding.gesture_id: _GestureState() for binding in self.bindings}

    @staticmethod
    def _validate_thresholds(enter_distance: float, exit_distance: float) -> None:
        if not 0 < float(enter_distance) < float(exit_distance):
            raise ValueError("thresholds must satisfy 0 < enter_distance < exit_distance")

    def _coerce_binding(self, gesture_id: str, spec: Mapping[str, Any]) -> GestureBinding:
        try:
            hand = str(spec["hand"]).strip().lower()
            finger = str(spec["finger"]).strip().lower()
            action = str(spec["action"]).strip().lower()
            key = str(spec["key"]).strip().upper()
        except KeyError as exc:
            raise ValueError(f"binding {gesture_id!r} is missing {exc.args[0]!r}") from exc
        if hand not in {"left", "right"}:
            raise ValueError(f"binding {gesture_id!r} hand must be 'left' or 'right'")
        if finger not in FINGER_TIPS:
            raise ValueError(f"binding {gesture_id!r} has unsupported finger {finger!r}")
        if action not in {"hold", "pulse"}:
            raise ValueError(f"binding {gesture_id!r} action must be 'hold' or 'pulse'")
        if not key:
            raise ValueError(f"binding {gesture_id!r} key must not be empty")
        enter = float(spec.get("enter_distance", self.enter_distance))
        exit_ = float(spec.get("exit_distance", self.exit_distance))
        margin = float(spec.get("ambiguity_margin", self.ambiguity_margin))
        self._validate_thresholds(enter, exit_)
        if margin < 0:
            raise ValueError(f"binding {gesture_id!r} ambiguity_margin must be non-negative")
        return GestureBinding(
            gesture_id=str(gesture_id),
            hand=hand,
            finger=finger,
            landmark=FINGER_TIPS[finger],
            action=action,
            key=key,
            enter_distance=enter,
            exit_distance=exit_,
            ambiguity_margin=margin,
        )

    def update_hands(self, hands: Mapping[str, Sequence[Any]]) -> RecognitionResult:
        """Process one frame and return held and newly pulsed keys.

        Missing or malformed hands are treated as tracking loss.  A tracking
        loss clears that gesture's contact state, which releases held keys;
        pulse bindings are re-armed so a later new contact can pulse again.
        """

        observations = self._observations_by_hand(hands)
        held: Set[str] = set()
        pulses = []
        states: Dict[str, bool] = {}
        confidences: Dict[str, float] = {}
        distances: Dict[str, Optional[float]] = {}

        for binding in self.bindings:
            state = self._states[binding.gesture_id]
            observation = observations.get(binding.hand)
            if observation is None:
                state.contact = False
                states[binding.gesture_id] = False
                confidences[binding.gesture_id] = 0.0
                distances[binding.gesture_id] = None
                continue

            target_distance = observation.normalized_distance(binding.landmark)
            other_distances = [
                observation.normalized_distance(index)
                for index in FINGER_TIPS.values()
                if index != binding.landmark
            ]
            nearest_other = min(other_distances)
            unambiguous = target_distance + binding.ambiguity_margin <= nearest_other

            edge_distance = (observation.folded_index_palm_distance()
                             if binding.finger == "index" else float("inf"))
            edge_contact = (unambiguous and edge_distance <=
                            (0.35 if state.contact else 0.25))

            if not state.contact:
                if (target_distance <= binding.enter_distance and unambiguous) or edge_contact:
                    state.contact = True
                    if binding.action == "pulse":
                        pulses.append(binding.key)
            elif target_distance >= binding.exit_distance and not edge_contact:
                state.contact = False

            if state.contact and binding.action == "hold":
                held.add(binding.key)

            states[binding.gesture_id] = state.contact
            distances[binding.gesture_id] = target_distance
            confidences[binding.gesture_id] = self._confidence(
                state.contact,
                target_distance,
                nearest_other,
                binding,
            )
            if state.contact and edge_contact:
                # Score the path that actually recognized contact. Center
                # distance alone would show zero confidence at the palm edge.
                confidences[binding.gesture_id] = max(
                    confidences[binding.gesture_id],
                    round(self._clamp((0.35 - edge_distance) / 0.10), 6))

        return RecognitionResult(
            held_keys=frozenset(held),
            pulse_keys=tuple(pulses),
            contact_states=states,
            confidences=confidences,
            distances=distances,
            tracked_hands=frozenset(observations),
        )

    def reset(self) -> None:
        """Clear all contact memory and therefore release every held binding."""

        for state in self._states.values():
            state.contact = False

    def _confidence(
        self,
        contact: bool,
        target_distance: float,
        nearest_other: float,
        binding: GestureBinding,
    ) -> float:
        band = binding.exit_distance - binding.enter_distance
        if contact:
            state_confidence = 1.0 - self._clamp(
                (target_distance - binding.enter_distance) / band
            )
        else:
            state_confidence = self._clamp(
                (target_distance - binding.enter_distance) / band
            )
        specificity = self._clamp((nearest_other - target_distance) / max(band, 1e-9))
        return round(self._clamp(state_confidence * (0.5 + 0.5 * specificity)), 6)

    @staticmethod
    def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
        return max(low, min(high, value))

    def _observations_by_hand(self, hands: Mapping[str, Sequence[Any]]) -> Dict[str, _Observation]:
        observations: Dict[str, _Observation] = {}
        if not isinstance(hands, Mapping):
            return observations
        for handedness, landmarks in hands.items():
            hand = str(handedness).strip().lower()
            if hand not in {"left", "right"}:
                continue
            observation = self._make_observation(landmarks)
            if observation is not None:
                observations[hand] = observation
        return observations

    def _make_observation(self, landmarks: Sequence[Any]) -> Optional[_Observation]:
        try:
            if len(landmarks) <= max(PALM_POINTS + tuple(FINGER_TIPS.values())):
                return None
            points = tuple(self._point_xy(landmark) for landmark in landmarks)
        except (TypeError, ValueError, AttributeError, IndexError, KeyError):
            return None
        if any(not (isfinite(x) and isfinite(y)) for x, y in points):
            return None
        palm_points = [points[index] for index in PALM_POINTS]
        palm_x = sum(point[0] for point in palm_points) / len(palm_points)
        palm_y = sum(point[1] for point in palm_points) / len(palm_points)
        squared = sum(hypot(x - palm_x, y - palm_y) ** 2 for x, y in palm_points)
        scale = (squared / len(palm_points)) ** 0.5
        if not isfinite(scale) or scale <= 1e-9:
            return None
        return _Observation((palm_x, palm_y), scale, points)

    @staticmethod
    def _point_xy(point: Any) -> Tuple[float, float]:
        if hasattr(point, "x") and hasattr(point, "y"):
            return float(point.x), float(point.y)
        if isinstance(point, Mapping):
            return float(point["x"]), float(point["y"])
        return float(point[0]), float(point[1])


__all__ = [
    "DEFAULT_BINDINGS",
    "FINGER_TIPS",
    "GestureBinding",
    "INDEX_TIP",
    "MIDDLE_TIP",
    "PALM_POINTS",
    "PalmContactRecognizer",
    "PINKY_TIP",
    "RecognitionResult",
    "RING_TIP",
    "THUMB_TIP",
    "WRIST",
]
