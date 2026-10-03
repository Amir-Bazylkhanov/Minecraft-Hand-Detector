"""Pure gesture engine.

Consumes a stream of hand landmarks (or ``None`` when tracking is lost)
plus monotonic timestamps, and emits semantic :class:`GestureEvent`s.
No I/O of any kind: no camera, no OS input, no clock reads — every call
is driven by arguments, which makes the whole state machine unit-testable.

Gesture set (mining + inventory prototype):

* Fist + fast palm-normalized wrist jab  -> ``PUNCH`` (holds left mouse
  via a lease; repeated jabs refresh the lease, so mining is one long
  hold, never a series of short clicks).
* Any departure from FIST, lease expiry (no jab for ``punch_lease_s``),
  or tracking loss -> ``STOP_ATTACK`` (release left mouse). Punches only
  fire from a rearmed state: after every punch the hand must first slow
  below the rearm threshold, and lease expiry never rearms — continuous
  fast motion cannot restart mining on its own.
* Stable open palm while inventory believed closed -> ``OPEN_INVENTORY``.
* Stable fist while inventory believed open -> ``CLOSE_INVENTORY``
  followed by an attack-suppression window so the closing motion cannot
  immediately mine.

Safety properties implemented here:

* Acquire dwell: after tracking is (re)gained, no gesture can fire until
  the hand has been present and tracked for ``acquire_dwell_s`` — no
  surprise punches on reacquire.
* Stability dwell: inventory gestures need the classification to hold for
  ``gesture_dwell_s``.
* Inventory open/close fire exactly once per believed state transition;
  a manual sync (``set_inventory_state``) reconciles belief with reality.
* Punches are disabled entirely while the inventory is believed open.
* Attack release precedes inventory: when an open/close event fires
  while attacking, ``STOP_ATTACK`` is emitted first.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, List, Optional, Sequence, Tuple

from .geometry import HandClass, Point, classify_hand, dist, palm_size


class GestureKind(Enum):
    PUNCH = "PUNCH"                # start/refresh the left-mouse mining hold
    STOP_ATTACK = "STOP_ATTACK"    # release the left-mouse hold
    OPEN_INVENTORY = "OPEN_INVENTORY"
    CLOSE_INVENTORY = "CLOSE_INVENTORY"


@dataclass(frozen=True)
class GestureEvent:
    kind: GestureKind
    at: float  # monotonic timestamp the event was decided


@dataclass
class GestureConfig:
    """Tunable thresholds. ``sensitivity`` scales the punch threshold and
    dwell times; 1.0 is the default feel, higher is more twitchy."""

    sensitivity: float = 1.0

    punch_velocity_base: float = 5.0   # palm-lengths / second
    rearm_ratio: float = 0.5           # return speed ratio that rearms a punch
    punch_lease_s: float = 0.8         # hold refreshed by each punch
    min_punch_interval_s: float = 0.15

    acquire_dwell_s: float = 0.35      # no actions right after (re)acquire
    gesture_dwell_s: float = 0.45      # stability required for open/close
    inventory_cooldown_s: float = 0.8  # min gap between inventory events
    post_close_suppression_s: float = 0.7  # no punches right after closing

    velocity_window_s: float = 0.12    # maximum gap for a velocity sample
    history_window_s: float = 0.30     # how long wrist history is kept

    @property
    def punch_velocity(self) -> float:
        s = min(max(self.sensitivity, 0.25), 4.0)
        return self.punch_velocity_base / s

    @property
    def effective_acquire_dwell_s(self) -> float:
        return self.acquire_dwell_s

    @property
    def effective_gesture_dwell_s(self) -> float:
        s = min(max(self.sensitivity, 0.25), 4.0)
        return self.gesture_dwell_s / s


class GestureEngine:
    """Stateful gesture recognizer. Feed one sample per camera frame via
    :meth:`update`; feed ``None`` whenever tracking is lost or a frame is
    stale."""

    def __init__(self, config: Optional[GestureConfig] = None):
        self.config = config or GestureConfig()
        # tracking state
        self._hand_present = False
        self._acquired_at = 0.0
        self._history: Deque[Tuple[float, Point, float]] = deque()
        # classification stability
        self._current_class = HandClass.OTHER
        self._class_since = 0.0
        # attack lease
        self._attacking = False
        self._last_punch_at = 0.0
        self._rearmed = True
        # inventory belief: True open, False closed, None unknown
        self._inventory_believed: Optional[bool] = False
        self._last_inventory_event_at = -10.0
        self._suppression_until = 0.0

    # ------------------------------------------------------------------ API

    @property
    def attacking(self) -> bool:
        return self._attacking

    @property
    def inventory_believed(self) -> Optional[bool]:
        return self._inventory_believed

    def set_inventory_state(self, state: Optional[bool]) -> None:
        """Manual inventory sync. Does not emit any event."""
        self._inventory_believed = state

    def set_sensitivity(self, sensitivity: float) -> None:
        self.config.sensitivity = sensitivity

    def reset(self) -> List[GestureEvent]:
        """Drop all transient state; releases the attack hold if active."""
        events: List[GestureEvent] = []
        if self._attacking:
            events.append(GestureEvent(GestureKind.STOP_ATTACK, self._last_punch_at))
        self._reset_tracking()
        self._attacking = False
        self._rearmed = True
        self._suppression_until = 0.0
        return events

    def update(self, landmarks: Optional[Sequence[Point]], now: float
               ) -> List[GestureEvent]:
        """Advance the state machine with one frame (or tracking loss)."""
        events: List[GestureEvent] = []

        if landmarks is None or len(landmarks) < 21:
            events.extend(self._on_tracking_lost(now))
            return events

        # Lease expiry is checked every live frame too.
        events.extend(self._check_lease(now))

        palm = palm_size(landmarks)
        if palm < 1e-4:
            # Degenerate landmarks: treat as tracking loss for safety.
            events.extend(self._on_tracking_lost(now))
            return events

        self._push_history(now, landmarks[0], palm)

        if not self._hand_present:
            # (Re)acquisition: start the dwell clock, act on nothing yet.
            self._hand_present = True
            self._acquired_at = now
            self._current_class = classify_hand(landmarks)
            self._class_since = now
            return events

        if now - self._acquired_at < self.config.effective_acquire_dwell_s:
            return events

        cls = classify_hand(landmarks)
        if cls is not self._current_class:
            self._current_class = cls
            self._class_since = now
        stable = now - self._class_since >= self.config.effective_gesture_dwell_s

        # Attack only survives a fist: any class departure releases the
        # hold immediately, without waiting for a dwell or lease expiry.
        if self._attacking and cls is not HandClass.FIST:
            self._attacking = False
            events.append(GestureEvent(GestureKind.STOP_ATTACK, now))

        velocity = self._wrist_velocity(now)

        # Inventory close first: a stable fist while the inventory is
        # believed open closes it and suppresses attacks afterwards.
        if stable and cls is HandClass.FIST and self._inventory_believed is True:
            if now - self._last_inventory_event_at >= self.config.inventory_cooldown_s:
                self._inventory_believed = False
                self._last_inventory_event_at = now
                self._suppression_until = now + self.config.post_close_suppression_s
                if self._attacking:
                    self._attacking = False
                    events.append(GestureEvent(GestureKind.STOP_ATTACK, now))
                events.append(GestureEvent(GestureKind.CLOSE_INVENTORY, now))
                return events

        # Inventory open: stable open palm while not believed open. Unknown
        # belief (None) is allowed to open; closing requires certainty.
        if stable and cls is HandClass.OPEN_PALM and self._inventory_believed is not True:
            if now - self._last_inventory_event_at >= self.config.inventory_cooldown_s:
                self._inventory_believed = True
                self._last_inventory_event_at = now
                if self._attacking:
                    self._attacking = False
                    events.append(GestureEvent(GestureKind.STOP_ATTACK, now))
                events.append(GestureEvent(GestureKind.OPEN_INVENTORY, now))
                return events

        # Punch: only with a fist, never while inventory is believed open,
        # never inside the post-close suppression window.
        if (cls is HandClass.FIST
                and self._inventory_believed is not True
                and now >= self._suppression_until):
            threshold = self.config.punch_velocity
            if velocity > threshold:
                # A punch needs a rearmed engine: after every punch the
                # hand must first slow below the rearm threshold. Lease
                # expiry does not rearm, so uninterrupted fast motion
                # cannot start a new hold on its own.
                if (self._rearmed
                        and (not self._attacking
                             or now - self._last_punch_at >= self.config.min_punch_interval_s)):
                    self._attacking = True
                    self._last_punch_at = now
                    self._rearmed = False
                    events.append(GestureEvent(GestureKind.PUNCH, now))
            elif velocity < threshold * self.config.rearm_ratio:
                # The "return" stroke rearms the next jab — whether or not
                # the previous hold is still alive.
                self._rearmed = True

        return events

    # ------------------------------------------------------------- internal

    def _reset_tracking(self) -> None:
        self._hand_present = False
        self._history.clear()
        self._current_class = HandClass.OTHER
        self._class_since = 0.0

    def _on_tracking_lost(self, now: float) -> List[GestureEvent]:
        events: List[GestureEvent] = []
        if self._attacking:
            self._attacking = False
            events.append(GestureEvent(GestureKind.STOP_ATTACK, now))
        # Acquire reset: a fresh track may throw one first punch.
        self._rearmed = True
        self._reset_tracking()
        return events

    def _check_lease(self, now: float) -> List[GestureEvent]:
        # Expiry stops the attack but deliberately does NOT rearm: a new
        # punch still requires the hand to slow below the rearm threshold.
        if self._attacking and now - self._last_punch_at > self.config.punch_lease_s:
            self._attacking = False
            return [GestureEvent(GestureKind.STOP_ATTACK, now)]
        return []

    def _push_history(self, now: float, wrist: Point, palm: float) -> None:
        self._history.append((now, wrist, palm))
        cutoff = now - self.config.history_window_s
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

    def _wrist_velocity(self, now: float) -> float:
        """Speed between adjacent frames, without averaging away a fast jab.

        An out-and-back hit can have zero net displacement over a long
        window. Use the latest motion segment so the first fast fist frame
        can start mining. Long capture gaps contribute no velocity.
        """
        if len(self._history) < 2:
            return 0.0
        newest_t, newest_p, newest_palm = self._history[-1]
        oldest_t, oldest_p, oldest_palm = self._history[-2]
        palm = (newest_palm + oldest_palm) / 2
        dt = newest_t - oldest_t
        if dt <= 1e-4 or dt > self.config.velocity_window_s or palm <= 1e-6:
            return 0.0
        return dist(newest_p, oldest_p) / (dt * palm)
