"""Exclusive full-hand controls, independent of cameras and OS input.

Image coordinates are the mirrored coordinates supplied by TrackerWorker.
The caller owns inventory belief, focus gating, and key/button release.
"""
from dataclasses import dataclass, field
import math
from typing import Mapping, Sequence

from .geometry import (Point, FINGERS, THUMB_GAP_EXTENDED,
                       finger_extension, palm_size, thumb_gap_ratio,
                       thumb_hysteresis)
from .gestures import GestureEvent, GestureKind


@dataclass(frozen=True)
class PoseResult:
    held_keys: tuple[str, ...] = ()
    pulse_keys: tuple[str, ...] = ()
    events: tuple[GestureEvent, ...] = ()
    look_dx: float = 0.0
    look_dy: float = 0.0
    cursor: tuple[float, float] | None = None
    click: str | None = None
    scroll: int = 0
    labels: dict[str, str] = field(default_factory=dict)
    hotbar_active: bool = False


def _valid(hand):
    return (hand is not None and len(hand) == 21
            and all(len(p) == 3 and all(math.isfinite(v) for v in p) for p in hand)
            and palm_size(hand) >= 1e-4)


def _shape(hand):
    if not _valid(hand):
        return "none", False
    palm = palm_size(hand)
    values = [finger_extension(hand, *f, palm) for f in FINGERS]
    extended = [v > .15 for v in values]
    curled = [v < .05 for v in values]
    # Relaxed modern thumb line (.65): a moderate extension already reads
    # as "thumb out", matching the stateful hysteresis open threshold.
    thumb = thumb_gap_ratio(hand, palm) > THUMB_GAP_EXTENDED
    if all(extended):
        return "open", thumb
    if extended[0] and all(curled[1:]):
        return "point", thumb
    if all(extended[:2]) and all(curled[2:]):
        return "v", thumb
    if all(curled[:3]) and extended[3]:
        return "pinky", thumb
    if all(curled):
        return "thumb" if thumb else "fist", thumb
    return "neutral", thumb


class PoseMapEngine:
    """Movement joystick, looking, thumb-cycle mining, use, menus, hotbar."""
    def __init__(self, config=None):
        self.centers: dict[str, tuple[float, float]] = {}
        self.sensitivity = 1.0
        self.inventory_cursor_speed = 80.0
        self._cycle_hold = False
        self._cycle_last = None
        self.reset()

    def set_sensitivity(self, value):
        # Compat with the app slider: the thumb-cycle trigger has no
        # velocity threshold, so the value is only recorded.
        self.sensitivity = value

    def set_inventory_cursor_speed(self, value):
        """Set the fixed inventory cursor speed in px/s, clamped to
        20..240. Invalid values (non-numeric, NaN, infinite) are rejected
        and leave the current speed unchanged; the speed survives
        reset() so a menu reopen keeps the tuned pace."""
        try:
            value = float(value)
        except (TypeError, ValueError):
            return
        if not math.isfinite(value):
            return
        self.inventory_cursor_speed = min(240.0, max(20.0, value))

    def calibrate(self, hands: Mapping[str, Sequence[Point]]):
        """Record each visible wrist as a calibration center.

        Advisory only: joystick and look sessions always re-anchor from
        their own first stable frame, so a stale calibration never moves
        the player."""
        for side in ("left", "right"):
            hand = hands.get(side)
            if _valid(hand):
                self.centers[side] = hand[0][:2]

    def reset(self):
        # Release BEFORE clearing cycle state: a promoted mining hold must
        # still emit its STOP_ATTACK here.
        events = []
        if self._cycle_hold:
            events.append(GestureEvent(GestureKind.STOP_ATTACK,
                                       self._cycle_last or 0.0))
        self._cycle_hold = False
        self._cycle_last = None
        self._cycle_armed = False
        self._cycle_out = False
        self._inv_armed = False
        self._inv_out = False
        self._inv_cursor_pause_until = None
        self._last_time = None
        self._poses = {}
        self._use_latched = False
        self._hotbar = False
        self._hotbar_ready = False
        self._hotbar_block_mining = False
        self._thumb_out = False
        self._pinky_out = False
        self._fold_since = None
        self._fold_shape = None
        self._restore_since = None
        self._both_latched = False
        self._left_thumb_out = False
        self._left_center = None
        self._left_anchor = None
        self._left_dir = (0, 0)
        self._right_center = None
        self._right_anchor = None
        self._cursor_center = None
        self._cursor_anchor = None
        self._prev_inventory = False
        return events

    def _clear_hotbar(self, unblock=False):
        self._hotbar = self._hotbar_ready = False
        self._thumb_out = self._pinky_out = False
        self._fold_shape = self._fold_since = self._restore_since = None
        if unblock:
            self._hotbar_block_mining = False

    def _stable(self, side, pose, now, dwell=.12):
        old_pose, since = self._poses.get(side, (None, now))
        if pose != old_pose:
            since = now
        self._poses[side] = (pose, since)
        return now - since >= dwell - 1e-9

    @staticmethod
    def _rate(v):
        """Signed joystick rate for look/cursor: .2 palm deadzone, the
        excess capped at 1.5 palm-widths."""
        return math.copysign(min(1.5, max(0., abs(v) - .2)), v)

    @staticmethod
    def _axis(value, current, enter=.40, exit=.25):
        if current > 0:
            return 1 if value > exit else 0
        if current < 0:
            return -1 if value < -exit else 0
        if value > enter:
            return 1
        if value < -enter:
            return -1
        return 0

    def _stop_cycle(self, now):
        """End the gameplay thumb cycle: release a promoted hold exactly
        once and drop burst/arm state, so the next cycle is a first click.
        Nothing crosses a mode, pose, hotbar, loss or reset boundary."""
        events = []
        if self._cycle_hold:
            events.append(GestureEvent(GestureKind.STOP_ATTACK, now))
        self._cycle_hold = False
        self._cycle_last = None
        self._cycle_armed = False
        self._cycle_out = False
        return events

    def _complete_cycle(self, now):
        """One finished extend->fold thumb cycle in gameplay.

        A first (or isolated) cycle maps to one timed left click. A second
        cycle completed within .6s of the last promotes the burst to held
        mining: its PUNCH is forwarded so the controller lease-holds the
        button, and further completed cycles forward PUNCH to refresh that
        .8s lease — never another click. The lease clock is this engine's
        own ``now``, not any velocity recognizer."""
        if self._cycle_hold:
            events, click = [GestureEvent(GestureKind.PUNCH, now)], None
        elif (self._cycle_last is not None
              and now - self._cycle_last <= .6):
            self._cycle_hold = True  # second cycle: promote to the hold
            events, click = [GestureEvent(GestureKind.PUNCH, now)], None
        else:
            events, click = [], "left"  # first cycle of a burst: one click
        self._cycle_last = now
        return events, click

    @staticmethod
    def _thumb_edges(hand, out):
        """Advance the shared .65/.55 thumb-gap hysteresis for one frame.

        Returns (out, rose, fell). Inside the band both edges stay false,
        so jitter around a threshold neither arms nor completes a cycle."""
        return thumb_hysteresis(thumb_gap_ratio(hand), out)

    def update(self, hands, now: float, inventory_open: bool = False):
        if not math.isfinite(now):
            return PoseResult(events=tuple(self.reset()))
        dt = 0 if self._last_time is None else now - self._last_time
        releases = []
        if dt < 0 or dt > .3:
            releases = self.reset()
            dt = 0
        self._last_time = now
        hands = hands or {}
        left, right = hands.get("left"), hands.get("right")
        lp, _lt = _shape(left)
        rp, rt = _shape(right)
        inventory_open = bool(inventory_open)
        if inventory_open != self._prev_inventory:
            # Mode edge: the shared right pose dwell must NOT carry across
            # the boundary — a point held continuously through an open or
            # close would otherwise lock the new mode's center only frames
            # in and move the cursor/view immediately. Clear the right
            # dwell, both right-hand sessions and the mode-owned click
            # state so the new mode settles a fresh .12s in BOTH
            # directions. The initial mode settles the same way.
            self._poses.pop("right", None)
            self._right_center = self._right_anchor = None
            self._cursor_center = self._cursor_anchor = None
            self._inv_armed = self._inv_out = False
            self._inv_cursor_pause_until = None
        self._prev_inventory = inventory_open
        ls = self._stable("left", lp, now)
        rs = self._stable("right", rp, now)
        labels = {"left": "Neutral", "right": "Neutral"}
        keys, pulses, events = [], [], list(releases)
        dx = dy = 0.0
        cursor = click = None
        scroll = 0

        # Both open palms form an exclusive menu chord that toggles on the
        # first recognized two-open frame. Holding the chord — still or
        # moving — never repeats the toggle; leaving two-open re-arms it.
        both = lp == rp == "open"
        if both:
            self._use_latched = False
            self._clear_hotbar(unblock=True)
            self._left_center = self._left_anchor = None
            self._left_dir = (0, 0)
            self._left_thumb_out = False
            self._right_center = self._right_anchor = None
            self._cursor_center = self._cursor_anchor = None
            labels = {"left": "Inventory chord", "right": "Inventory chord"}
            events.extend(self._stop_cycle(now))
            self._inv_armed = self._inv_out = False
            self._inv_cursor_pause_until = None
            if not self._both_latched:
                kind = GestureKind.CLOSE_INVENTORY if inventory_open else GestureKind.OPEN_INVENTORY
                events.append(GestureEvent(kind, now))
                self._both_latched = True
            return PoseResult(events=tuple(events), labels=labels)
        self._both_latched = False

        # Left joystick sessions always start neutral. The wrist drifts during
        # the initial .12s settle; the first stable point/V frame locks that
        # wrist as the center and emits nothing. Only later displacement from
        # that center (with per-axis hysteresis) presses direction keys.
        if not inventory_open and lp in ("point", "v"):
            labels["left"] = "Sneak joystick" if lp == "v" else "Move joystick"
            if self._left_center is None:
                self._left_anchor = left[0][:2]
                if ls:
                    self._left_center = self._left_anchor
                    self._left_dir = (0, 0)
            elif ls:
                palm = palm_size(left)
                x = (left[0][0] - self._left_center[0]) / palm
                y = (left[0][1] - self._left_center[1]) / palm
                xd = self._axis(x, self._left_dir[0])
                yd = self._axis(y, self._left_dir[1])
                self._left_dir = (xd, yd)
                if xd < 0: keys.append("A")
                if xd > 0: keys.append("D")
                if yd < 0: keys.append("W")
                if yd > 0: keys.append("S")
                if lp == "v": keys.append("SHIFT")
                elif yd < 0 and y < -.95: keys.append("CTRL")
        else:
            self._left_center = self._left_anchor = None
            self._left_dir = (0, 0)

        # Jump is decoupled from the joystick: a recognized left
        # point/V/thumb shape with the thumb extended holds SPACE on the
        # first recognized frame, whether or not a movement session has
        # anchored or settled. The thumb runs the same stateful .65/.55
        # gap hysteresis as the right hand: extension arms and holds
        # SPACE, gaps inside the band keep the current state (no
        # chattering), and only a fold below .55 — or an incompatible
        # pose, open palm, ambiguous neutral, invalid or lost hand, or
        # any menu state — releases it. A fist is eligible ONLY while the
        # latch is already out: a band dip on a thumb-only hand reads as
        # a fist shape, and that dip must keep feeding the hysteresis so
        # SPACE holds through the band and releases on the real fold —
        # while a cold fist can never initiate a jump.
        if (not inventory_open and (lp in ("point", "v", "thumb")
                or (lp == "fist" and self._left_thumb_out))):
            self._left_thumb_out, _, _ = thumb_hysteresis(
                thumb_gap_ratio(left), self._left_thumb_out)
            if self._left_thumb_out:
                keys.append("SPACE")
        else:
            self._left_thumb_out = False

        # Shaka (thumb + pinky out, index/middle/ring curled) arms a
        # distinct hotbar session. Inside it, raw thumb-gap and
        # pinky-extension metrics with hysteresis — not whole-hand shapes —
        # drive the fold state, so a partial bend that reads as a "neutral"
        # shape never cancels the session, and jitter around a threshold
        # neither steps nor re-arms. The session ends only on hand loss, an
        # inventory transition, or an explicit pose whose index/middle/ring
        # leave the curl (open, point, V, or a genuine neutral). Mining
        # stays blocked for the whole session, including both-folded rests.
        if inventory_open:
            self._clear_hotbar(unblock=True)
        shaka = not inventory_open and rp == "pinky" and rt
        if not self._hotbar and shaka and rs:
            self._hotbar = self._hotbar_ready = True
            self._hotbar_block_mining = True
            self._thumb_out = self._pinky_out = True
            self._fold_shape = self._fold_since = self._restore_since = None
        elif self._hotbar:
            if not _valid(right):
                self._clear_hotbar(unblock=True)
            else:
                palm = palm_size(right)
                curled3 = all(finger_extension(right, *f, palm) < .05
                              for f in FINGERS[:3])
                if not curled3:
                    self._clear_hotbar(unblock=True)
                else:
                    thumb_out, _, _ = thumb_hysteresis(
                        thumb_gap_ratio(right, palm), self._thumb_out)
                    self._thumb_out = thumb_out
                    pinky_ext = finger_extension(right, *FINGERS[3], palm)
                    # The pinky folds shallowly, so its fold line sits a
                    # little above the .05 curl used by whole-hand shapes.
                    if pinky_ext > .15:
                        self._pinky_out = True
                    elif pinky_ext < .10:
                        self._pinky_out = False
                    if self._thumb_out and self._pinky_out:
                        self._fold_shape = self._fold_since = None
                        if self._restore_since is None:
                            self._restore_since = now
                        # Both digits restored stably re-arms the next step.
                        if now - self._restore_since >= .12 - 1e-9:
                            self._hotbar_ready = True
                    else:
                        self._restore_since = None
                        if not self._thumb_out and self._pinky_out:
                            fold = "previous"
                        elif self._thumb_out and not self._pinky_out:
                            fold = "next"
                        else:  # both folded: a rest, never a step
                            fold = None
                        if fold != self._fold_shape:
                            self._fold_shape, self._fold_since = fold, now
                        if (fold is not None and self._hotbar_ready
                                and now - self._fold_since >= .12 - 1e-9):
                            scroll = 1 if fold == "previous" else -1
                            self._hotbar_ready = False
        # A right look session lives only while the right hand points in
        # gameplay. Any other pose, hand loss, or an open inventory ends
        # it; the next point session re-anchors from scratch. The
        # inventory cursor session is the mirror image: it lives only
        # while the right hand points in an OPEN menu, and its anchors
        # are fully independent so no center leaks across the boundary.
        if inventory_open or rp != "point":
            self._right_center = self._right_anchor = None
        if not inventory_open or rp != "point":
            self._cursor_center = self._cursor_anchor = None
        if self._hotbar or shaka:
            labels["right"] = "Hotbar: fold thumb left / pinky right"
        elif rp == "point":
            labels["right"] = "Inventory cursor" if inventory_open else "Look"
            if inventory_open:
                # Inventory attack FIRST: thumb edges and the click are
                # decided before the movement decision so the arm frame
                # and the completing fold frame emit ZERO movement. A
                # thumb extend->fold cycle clicks the slot once at the
                # CURRENT cursor — the click frame never carries an
                # absolute position and never moves the cursor either.
                # The inventory cycle has its own hysteresis state —
                # nothing crosses into or out of gameplay. No burst
                # promotion here.
                self._inv_out, rose, fell = self._thumb_edges(right, self._inv_out)
                if rose:
                    self._inv_armed = True
                if fell and self._inv_armed:
                    self._inv_armed = False
                    click = "left"
                    # 100ms settle cooldown after a successful fold: the
                    # wrist often keeps drifting right after the click,
                    # and that drift must not nudge the cursor off the
                    # clicked slot. Bounded and finite (now is validated
                    # above); cleared on reset, hand loss, non-point and
                    # both menu edges, so it never crosses a boundary.
                    self._inv_cursor_pause_until = now + .1
                if (self._inv_cursor_pause_until is not None
                        and now >= self._inv_cursor_pause_until):
                    self._inv_cursor_pause_until = None
                paused = (self._inv_armed
                          or self._inv_cursor_pause_until is not None)
                # Relative inventory cursor: same wrist joystick as
                # gameplay look, with its own center/anchor. A fresh
                # session tracks the drifting wrist during the .12s
                # settle; the first stable point frame locks the center
                # and emits nothing, so opening a menu never teleports
                # the cursor. Afterwards wrist displacement from the
                # locked center emits relative movement (move_relative)
                # — never an absolute position. While the thumb click
                # pause is active the center/anchor still settles
                # normally but the frozen cursor never recenters the
                # locked wrist; when the cooldown ends, movement resumes
                # at the constant speed from the current wrist offset —
                # nothing accumulates to catch up.
                if self._cursor_center is None:
                    self._cursor_anchor = right[0][:2]
                    if rs:
                        self._cursor_center = self._cursor_anchor
                elif rs and not paused:
                    # Fixed-speed inventory cursor: the axis-aligned .2
                    # palm deadzone decides which axes are active; the
                    # active offset axes give only the direction, which is
                    # normalized so the TOTAL speed is
                    # ``inventory_cursor_speed`` px/s no matter how far
                    # the wrist travels — a diagonal moves at the same
                    # total speed as a single axis. Center stops.
                    palm = palm_size(right)
                    x = (right[0][0] - self._cursor_center[0]) / palm
                    y = (right[0][1] - self._cursor_center[1]) / palm
                    x = x if abs(x) > .2 else 0.
                    y = y if abs(y) > .2 else 0.
                    length = math.hypot(x, y)
                    if length:
                        step = self.inventory_cursor_speed * min(dt, .08) / length
                        dx, dy = x * step, y * step
            elif self._right_center is None:
                # Fresh session: the wrist drifts during the .12s settle,
                # so the anchor tracks it; the first stable point frame
                # locks that wrist as the center and emits nothing.
                self._right_anchor = right[0][:2]
                if rs:
                    self._right_center = self._right_anchor
            elif rs:
                # Locked center never recenters mid-session, even while
                # the point keeps moving; only the offset from it looks.
                palm = palm_size(right)
                x = (right[0][0] - self._right_center[0]) / palm
                y = (right[0][1] - self._right_center[1]) / palm
                rate = self._rate
                cap = 700 * min(dt, .08)
                dx, dy = rate(x) * cap, rate(y) * cap
        elif rp == "v" and not inventory_open:
            labels["right"] = "Use / place"
            if rs and not self._use_latched:
                click = "right"
                self._use_latched = True
        elif rp in ("fist", "thumb") and not inventory_open and not self._hotbar_block_mining:
            labels["right"] = "Thumb attack gesture"
        if rp != "v":
            self._use_latched = False
        if not (inventory_open and rp == "point"):
            self._inv_armed = self._inv_out = False
            self._inv_cursor_pause_until = None

        # Gameplay attack: with a right point/fist/thumb (never V, open,
        # pinky/hotbar or neutral) the thumb extend->fold cycle is the only
        # attack trigger — wrist motion never attacks. The shared
        # thumb-gap hysteresis (.65 extend / .55 fold) arms only on
        # extension; the completing fold fires exactly one event per cycle.
        cycle_pose = (not inventory_open and rp in ("point", "fist", "thumb")
                      and not self._hotbar_block_mining)
        if cycle_pose:
            # Lease expiry is checked BEFORE this frame's thumb edges: a
            # fold landing after the .8s lease releases the old hold first
            # (STOP_ATTACK) and completes as a fresh first click. Expiry
            # clears only the hold/burst timestamp — the already-armed
            # current cycle (_cycle_armed/_cycle_out) survives.
            if self._cycle_hold and now - self._cycle_last > .8:
                events.append(GestureEvent(GestureKind.STOP_ATTACK, now))
                self._cycle_hold = False
                self._cycle_last = None
            self._cycle_out, rose, fell = self._thumb_edges(right, self._cycle_out)
            if rose:
                self._cycle_armed = True
            if fell and self._cycle_armed:
                self._cycle_armed = False
                cycle_events, cycle_click = self._complete_cycle(now)
                events.extend(cycle_events)
                if cycle_click is not None:
                    click = cycle_click
        else:
            # Menu, hotbar session, incompatible pose or tracking loss:
            # release a promoted hold immediately and drop all burst/arm
            # state so nothing stale crosses the boundary.
            events.extend(self._stop_cycle(now))
        return PoseResult(tuple(keys), tuple(pulses), tuple(events), dx, dy,
                          cursor, click, scroll, labels, self._hotbar)
