"""Cosmetic camera control-zone overlay drawn on the mirrored preview.

``draw_control_overlay(rgb, hands, state)`` paints the movement joystick
zone (left hand), the look zone (right hand), the inventory wrist-cursor
zone, hotbar slot hints, per-hand finger labels and a mode caption onto
the already-mirrored RGB preview frame. cv2 is imported lazily inside the draw call so importing this
module stays cheap and testable without native deps. All drawing uses
thin lines and small text backgrounds - it must never fill the frame
opaquely, and the caller wraps the call so a cosmetic failure never
drops a frame.

All coordinates are normalized and already mirrored (matching the
preview), exactly as emitted by CameraWorker. Hand labels ("left"/
"right") are used as given; they are never mirrored, swapped or
re-derived here.

State snapshot keys (all optional, missing means "draw nothing extra"):
    left_center:    (x, y) | None - movement joystick neutral center
    right_center:   (x, y) | None - look neutral center
    cursor_center:  (x, y) | None - inventory cursor neutral center
    held_keys:      tuple of currently held key names ("W", "A", ...)
    labels:         {"left": str, "right": str} finger/pose labels
    inventory_open: bool - hide world zones, show the wrist cursor zone
    hotbar_active:  bool - annotate thumb/pinky tips as slot steppers
    left_ready:     bool - False shows the "hold still" centering caption
    cursor_ready:   bool - False shows the cursor centering caption

Each valid hand also gets a raw thumb-gap readout (OUT / FOLDED /
BETWEEN plus the gap ratio) on its pose-label top row. This is pure
detection feedback so the user can watch the thumb classification live;
it never reflects or claims OS input state.
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

from .geometry import (INDEX_MCP, THUMB_TIP, Point, dist, palm_size)

try:
    # Modern shared thumb-gap helper and thresholds (preferred).
    from .geometry import (THUMB_GAP_EXTENDED, THUMB_GAP_FOLDED,
                           thumb_gap_ratio as _thumb_gap_ratio)
except ImportError:  # fallback while the shared geometry helper lands
    THUMB_GAP_EXTENDED = 0.65
    THUMB_GAP_FOLDED = 0.55
    _thumb_gap_ratio = None

# Deadzone half-extents in palm-size units (match posemap thresholds).
ENTER_DEADZONE = 0.40    # left joystick engage box, x and y
RELEASE_DEADZONE = 0.25  # inner release box (dotted outline)
LOOK_DEADZONE = 0.20     # right-hand look deadzone
CURSOR_DEADZONE = 0.20   # inventory wrist-cursor deadzone

# Colors are RGB (the preview frame is RGB, not BGR).
ACTIVE_COLOR = (60, 230, 60)     # bright green: actively held key
NEUTRAL_COLOR = (210, 210, 210)  # light gray: idle key / generic text
LEFT_ZONE_COLOR = (255, 200, 60)   # warm yellow: movement joystick
RIGHT_ZONE_COLOR = (60, 210, 255)  # cyan: look zone (distinct from left)
THUMB_BETWEEN_COLOR = (250, 225, 50)  # yellow: ambiguous thumb gap
TEXT_BG = (0, 0, 0)

_FONT_SCALE = 0.4
_THICKNESS = 1


def _valid(hand) -> bool:
    return (hand is not None and len(hand) == 21
            and all(len(p) == 3 and all(math.isfinite(v) for v in p)
                    for p in hand)
            and palm_size(hand) >= 1e-4)


def _clamp(v: float, lo: float, hi: float) -> int:
    return int(min(max(v, lo), hi))


def _text(cv2, img, text: str, x: float, y: float, color) -> None:
    """Small text with a small opaque background, clamped into frame.

    Bounds use the real measured size from cv2.getTextSize (font, scale
    and thickness) plus its baseline, so neither the glyphs nor their
    background can leave the image even at the edges."""
    h, w = img.shape[:2]
    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, _FONT_SCALE, _THICKNESS)
    x = _clamp(x, 0, max(0, w - tw - 4))
    y = _clamp(y, th + 4, h - baseline - 4)
    cv2.rectangle(img, (x - 2, y - th - 2),
                  (x + tw + 2, y + baseline + 2), TEXT_BG, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, _FONT_SCALE,
                color, _THICKNESS, cv2.LINE_AA)


def _measure(cv2, text: str):
    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, _FONT_SCALE, _THICKNESS)
    return tw, th, baseline


def _thumb_gap(hand) -> float:
    """Thumb tip -> index MCP distance normalized by palm size."""
    if _thumb_gap_ratio is not None:
        return _thumb_gap_ratio(hand)
    return dist(hand[THUMB_TIP], hand[INDEX_MCP]) / palm_size(hand)


def _thumb_feedback(hand):
    """(word, gap, color) raw thumb readout for a valid hand.

    Matches the shared thumb_hysteresis comparisons exactly: gap strictly
    above THUMB_GAP_EXTENDED is OUT, strictly below THUMB_GAP_FOLDED is
    FOLDED, and the exact boundaries land in BETWEEN. Pure detection
    feedback: it reports what the tracker sees, never any OS input state.
    """
    gap = _thumb_gap(hand)
    if gap > THUMB_GAP_EXTENDED:
        return "OUT", gap, ACTIVE_COLOR
    if gap < THUMB_GAP_FOLDED:
        return "FOLDED", gap, NEUTRAL_COLOR
    return "BETWEEN", gap, THUMB_BETWEEN_COLOR


def _dotted_rect(cv2, img, x0: int, y0: int, x1: int, y1: int,
                 color, dash: int = 4) -> None:
    for x in range(x0, x1, dash * 2):
        cv2.line(img, (x, y0), (min(x + dash, x1), y0), color, _THICKNESS)
        cv2.line(img, (x, y1), (min(x + dash, x1), y1), color, _THICKNESS)
    for y in range(y0, y1, dash * 2):
        cv2.line(img, (x0, y), (x0, min(y + dash, y1)), color, _THICKNESS)
        cv2.line(img, (x1, y), (x1, min(y + dash, y1)), color, _THICKNESS)


def _draw_left_zone(cv2, rgb, hand, state, w: int, h: int,
                    held: frozenset) -> None:
    palm = palm_size(hand)
    wx, wy = hand[0][0] * w, hand[0][1] * h
    cv2.circle(rgb, (int(wx), int(wy)), 4, LEFT_ZONE_COLOR, _THICKNESS)

    center = state.get("left_center")
    if center is not None:
        cx, cy = center[0] * w, center[1] * h
        ex, ey = ENTER_DEADZONE * palm * w, ENTER_DEADZONE * palm * h
        cv2.rectangle(rgb, (int(cx - ex), int(cy - ey)),
                      (int(cx + ex), int(cy + ey)),
                      LEFT_ZONE_COLOR, _THICKNESS)
        rx, ry = RELEASE_DEADZONE * palm * w, RELEASE_DEADZONE * palm * h
        _dotted_rect(cv2, rgb, int(cx - rx), int(cy - ry),
                     int(cx + rx), int(cy + ry), LEFT_ZONE_COLOR)
        # Fixed neutral anchor (small cross) vs current wrist (marker):
        # the displacement line makes the offset the joystick reads.
        icx, icy = int(cx), int(cy)
        cv2.line(rgb, (icx - 5, icy), (icx + 5, icy),
                 LEFT_ZONE_COLOR, _THICKNESS)
        cv2.line(rgb, (icx, icy - 5), (icx, icy + 5),
                 LEFT_ZONE_COLOR, _THICKNESS)
        cv2.line(rgb, (icx, icy), (int(wx), int(wy)),
                 LEFT_ZONE_COLOR, _THICKNESS)
        # Directional axes arrows from the neutral center.
        cv2.arrowedLine(rgb, (icx, icy), (icx, int(cy - ey)),
                        LEFT_ZONE_COLOR, _THICKNESS)
        cv2.arrowedLine(rgb, (icx, icy), (icx, int(cy + ey)),
                        LEFT_ZONE_COLOR, _THICKNESS)
        cv2.arrowedLine(rgb, (icx, icy), (int(cx - ex), icy),
                        LEFT_ZONE_COLOR, _THICKNESS)
        cv2.arrowedLine(rgb, (icx, icy), (int(cx + ex), icy),
                        LEFT_ZONE_COLOR, _THICKNESS)

        def key_color(key):
            return ACTIVE_COLOR if key in held else NEUTRAL_COLOR

        # W/S centered on cx; A fully outside the left box edge, D
        # fully outside the right edge, each with an 8 px gap.
        tw, _, _ = _measure(cv2, "W Forward")
        _text(cv2, rgb, "W Forward", cx - tw / 2, cy - ey - 6,
              key_color("W"))
        tw, _, _ = _measure(cv2, "S Back")
        _text(cv2, rgb, "S Back", cx - tw / 2, cy + ey + 14,
              key_color("S"))
        tw, _, _ = _measure(cv2, "A Left")
        _text(cv2, rgb, "A Left", cx - ex - 8 - tw, cy + 4, key_color("A"))
        _text(cv2, rgb, "D Right", cx + ex + 8, cy + 4, key_color("D"))

    if not state.get("left_ready", True):
        _text(cv2, rgb, "Hold point still to center", wx + 8, wy - 10,
              LEFT_ZONE_COLOR)

    # Jump hint just left of the left thumb tip (landmark 4). The caption
    # reflects the real held state: green "held" while SPACE is down,
    # neutral guidance otherwise. Width is measured from the full dynamic
    # caption so neither variant overlaps the S Back caption below the box.
    if "SPACE" in held:
        caption, color = "JUMP: SPACE held", ACTIVE_COLOR
    else:
        caption, color = "JUMP: thumb out", NEUTRAL_COLOR
    tw, _, _ = _measure(cv2, caption)
    _text(cv2, rgb, caption, hand[4][0] * w - tw - 8, hand[4][1] * h, color)


def _draw_right_zone(cv2, rgb, hand, state, w: int, h: int) -> None:
    palm = palm_size(hand)
    wx, wy = hand[0][0] * w, hand[0][1] * h
    cv2.circle(rgb, (int(wx), int(wy)), 4, RIGHT_ZONE_COLOR, _THICKNESS)

    center = state.get("right_center")
    if center is not None:
        cx, cy = center[0] * w, center[1] * h
        dx, dy = LOOK_DEADZONE * palm * w, LOOK_DEADZONE * palm * h
        cv2.rectangle(rgb, (int(cx - dx), int(cy - dy)),
                      (int(cx + dx), int(cy + dy)),
                      RIGHT_ZONE_COLOR, _THICKNESS)
        _text(cv2, rgb, "LOOK", cx + dx + 6, cy, RIGHT_ZONE_COLOR)


def _draw_cursor_zone(cv2, rgb, hand, state, w: int, h: int) -> None:
    """Inventory cursor: right-wrist marker plus the deadzone box around
    the entry center, in the same compact language as the look zone."""
    palm = palm_size(hand)
    wx, wy = hand[0][0] * w, hand[0][1] * h
    cv2.circle(rgb, (int(wx), int(wy)), 4, RIGHT_ZONE_COLOR, _THICKNESS)

    center = state.get("cursor_center")
    if center is not None:
        cx, cy = center[0] * w, center[1] * h
        dx, dy = CURSOR_DEADZONE * palm * w, CURSOR_DEADZONE * palm * h
        cv2.rectangle(rgb, (int(cx - dx), int(cy - dy)),
                      (int(cx + dx), int(cy + dy)),
                      RIGHT_ZONE_COLOR, _THICKNESS)
        # Fixed entry center (small cross) vs the current wrist marker.
        icx, icy = int(cx), int(cy)
        cv2.line(rgb, (icx - 5, icy), (icx + 5, icy),
                 RIGHT_ZONE_COLOR, _THICKNESS)
        cv2.line(rgb, (icx, icy - 5), (icx, icy + 5),
                 RIGHT_ZONE_COLOR, _THICKNESS)
        _text(cv2, rgb, "CURSOR", cx + dx + 6, cy, RIGHT_ZONE_COLOR)

    if not state.get("cursor_ready", True):
        caption = "Hold point still to center"
        tw, _, baseline = _measure(cv2, caption)
        if center is not None:
            # Above the deadzone box, centered on it: the baseline sits
            # clear of the box top so it never meets the CURSOR label.
            _text(cv2, rgb, caption, cx - tw / 2,
                  cy - dy - baseline - 4, RIGHT_ZONE_COLOR)
        else:
            _text(cv2, rgb, caption, wx + 8, wy - 10, RIGHT_ZONE_COLOR)


def draw_control_overlay(rgb, hands: Mapping[str, Sequence[Point]] | None,
                         state: dict | None) -> None:
    """Draw control-zone hints on the mirrored RGB preview, in place.

    Best-effort and side-effect free beyond painting on ``rgb``; raises
    nothing on its own for missing/invalid hands or a sparse ``state``.
    """
    import cv2  # lazy: native dep only when actually drawing

    if rgb is None:
        return
    state = state or {}
    hands = hands or {}
    h, w = rgb.shape[:2]
    held = frozenset(state.get("held_keys") or ())
    labels = state.get("labels") or {}
    inventory = bool(state.get("inventory_open"))

    left, right = hands.get("left"), hands.get("right")
    left_ok, right_ok = _valid(left), _valid(right)

    # Top rows, stacked so they never overlap each other or the zones:
    # row 1 compact hint caption, rows 2/3 per-hand pose status labels.
    if inventory:
        caption = "Point, then move wrist; thumb out-in to click"
    else:
        caption = "Move your WRIST; open hand to recenter"
    tw, _, _ = _measure(cv2, caption)
    _text(cv2, rgb, caption, w / 2 - tw / 2, 18, NEUTRAL_COLOR)
    if left_ok and labels.get("left"):
        _text(cv2, rgb, str(labels["left"]), 6, 40, LEFT_ZONE_COLOR)
    if right_ok and labels.get("right"):
        _text(cv2, rgb, str(labels["right"]), 6, 62, RIGHT_ZONE_COLOR)

    # Raw thumb-gap feedback shares the per-hand label rows (40/62),
    # starting just right of the pose label - or at the row start when
    # no label is present - so rows never collide with the caption or
    # each other. Missing/invalid hands get no readout.
    for tag, hand, ok, row in (("L", left, left_ok, 40),
                               ("R", right, right_ok, 62)):
        if not ok:
            continue
        word, gap, color = _thumb_feedback(hand)
        x = 6
        label = labels.get("left" if tag == "L" else "right")
        if label:
            x += _measure(cv2, str(label))[0] + 14
        _text(cv2, rgb, f"{tag} thumb {word} {gap:.2f}", x, row, color)

    if inventory:
        # Menus own the cursor: world joystick/look zones stay hidden and
        # only the right wrist cursor zone is shown.
        if right_ok:
            _draw_cursor_zone(cv2, rgb, right, state, w, h)
    else:
        # Joystick/look zones are hidden while menus own the cursor.
        if left_ok:
            _draw_left_zone(cv2, rgb, left, state, w, h, held)
        if right_ok:
            _draw_right_zone(cv2, rgb, right, state, w, h)

    # Right thumb hint beside the right thumb tip (landmark 4): attack
    # cycles in gameplay, clicks in inventory. Hidden while hotbar mode
    # owns the thumb (folded thumb steps slots there, never attacks).
    if right_ok and not state.get("hotbar_active"):
        caption = "CLICK: thumb out-in" if inventory else "ATTACK: thumb out-in"
        _text(cv2, rgb, caption, right[4][0] * w + 8, right[4][1] * h,
              RIGHT_ZONE_COLOR)

    if state.get("hotbar_active") and right_ok:
        _text(cv2, rgb, "Prev slot", right[4][0] * w + 6,
              right[4][1] * h - 12, RIGHT_ZONE_COLOR)
        _text(cv2, rgb, "Next slot", right[20][0] * w + 6,
              right[20][1] * h + 12, RIGHT_ZONE_COLOR)

    mode = "inventory" if inventory else "gameplay"
    keys = "+".join(state.get("held_keys") or ()) or "-"
    _text(cv2, rgb, f"Mode: {mode}  keys: {keys}", 6, h - 8, NEUTRAL_COLOR)
