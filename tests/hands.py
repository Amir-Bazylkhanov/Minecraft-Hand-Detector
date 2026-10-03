"""Shared helpers: synthetic 21-landmark hands for tests."""

from __future__ import annotations

from typing import List, Sequence, Tuple

Point = Tuple[float, float, float]


def make_hand(kind: str = "open", wrist: Point = (0.0, 0.0, 0.0),
              scale: float = 1.0) -> List[Point]:
    """Build a synthetic MediaPipe-style 21-landmark hand.

    The palm runs along +y with palm size 0.2 * scale (wrist -> middle
    MCP). ``kind`` is "open" or "fist".
    """
    wx, wy, wz = wrist
    s = scale

    def p(x: float, y: float, z: float = 0.0) -> Point:
        return (wx + x * s, wy + y * s, wz + z * s)

    lm: List[Point] = [p(0, 0)] * 21
    lm[0] = p(0, 0)                       # wrist

    # Thumb (indices 1-4)
    if kind == "open":
        lm[1], lm[2], lm[3], lm[4] = p(-0.10, 0.05), p(-0.16, 0.09), p(-0.21, 0.12), p(-0.25, 0.15)
    else:
        lm[1], lm[2], lm[3], lm[4] = p(-0.08, 0.05), p(-0.06, 0.08), p(-0.05, 0.10), p(-0.05, 0.12)

    # Four fingers: (mcp, pip, dip, tip) with x offsets.
    offsets = {5: -0.06, 9: 0.0, 13: 0.06, 17: 0.11}
    for mcp, x in offsets.items():
        pip, dip, tip = mcp + 1, mcp + 2, mcp + 3
        lm[mcp] = p(x, 0.20)
        if kind == "open":
            lm[pip] = p(x, 0.32)
            lm[dip] = p(x, 0.40)
            lm[tip] = p(x, 0.47)
        else:  # fist: joints out, tip tucked back toward the wrist
            lm[pip] = p(x, 0.32)
            lm[dip] = p(x, 0.22)
            lm[tip] = p(x, 0.10)
    return lm


def moved(hand: Sequence[Point], dx: float = 0.0, dy: float = 0.0,
          dz: float = 0.0) -> List[Point]:
    """Translate a whole hand (moves the wrist for velocity tests)."""
    return [(x + dx, y + dy, z + dz) for x, y, z in hand]
