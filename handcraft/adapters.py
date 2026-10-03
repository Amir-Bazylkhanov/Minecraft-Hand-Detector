"""Input adapters: the seam between the pure gesture engine and the OS.

``InputAdapter`` is the contract. ``RecordingAdapter`` and ``NullAdapter``
are used by tests and by practice mode; :mod:`handcraft.win32input`
provides the real SendInput-backed implementation on Windows.

Binding keys arrive as human-readable names from the palm-contact
recognizer (``"W"``, ``"SPACE"``, ``"E"`` ...); :func:`vk_for_key` maps
them to Win32 virtual-key codes so the controller stays layout agnostic.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

VK_E = 0x45
VK_W = 0x57
VK_SPACE = 0x20

# Named keys that cannot be derived from ord().
_NAMED_VK: Dict[str, int] = {
    "SPACE": VK_SPACE,
    "SHIFT": 0x10,
    "CTRL": 0x11,
    "CONTROL": 0x11,
    "ALT": 0x12,
    "TAB": 0x09,
    "ESCAPE": 0x1B,
    "ESC": 0x1B,
    "ENTER": 0x0D,
    "RETURN": 0x0D,
}


def vk_for_key(name: object) -> Optional[int]:
    """Map a binding key name (``"W"``, ``"SPACE"``, ``"F3"``) to a Win32
    virtual-key code. Returns ``None`` for anything unmappable — callers
    must skip unknown keys, never guess."""
    if name is None:
        return None
    text = str(name).strip().upper()
    if not text:
        return None
    if text in _NAMED_VK:
        return _NAMED_VK[text]
    if len(text) == 1 and ("A" <= text <= "Z" or "0" <= text <= "9"):
        return ord(text)
    if text.startswith("F") and text[1:].isdigit():
        number = int(text[1:])
        if 1 <= number <= 12:
            return 0x70 + number - 1  # VK_F1 .. VK_F12
    return None


class InputAdapter:
    """Interface for low-level input injection. Implementations must make
    ``left_mouse_down`` and ``key_down`` idempotent (repeated calls keep
    the button/key held, they must not produce clicks/repeats) and
    ``release_all`` total."""

    def left_mouse_down(self) -> None:
        raise NotImplementedError

    def left_mouse_up(self) -> None:
        raise NotImplementedError

    def key_down(self, vk: int) -> None:
        raise NotImplementedError

    def key_up(self, vk: int) -> None:
        raise NotImplementedError

    def tap_key(self, vk: int) -> None:
        self.key_down(vk)
        self.key_up(vk)

    def service_pending(self) -> None:
        """Release any timed pulses; called by the GUI without blocking."""

    def cancel_pending(self) -> None:
        """Cancel timed pulses when movement is blocked or tracking is lost."""

    def move_relative(self, dx: int, dy: int) -> None:
        raise NotImplementedError

    def move_cursor(self, x: float, y: float) -> None:
        raise NotImplementedError

    def click_mouse(self, button: str) -> None:
        raise NotImplementedError

    def scroll(self, slots: int) -> None:
        raise NotImplementedError

    def release_all(self) -> None:
        self.left_mouse_up()


class NullAdapter(InputAdapter):
    """Swallows everything. Default for practice mode and non-Windows."""

    def left_mouse_down(self) -> None:
        pass

    def left_mouse_up(self) -> None:
        pass

    def key_down(self, vk: int) -> None:
        pass

    def key_up(self, vk: int) -> None:
        pass

    def tap_key(self, vk: int) -> None:
        pass

    def move_relative(self, dx: int, dy: int) -> None:
        pass

    def move_cursor(self, x: float, y: float) -> None:
        pass

    def click_mouse(self, button: str) -> None:
        pass

    def scroll(self, slots: int) -> None:
        pass


class RecordingAdapter(InputAdapter):
    """Records every call for tests and for the practice-mode log."""

    def __init__(self) -> None:
        self.calls: List[Tuple[str, object]] = []
        self.left_held = False
        self.keys_held: Set[int] = set()

    def left_mouse_down(self) -> None:
        self.calls.append(("left_down", None))
        self.left_held = True

    def left_mouse_up(self) -> None:
        self.calls.append(("left_up", None))
        self.left_held = False

    def key_down(self, vk: int) -> None:
        self.calls.append(("key_down", vk))
        self.keys_held.add(vk)

    def key_up(self, vk: int) -> None:
        self.calls.append(("key_up", vk))
        self.keys_held.discard(vk)

    def tap_key(self, vk: int) -> None:
        self.calls.append(("tap_key", vk))

    def release_all(self) -> None:
        self.left_mouse_up()
        for vk in sorted(self.keys_held):
            self.key_up(vk)

    def kinds(self) -> List[str]:
        return [k for k, _ in self.calls]

    def move_relative(self, dx: int, dy: int) -> None:
        self.calls.append(("move_relative", (dx, dy)))

    def move_cursor(self, x: float, y: float) -> None:
        self.calls.append(("move_cursor", (x, y)))

    def click_mouse(self, button: str) -> None:
        self.calls.append(("click_mouse", button))

    def scroll(self, slots: int) -> None:
        self.calls.append(("scroll", slots))
