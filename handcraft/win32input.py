"""Real input injection via Win32 SendInput (ctypes).

Correct 64-bit signatures throughout: ``ULONG_PTR`` is ``c_size_t``,
``SendInput(UINT, LPINPUT, int) -> UINT`` with explicit argtypes/restype,
and the INPUT union is padded via HARDWAREINPUT so ``sizeof(INPUT)`` is
right on x64.

This module is Windows-only. It is imported lazily by the app so the rest
of the package (and the test suite) never touches it.
"""

from __future__ import annotations

import ctypes
import time
from ctypes import wintypes
from typing import Callable, Optional, Set

from .adapters import InputAdapter, VK_SPACE

JUMP_PRESS_S = 0.12

ULONG_PTR = ctypes.c_size_t  # ULONG_PTR on Win32/Win64

INPUT_MOUSE = 0
INPUT_KEYBOARD = 1

MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
KEYEVENTF_KEYUP = 0x0002


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class _INPUT_UNION(ctypes.Union):
    _fields_ = [
        ("mi", MOUSEINPUT),
        ("ki", KEYBDINPUT),
        ("hi", HARDWAREINPUT),
    ]


class INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [
        ("type", wintypes.DWORD),
        ("u", _INPUT_UNION),
    ]


_LPINPUT = ctypes.POINTER(INPUT)

_user32 = ctypes.windll.user32
_user32.SendInput.argtypes = (wintypes.UINT, _LPINPUT, ctypes.c_int)
_user32.SendInput.restype = wintypes.UINT


class SendInputError(OSError):
    """SendInput injected fewer events than requested (possibly none)."""


def _send(*inputs: INPUT) -> int:
    """Send a batch; returns how many were actually injected.

    Raises :class:`SendInputError` on a failed or partial send — callers
    must never assume a keystroke or click happened when it did not.
    """
    count = len(inputs)
    arr = (INPUT * count)(*inputs)
    sent = _user32.SendInput(count, arr, ctypes.sizeof(INPUT))
    if sent != count:
        raise SendInputError(f"SendInput injected {sent} of {count} events")
    return sent


def _mouse_event(flags: int, dx: int = 0, dy: int = 0, data: int = 0) -> INPUT:
    inp = INPUT()
    inp.type = INPUT_MOUSE
    inp.mi = MOUSEINPUT(dx, dy, data & 0xffffffff, flags, 0, 0)
    return inp


def _key_event(vk: int, flags: int) -> INPUT:
    inp = INPUT()
    inp.type = INPUT_KEYBOARD
    inp.ki = KEYBDINPUT(vk, 0, flags, 0, 0)
    return inp


class SendInputAdapter(InputAdapter):
    """Injects real mouse/keyboard input. Mouse-down and key-down are
    idempotent: an input is only pressed once and stays held until
    released. Held state is cleared only by a successful release, so a
    failed release never loses track of a still-stuck input."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._left_held = False
        self._held_keys: Set[int] = set()
        self._clock = clock
        self._pulse_deadlines: dict[int, float] = {}
        self._right_held = False
        self._click_deadlines: dict[str, float] = {}

    @property
    def left_held(self) -> bool:
        return self._left_held

    @property
    def right_held(self) -> bool:
        return self._right_held

    @property
    def held_keys(self) -> frozenset:
        """Virtual-key codes currently believed to be held down."""
        return frozenset(self._held_keys)

    def left_mouse_down(self) -> None:
        self._click_deadlines.pop("left", None)
        if self._left_held:
            return  # hold, never re-click
        _send(_mouse_event(MOUSEEVENTF_LEFTDOWN))
        self._left_held = True

    def left_mouse_up(self) -> None:
        # Always send the release, even if we think it is already up —
        # releasing an unheld button is harmless, a stuck button is not.
        # On failure the held flag survives so release_all can retry.
        _send(_mouse_event(MOUSEEVENTF_LEFTUP))
        self._left_held = False
        self._click_deadlines.pop("left", None)

    def key_down(self, vk: int) -> None:
        # An explicit hold takes ownership from any scheduled pulse.
        self._pulse_deadlines.pop(vk, None)
        if vk in self._held_keys:
            return  # hold, never re-press
        _send(_key_event(vk, 0))
        self._held_keys.add(vk)

    def key_up(self, vk: int) -> None:
        if vk not in self._held_keys:
            return
        # Only a successful send clears the held state.
        _send(_key_event(vk, KEYEVENTF_KEYUP))
        self._held_keys.discard(vk)
        self._pulse_deadlines.pop(vk, None)

    def tap_key(self, vk: int) -> None:
        if vk == VK_SPACE:
            # Minecraft polls jump state. Down+up in one SendInput batch
            # can disappear between game updates, so hold across updates.
            if vk in self._held_keys:
                return
            self.key_down(vk)
            self._pulse_deadlines[vk] = self._clock() + JUMP_PRESS_S
            return
        try:
            _send(_key_event(vk, 0), _key_event(vk, KEYEVENTF_KEYUP))
        except SendInputError:
            # The key may be stuck down: best-effort cleanup release.
            try:
                _send(_key_event(vk, KEYEVENTF_KEYUP))
            except SendInputError:
                # Cleanup failed too: track the key as possibly stuck so
                # release_all retries it later.
                self._held_keys.add(vk)
            else:
                self._held_keys.discard(vk)
            raise

    def service_pending(self) -> None:
        now = self._clock()
        for vk, deadline in list(self._pulse_deadlines.items()):
            if now >= deadline:
                self.key_up(vk)
        for button, deadline in list(self._click_deadlines.items()):
            if now >= deadline:
                self._release_click(button)

    def cancel_pending(self) -> None:
        for vk in list(self._pulse_deadlines):
            self.key_up(vk)
        for button in list(self._click_deadlines):
            self._release_click(button)

    def _release_click(self, button: str) -> None:
        if button == "left":
            self.left_mouse_up()
        else:
            _send(_mouse_event(MOUSEEVENTF_RIGHTUP))
            self._right_held = False
            self._click_deadlines.pop("right", None)

    def click_mouse(self, button: str) -> None:
        if button not in {"left", "right"}:
            raise ValueError("mouse button must be left or right")
        if self._left_held if button == "left" else self._right_held:
            return
        _send(_mouse_event(MOUSEEVENTF_LEFTDOWN if button == "left" else MOUSEEVENTF_RIGHTDOWN))
        if button == "left":
            self._left_held = True
        else:
            self._right_held = True
        self._click_deadlines[button] = self._clock() + 0.10

    def move_relative(self, dx: int, dy: int) -> None:
        if dx or dy:
            _send(_mouse_event(0x0001, int(dx), int(dy)))

    def move_cursor(self, x: float, y: float) -> None:
        # Map pointing inside the focused Minecraft client, including on
        # a second monitor. Absolute SendInput uses virtual desktop space.
        _user32.GetForegroundWindow.restype = wintypes.HWND
        _user32.GetClientRect.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.RECT))
        _user32.ClientToScreen.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.POINT))
        hwnd = _user32.GetForegroundWindow()
        rect, origin = wintypes.RECT(), wintypes.POINT()
        if not hwnd or not _user32.GetClientRect(hwnd, ctypes.byref(rect)) or not _user32.ClientToScreen(hwnd, ctypes.byref(origin)):
            raise SendInputError("Cannot locate Minecraft client for inventory cursor")
        width, height = _user32.GetSystemMetrics(78), _user32.GetSystemMetrics(79)
        if width <= 1 or height <= 1:
            raise SendInputError("Invalid desktop size")
        px = origin.x + max(0., min(1., x)) * max(0, rect.right - 1)
        py = origin.y + max(0., min(1., y)) * max(0, rect.bottom - 1)
        ax = round((px - _user32.GetSystemMetrics(76)) * 65535 / (width - 1))
        ay = round((py - _user32.GetSystemMetrics(77)) * 65535 / (height - 1))
        _send(_mouse_event(0x0001 | 0x8000 | 0x4000, max(0, min(65535, ax)), max(0, min(65535, ay))))

    def scroll(self, slots: int) -> None:
        if slots:
            _send(_mouse_event(0x0800, data=int(slots) * 120))

    def release_all(self) -> None:
        # Attempt every release even if one fails; report afterwards.
        first_error: Optional[SendInputError] = None
        for vk in list(self._held_keys):
            try:
                self.key_up(vk)
            except SendInputError as exc:
                if first_error is None:
                    first_error = exc
        if self._left_held:
            try:
                self.left_mouse_up()
            except SendInputError as exc:
                if first_error is None:
                    first_error = exc
        if self._right_held:
            try:
                self._release_click("right")
            except SendInputError as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error
