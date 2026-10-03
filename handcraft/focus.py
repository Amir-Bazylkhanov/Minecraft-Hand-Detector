"""Foreground-window focus guard and emergency hotkey.

Two layers:

* :func:`evaluate_focus` — pure decision logic (title + exe name), fully
  unit-testable. Fail closed: anything that is not *both* a Minecraft
  title *and* a Java runtime process is rejected.
* Windows glue via ctypes — GetForegroundWindow / GetWindowTextW /
  GetWindowThreadProcessId / QueryFullProcessImageNameW, plus
  GetAsyncKeyState for the global F8 emergency stop. Any OS error or
  exception yields "not focused" / "not pressed".
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

MINECRAFT_TITLE_NEEDLE = "minecraft"
JAVA_PROCESS_NAMES = frozenset({"javaw.exe", "java.exe"})

VK_F8 = 0x77


def evaluate_focus(title: Optional[str], exe_name: Optional[str]) -> bool:
    """True only if the foreground window looks like Minecraft Java
    Edition: title mentions Minecraft *and* the owning process is
    javaw.exe / java.exe. Fail closed on missing data."""
    if not title or not exe_name:
        return False
    base = os.path.basename(exe_name.strip().lower())
    return MINECRAFT_TITLE_NEEDLE in title.lower() and base in JAVA_PROCESS_NAMES


def is_windows() -> bool:
    return sys.platform == "win32"


# ---------------------------------------------------------------- Win32 glue

if is_windows():
    import ctypes
    from ctypes import wintypes

    _user32 = ctypes.windll.user32
    _kernel32 = ctypes.windll.kernel32

    _user32.GetForegroundWindow.argtypes = ()
    _user32.GetForegroundWindow.restype = wintypes.HWND
    _user32.GetWindowTextW.argtypes = (wintypes.HWND, wintypes.LPWSTR, ctypes.c_int)
    _user32.GetWindowTextW.restype = ctypes.c_int
    _user32.GetWindowThreadProcessId.argtypes = (
        wintypes.HWND, ctypes.POINTER(wintypes.DWORD))
    _user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    _user32.GetAsyncKeyState.argtypes = (ctypes.c_int,)
    _user32.GetAsyncKeyState.restype = wintypes.SHORT

    _PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    _kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    _kernel32.OpenProcess.restype = wintypes.HANDLE
    _kernel32.QueryFullProcessImageNameW.argtypes = (
        wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD))
    _kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
    _kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    _kernel32.CloseHandle.restype = wintypes.BOOL


def foreground_window_info() -> Tuple[Optional[str], Optional[str]]:
    """(title, exe_path) of the foreground window, or (None, None).

    Never raises — every failure path returns (None, None) so callers
    fail closed.
    """
    if not is_windows():
        return None, None
    try:
        hwnd = _user32.GetForegroundWindow()
        if not hwnd:
            return None, None
        buf = ctypes.create_unicode_buffer(512)
        _user32.GetWindowTextW(hwnd, buf, len(buf))
        title = buf.value or None

        pid = wintypes.DWORD(0)
        _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        exe = None
        if pid.value:
            handle = _kernel32.OpenProcess(
                _PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
            if handle:
                try:
                    path_buf = ctypes.create_unicode_buffer(1024)
                    size = wintypes.DWORD(len(path_buf))
                    if _kernel32.QueryFullProcessImageNameW(
                            handle, 0, path_buf, ctypes.byref(size)):
                        exe = path_buf.value or None
                finally:
                    _kernel32.CloseHandle(handle)
        return title, exe
    except Exception:
        return None, None


def minecraft_java_focused() -> bool:
    """Fail-closed live check: is the foreground window Minecraft Java?"""
    title, exe = foreground_window_info()
    return evaluate_focus(title, exe)


def emergency_stop_pressed() -> bool:
    """True while F8 is physically held (global, works without focus)."""
    if not is_windows():
        return False
    try:
        return bool(_user32.GetAsyncKeyState(VK_F8) & 0x8000)
    except Exception:
        return False
