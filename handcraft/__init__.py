"""Minecraft Hand Detector - Minecraft Java Edition hand-gesture companion app.

Pure-python package. The only modules that import third-party / OS-specific
libraries are ``tracker`` (cv2 + mediapipe, lazily imported inside the worker)
and ``win32input`` / ``focus`` (ctypes Win32). Everything else is unit-testable
without a camera, a display, or real input injection.
"""

__version__ = "0.1.0"
