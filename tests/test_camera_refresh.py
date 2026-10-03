"""Regression tests for camera refresh re-selection.

A Refresh re-enumerates cameras and can return them in a DIFFERENT order.
The previously selected camera must be re-found by its stable identity,
never by its old dropdown position — an old index mapped onto the new list
would silently switch to the laptop webcam when the phone moves position.

Everything runs with a mocked dropdown: no camera is opened and no real
Tk window is created.
"""

import unittest
from pathlib import Path
from unittest import mock

from handcraft import cameras
from handcraft.app import HandCraftApp


def make_device(name, path, index=0):
    return cameras.CameraDevice(
        index=index, backend=700, backend_name="MSMF", name=name, path=path)


XIAOMI_PATH = "\\\\?\\usb#vid_2717&pid_ff48#6&2a7b#{guid-xiaomi}"
ASUS_PATH = "\\\\?\\usb#vid_13d3&pid_56eb#5&1c4f#{guid-asus}"


def xiaomi(index=0):
    return make_device("Xiaomi connected camera", XIAOMI_PATH, index)


def asus(index=0):
    return make_device("ASUS Integrated Camera", ASUS_PATH, index)


class _FakeCombo:
    """Stand-in for the ttk.Combobox with a textvariable: current() is the
    index of the variable's text inside values (-1 when the text matches no
    entry), so clearing the variable also clears the position — exactly how
    the real widget behaves."""

    def __init__(self, var):
        self._var = var
        self.values = ()

    def configure(self, values=()):
        self.values = tuple(values)

    def current(self, idx=None):
        if idx is None:
            text = self._var.get()
            return self.values.index(text) if text in self.values else -1
        self._var.set(self.values[idx] if 0 <= idx < len(self.values)
                      else "")


class _Var:
    def __init__(self, value=""):
        self._value = value

    def get(self):
        return self._value

    def set(self, value):
        self._value = value


def make_app(old_devices, old_idx):
    app = object.__new__(HandCraftApp)
    app.worker = None
    app._preparing = False
    app.log_lines = []
    app.status_lines = []
    app._log_action = app.log_lines.append
    app._set_status = lambda level, text: app.status_lines.append(
        (level, text))
    app._camera_config_path = Path("nonexistent_test_camera.json")
    app._camera_devices = list(old_devices)
    app.camera_var = _Var(
        old_devices[old_idx].display_name if old_idx >= 0 else "")
    app.camera_box = _FakeCombo(app.camera_var)
    app.camera_box.configure(
        values=[d.display_name for d in old_devices])
    return app


class RefreshReselectTests(unittest.TestCase):
    def test_reordered_list_keeps_xiaomi_by_identity(self):
        # Old order: Xiaomi at 0, ASUS at 1. New order: ASUS at 0, Xiaomi
        # at 1. The old dropdown position (0) now points at ASUS.
        app = make_app([xiaomi(0), asus(1)], 0)
        new_devices = [asus(0), xiaomi(1)]
        with mock.patch.object(cameras, "enumerate_camera_devices",
                               return_value=new_devices), \
                mock.patch.object(cameras, "save_camera") as save:
            app._refresh_cameras()
        selected = app._selected_camera_device()
        self.assertIsNotNone(selected)
        self.assertEqual(selected.identity, xiaomi().identity,
                         "Refresh must retain the Xiaomi by identity, not "
                         "switch to whatever now sits at the old position")
        self.assertEqual(app.camera_box.current(), 1)

    def test_disconnected_xiaomi_never_selects_asus(self):
        # Xiaomi selected, then unplugged: new list is ASUS only, sitting
        # at the Xiaomi's old position 0.
        app = make_app([xiaomi(0), asus(1)], 0)
        with mock.patch.object(cameras, "enumerate_camera_devices",
                               return_value=[asus(0)]), \
                mock.patch.object(cameras, "save_camera") as save:
            app._refresh_cameras()
        self.assertIsNone(app._selected_camera_device(),
                          "a missing phone must leave NO selection — never "
                          "map the old position onto the ASUS webcam")
        self.assertEqual(app.camera_var.get(), "")
        save.assert_not_called()
        self.assertTrue(any(level == "alert" and "not found" in text
                            for level, text in app.status_lines),
                        "expected an actionable 'not found' status")

    def test_enumeration_failure_clears_selection(self):
        app = make_app([xiaomi(0)], 0)
        with mock.patch.object(cameras, "enumerate_camera_devices",
                               side_effect=RuntimeError("no backend")):
            app._refresh_cameras()
        self.assertEqual(app._camera_devices, [])
        self.assertIsNone(app._selected_camera_device())
        self.assertEqual(app.camera_var.get(), "")
        self.assertTrue(any(level == "alert" and "enumeration failed"
                            in text.lower()
                            for level, text in app.status_lines))


class CurrentCameraRequestShellTests(unittest.TestCase):
    """Bare-shell path (no _camera_devices attribute): explicit numeric
    index only, no silent fallback on unparseable input."""

    def make_shell(self, text):
        app = object.__new__(HandCraftApp)
        app.camera_var = _Var(text)
        return app

    def test_explicit_numeric_zero_still_works(self):
        self.assertEqual(self.make_shell("0")._current_camera_request(), 0)

    def test_invalid_input_returns_none(self):
        self.assertIsNone(self.make_shell("not-a-number")
                          ._current_camera_request())
        self.assertIsNone(self.make_shell("")._current_camera_request())


if __name__ == "__main__":
    unittest.main()
