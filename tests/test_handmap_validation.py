"""Handmap save-validation tests.

Everything is mocked/faked: no camera is opened, no model is downloaded, no
Tk root or real widgets are created (a bare panel instance with fake editor
widgets), and no real OS input is ever sent.  The binding config lives in a
workspace-local temporary JSON file that is removed afterwards.
"""

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path

from handcraft import handmap
from handcraft.bindings import DEFAULT_BINDINGS, PalmContactRecognizer

WORKSPACE = Path(__file__).resolve().parent.parent


class FakeEntry:
    """Stand-in for the panel's key entry / action combobox widgets."""

    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


class FakeMessagebox:
    """Records showerror/showinfo calls instead of opening dialogs."""

    def __init__(self):
        self.errors = []
        self.infos = []

    def showerror(self, title, message):
        self.errors.append((title, message))

    def showinfo(self, title, message):
        self.infos.append((title, message))


class HandmapSaveValidationTests(unittest.TestCase):
    def setUp(self):
        # Workspace-local temp file: no directories — exec sandboxes may deny
        # writes inside freshly created temp dirs.
        fd, tmp_path = tempfile.mkstemp(prefix="handmap_test_", suffix=".json",
                                        dir=WORKSPACE)
        os.close(fd)
        self.config_path = Path(tmp_path)
        self.addCleanup(lambda: self.config_path.unlink(missing_ok=True))
        self.messagebox = FakeMessagebox()
        self._real_messagebox = handmap.messagebox
        handmap.messagebox = self.messagebox
        self.addCleanup(self._restore_messagebox)

    def _restore_messagebox(self):
        handmap.messagebox = self._real_messagebox

    def _make_panel(self, bindings=None):
        """Bare panel instance with fake widgets — no Tk, no camera."""
        bindings = copy.deepcopy(DEFAULT_BINDINGS if bindings is None else bindings)
        saved = []
        panel = object.__new__(handmap.GestureConfigPanel)
        panel.config_path = self.config_path
        panel.config_data = {"version": handmap.CONFIG_VERSION, "bindings": bindings}
        panel._editors = {
            gesture_id: (
                FakeEntry(str(binding.get("action", "hold")) if isinstance(binding, dict) else "hold"),
                FakeEntry(str(binding.get("key", "")) if isinstance(binding, dict) else ""),
            )
            for gesture_id, binding in bindings.items()
        }
        # The live recognizer only ever holds previously-validated bindings;
        # config_data can carry malformed entries from a hand-edited JSON.
        panel._live_recognizer = PalmContactRecognizer(
            bindings=copy.deepcopy(DEFAULT_BINDINGS))
        panel._on_bindings_changed = lambda b: saved.append(copy.deepcopy(b))
        panel._refresh_command_text = lambda: None  # needs the real Text widget
        return panel, saved

    def _write_json(self, bindings):
        payload = {"version": handmap.CONFIG_VERSION, "bindings": bindings}
        self.config_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def test_invalid_banana_action_preserves_everything(self):
        bindings = copy.deepcopy(DEFAULT_BINDINGS)
        self._write_json(bindings)
        before_bytes = self.config_path.read_bytes()
        panel, saved = self._make_panel(bindings)
        before_config = copy.deepcopy(panel.config_data)
        before_recognizer = panel._live_recognizer

        # Fake editor returns an action the readonly combobox can't produce.
        panel._editors["left_index_palm"][0].value = "BANANA"
        panel.save()

        self.assertEqual(len(self.messagebox.errors), 1, "exactly one error dialog")
        self.assertIn("action", self.messagebox.errors[0][1])
        self.assertEqual(self.messagebox.infos, [], "no success dialog on rejection")
        self.assertEqual(saved, [], "callback must not fire on invalid input")
        self.assertEqual(self.config_path.read_bytes(), before_bytes,
                         "JSON file must be left untouched")
        self.assertEqual(panel.config_data, before_config,
                         "in-memory config_data must be preserved")
        self.assertIs(panel._live_recognizer, before_recognizer,
                      "live recognizer must be preserved")

    def test_unknown_banana_key_preserves_everything(self):
        bindings = copy.deepcopy(DEFAULT_BINDINGS)
        self._write_json(bindings)
        before_bytes = self.config_path.read_bytes()
        panel, saved = self._make_panel(bindings)
        before_config = copy.deepcopy(panel.config_data)
        before_recognizer = panel._live_recognizer

        panel._editors["left_thumb_palm"][1].value = "BANANA"  # no VK mapping
        panel.save()

        self.assertEqual(len(self.messagebox.errors), 1)
        self.assertIn("virtual-key", self.messagebox.errors[0][1])
        self.assertEqual(saved, [])
        self.assertEqual(self.config_path.read_bytes(), before_bytes)
        self.assertEqual(panel.config_data, before_config)
        self.assertIs(panel._live_recognizer, before_recognizer)

    def test_malformed_binding_value_preserves_everything(self):
        bindings = copy.deepcopy(DEFAULT_BINDINGS)
        # A non-mapping binding smuggled in via the loaded config.
        bindings["broken"] = "not-a-binding"
        self._write_json(bindings)
        before_bytes = self.config_path.read_bytes()
        panel, saved = self._make_panel(bindings)
        before_config = copy.deepcopy(panel.config_data)
        before_recognizer = panel._live_recognizer

        panel.save()

        self.assertEqual(len(self.messagebox.errors), 1)
        self.assertEqual(saved, [])
        self.assertEqual(self.config_path.read_bytes(), before_bytes)
        self.assertEqual(panel.config_data, before_config)
        self.assertIs(panel._live_recognizer, before_recognizer)

    def test_valid_w_space_save_persists_and_notifies(self):
        bindings = copy.deepcopy(DEFAULT_BINDINGS)
        panel, saved = self._make_panel(bindings)
        before_recognizer = panel._live_recognizer
        panel._editors["left_index_palm"][1].value = "w"      # hold W (walk)
        panel._editors["left_thumb_palm"][1].value = "space"  # pulse SPACE (jump)

        panel.save()

        self.assertEqual(self.messagebox.errors, [], "valid save must not error")
        self.assertEqual(len(self.messagebox.infos), 1, "success dialog shown")
        self.assertEqual(len(saved), 1, "callback fired exactly once")
        saved_bindings = saved[0]
        self.assertEqual(saved_bindings["left_index_palm"]["key"], "W",
                         "key is normalized to upper case")
        self.assertEqual(saved_bindings["left_index_palm"]["action"], "hold")
        self.assertEqual(saved_bindings["left_thumb_palm"]["key"], "SPACE")
        self.assertEqual(saved_bindings["left_thumb_palm"]["action"], "pulse")
        self.assertIsNot(panel._live_recognizer, before_recognizer,
                         "live recognizer rebuilt from saved bindings")
        self.assertIsInstance(panel._live_recognizer, PalmContactRecognizer)

        # The persisted JSON round-trips and is accepted by the recognizer.
        loaded = handmap.load_binding_config(self.config_path)
        self.assertEqual(loaded["bindings"]["left_index_palm"]["key"], "W")
        self.assertEqual(loaded["bindings"]["left_thumb_palm"]["key"], "SPACE")
        PalmContactRecognizer(bindings=loaded["bindings"])
        # Panel state and file agree with the callback payload.
        self.assertEqual(panel.config_data["bindings"], saved_bindings)


if __name__ == "__main__":
    unittest.main()
