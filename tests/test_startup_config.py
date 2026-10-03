"""Startup binding-config hardening for ``HandCraftApp._reload_recognizer``.

Regression tests for the startup crash: ``__init__`` calls
``_reload_recognizer`` BEFORE the log widget exists, so a corrupt or
invalid ``handcraft_bindings.json`` used to raise (``_log_action`` hit a
missing ``self.log``, or the recognizer constructor's ``ValueError``
escaped the try block). These tests exercise the real method on a bare
stub app — no Tk root, no camera, no mediapipe.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from handcraft import handmap
from handcraft.adapters import vk_for_key
from handcraft.app import HandCraftApp
from handcraft.bindings import DEFAULT_BINDINGS, PalmContactRecognizer


def _make_stub_app(config_path, with_log=False):
    """Bare HandCraftApp stand-in: only what _reload_recognizer touches.

    Without ``with_log`` there is deliberately NO ``log`` attribute,
    mirroring the pre-``_build_ui`` state during ``__init__``.
    """
    app = SimpleNamespace(
        _handmap=handmap,
        _config_path=Path(config_path),
        dispatcher=SimpleNamespace(recognizer=None),
    )
    app._config_file_mtime = HandCraftApp._config_file_mtime.__get__(app)
    app._reload_recognizer = HandCraftApp._reload_recognizer.__get__(app)
    if with_log:
        app.log = object()  # sentinel: the log widget "exists"
        app.logged = []
        app._log_action = app.logged.append
    return app


def _write_config(path, payload):
    Path(path).write_text(json.dumps(payload), encoding="utf-8")


class ReloadRecognizerInvalidConfigTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.config_path = Path(self._tmp.name) / "handcraft_bindings.json"

    def _assert_default_recognizer(self, app):
        recognizer = app.recognizer
        self.assertIsInstance(recognizer, PalmContactRecognizer)
        self.assertIs(app.dispatcher.recognizer, recognizer)
        gesture_ids = {binding.gesture_id for binding in recognizer.bindings}
        self.assertEqual(gesture_ids, set(DEFAULT_BINDINGS))
        # Default bindings are all mappable to real virtual keys.
        for binding in recognizer.bindings:
            self.assertIsNotNone(vk_for_key(binding.key))
        self.assertEqual(app._config_mtime, os.path.getmtime(self.config_path))

    def test_malformed_json_defers_warning_and_keeps_defaults(self):
        self.config_path.write_text("{not valid json", encoding="utf-8")
        before = self.config_path.read_bytes()
        app = _make_stub_app(self.config_path)  # no log attribute

        app._reload_recognizer()

        self._assert_default_recognizer(app)
        self.assertTrue(getattr(app, "_defer_log", ""))
        self.assertIn("invalid", app._defer_log)
        self.assertIn("defaults", app._defer_log)
        # The corrupt file is never overwritten.
        self.assertEqual(self.config_path.read_bytes(), before)

    def test_malformed_json_via_loader_patch(self):
        self.config_path.write_text("{}", encoding="utf-8")
        app = _make_stub_app(self.config_path)
        app._handmap = mock.Mock(wraps=handmap)
        app._handmap.load_binding_config.side_effect = ValueError("bad json")

        app._reload_recognizer()

        self._assert_default_recognizer(app)
        self.assertIn("bad json", app._defer_log)

    def test_invalid_hand_schema_falls_back_to_defaults(self):
        _write_config(self.config_path, {
            "version": 1,
            "bindings": {
                "evil": {"hand": "foot", "finger": "index",
                         "action": "hold", "key": "W"},
            },
        })
        before = self.config_path.read_bytes()
        app = _make_stub_app(self.config_path)

        app._reload_recognizer()

        self._assert_default_recognizer(app)
        self.assertIn("invalid", app._defer_log)
        self.assertEqual(self.config_path.read_bytes(), before)

    def test_unsupported_key_banana_without_log_attr(self):
        _write_config(self.config_path, {
            "version": 1,
            "bindings": {
                "banana_jump": {"hand": "left", "finger": "thumb",
                                "action": "pulse", "key": "BANANA"},
            },
        })
        self.assertIsNone(vk_for_key("BANANA"))  # premise: unsupported
        app = _make_stub_app(self.config_path)  # no log attribute

        app._reload_recognizer()

        self._assert_default_recognizer(app)
        self.assertIn("BANANA", app._defer_log)
        # The unsupported key never lands in the live recognizer.
        keys = {binding.key for binding in app.recognizer.bindings}
        self.assertNotIn("BANANA", keys)

    def test_warning_logged_directly_when_log_widget_exists(self):
        self.config_path.write_text("{not valid json", encoding="utf-8")
        app = _make_stub_app(self.config_path, with_log=True)

        app._reload_recognizer()

        self._assert_default_recognizer(app)
        self.assertEqual(len(app.logged), 1)
        self.assertIn("invalid", app.logged[0])
        self.assertFalse(hasattr(app, "_defer_log"))

    def test_valid_saved_bindings_load_without_warning(self):
        _write_config(self.config_path, {
            "version": 1,
            "bindings": {
                "right_pinky_inventory": {
                    "hand": "right", "finger": "pinky",
                    "action": "pulse", "key": "E",
                    "description": "Pulse E (inventory).",
                },
            },
        })
        app = _make_stub_app(self.config_path)

        app._reload_recognizer()

        recognizer = app.recognizer
        self.assertIs(app.dispatcher.recognizer, recognizer)
        self.assertFalse(hasattr(app, "_defer_log"))
        by_id = {binding.gesture_id: binding for binding in recognizer.bindings}
        # Saved binding is merged in alongside the untouched defaults.
        self.assertIn("right_pinky_inventory", by_id)
        self.assertTrue(set(DEFAULT_BINDINGS) <= set(by_id))
        self.assertEqual(by_id["right_pinky_inventory"].key, "E")
        self.assertEqual(by_id["right_pinky_inventory"].action, "pulse")

    def test_missing_config_file_loads_defaults_silently(self):
        app = _make_stub_app(self.config_path)  # file never created

        app._reload_recognizer()

        recognizer = app.recognizer
        gesture_ids = {binding.gesture_id for binding in recognizer.bindings}
        self.assertEqual(gesture_ids, set(DEFAULT_BINDINGS))
        self.assertFalse(hasattr(app, "_defer_log"))
        self.assertIsNone(app._config_mtime)


if __name__ == "__main__":
    unittest.main()
