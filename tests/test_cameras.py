"""Camera discovery/selection tests: fully mocked device listings.

No camera is ever opened and no frame is captured: ``_raw_enumerate`` and
``_backend_ids`` are patched, so these tests run anywhere. Covers the
Xiaomi-phone scenarios: laptop+phone listing, phone visible only via MSMF,
phone disconnected (no fallback), device reordering (stable identity),
saved-selection persistence, and enumeration errors.
"""

import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from handcraft import cameras
from handcraft.cameras import CameraDevice

DSHOW = 700
MSMF = 1400

LAPTOP_PATH = ("\\\\?\\usb#vid_3277&pid_0059&mi_00#7&17e4d989&0&0000#"
               "{65e8773d-8f56-11d0-a3b9-00a0c9223196}\\global")
LAPTOP_PATH_MSMF = ("\\\\?\\usb#vid_3277&pid_0059&mi_00#7&17e4d989&0&0000#"
                    "{e5323777-f976-4f5b-9b55-b94699c46e44}\\global")
PHONE_PATH = ("\\\\?\\swd#vcamdevapi#f16a59c8925a43653ec407d2ca54450b6c1aa2b"
              "786bf6eda554201cebc2c3ddf#{65e8773d-8f56-11d0-a3b9-00a0c922319"
              "6}\\{fcebba03-9d13-4c13-9940-cc84fcd132d1}")
PHONE_PATH_MSMF = ("\\\\?\\swd#vcamdevapi#f16a59c8925a43653ec407d2ca54450b6c1"
                   "aa2b786bf6eda554201cebc2c3ddf#{e5323777-f976-4f5b-9b55-b9"
                   "9b55-b94699c46e44}\\{fcebba03-9d13-4c13-9940-cc84fcd132d1}")


def _info(index, name, path="", vid=None, pid=None, backend=DSHOW):
    return types.SimpleNamespace(index=index, name=name, path=path,
                                 vid=vid, pid=pid, backend=backend)


def _laptop(index):
    return _info(index, "ASUS FHD webcam", LAPTOP_PATH, 0x3277, 0x0059)


def _phone(index):
    return _info(index, "Xiaomi 17 (connected camera)", PHONE_PATH)


def _run_enum(listing):
    """listing: {backend_id: [CameraInfo...]}. Returns the merged devices."""
    with mock.patch.object(cameras, "_backend_ids",
                           return_value=(("DSHOW", DSHOW), ("MSMF", MSMF))), \
            mock.patch.object(cameras, "_raw_enumerate",
                              side_effect=lambda api: list(
                                  listing.get(api, []))):
        return cameras.enumerate_camera_devices()


class EnumerationTests(unittest.TestCase):
    def test_laptop_and_phone_listed_with_real_names_and_backends(self):
        devices = _run_enum({
            DSHOW: [_laptop(0), _phone(1)],
            MSMF: [_info(0, "ASUS FHD webcam", LAPTOP_PATH_MSMF, 0x3277, 0x0059),
                   _info(1, "Xiaomi 17 (connected camera)", PHONE_PATH_MSMF)],
        })
        names = [d.name for d in devices]
        self.assertIn("ASUS FHD webcam", names)
        self.assertIn("Xiaomi 17 (connected camera)", names)
        phone = next(d for d in devices if "Xiaomi" in d.name)
        self.assertEqual(phone.index, 1)
        self.assertEqual(phone.backend, DSHOW)

    def test_same_physical_device_deduped_across_backends(self):
        devices = _run_enum({
            DSHOW: [_laptop(0), _phone(1)],
            MSMF: [_info(0, "ASUS FHD webcam", LAPTOP_PATH_MSMF, 0x3277, 0x0059),
                   _info(1, "Xiaomi 17 (connected camera)", PHONE_PATH_MSMF)],
        })
        self.assertEqual(len(devices), 2,
                         "DSHOW+MSMF entries for one device must dedupe")
        self.assertTrue(all(d.backend == DSHOW for d in devices),
                        "DirectShow is preferred for duplicated devices")

    def test_phone_visible_only_via_msmf_still_found(self):
        devices = _run_enum({
            DSHOW: [_laptop(0)],
            MSMF: [_info(0, "ASUS FHD webcam", LAPTOP_PATH_MSMF, 0x3277, 0x0059),
                   _info(1, "Xiaomi 17 (connected camera)", PHONE_PATH_MSMF)],
        })
        phone = cameras.prefer_device(devices)
        self.assertIsNotNone(phone, "MSMF-only phone camera must be found")
        self.assertEqual(phone.backend, MSMF)
        self.assertEqual(phone.backend_name, "MSMF")
        self.assertEqual(phone.index, 1)

    def test_xiaomi_preferred_over_integrated_laptop(self):
        devices = _run_enum({DSHOW: [_laptop(0), _phone(1)], MSMF: []})
        chosen = cameras.prefer_device(devices)
        self.assertIn("Xiaomi", chosen.name)
        self.assertNotEqual(chosen.index, 0,
                            "index 0 is the laptop here; preference is by name")

    def test_no_xiaomi_means_no_default_not_laptop(self):
        devices = _run_enum({DSHOW: [_laptop(0)], MSMF: []})
        self.assertIsNone(cameras.prefer_device(devices),
                          "missing phone must not fall back to the laptop")


class IdentityResolutionTests(unittest.TestCase):
    def test_phone_disconnected_resolve_returns_none_no_fallback(self):
        phone = _run_enum({DSHOW: [_laptop(0), _phone(1)], MSMF: []})[1]
        saved = {"identity": phone.identity, "name": phone.name}
        now = _run_enum({DSHOW: [_laptop(0)], MSMF: []})  # phone unplugged
        self.assertIsNone(cameras.resolve_device(
            now, identity=saved["identity"], name=saved["name"]))

    def test_reordered_devices_resolve_by_stable_identity(self):
        before = _run_enum({DSHOW: [_laptop(0), _phone(1)], MSMF: []})
        phone = next(d for d in before if "Xiaomi" in d.name)
        # Re-enumeration with indices swapped (phone now index 0).
        after = _run_enum({DSHOW: [_phone(0), _laptop(1)], MSMF: []})
        resolved = cameras.resolve_device(after, identity=phone.identity)
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved.index, 0,
                         "identity resolution follows the device, "
                         "not the stale index")
        self.assertIn("Xiaomi", resolved.name)

    def test_resolve_falls_back_to_exact_name_when_path_changes(self):
        device = CameraDevice(index=3, backend=MSMF, backend_name="MSMF",
                              name="Xiaomi 17 (connected camera)", path="")
        resolved = cameras.resolve_device(
            [device], identity="path:\\\\?\\changed\\device",
            name="xiaomi 17 (CONNECTED camera)")
        self.assertIs(resolved, device)

    def test_resolution_never_matches_index_alone(self):
        devices = _run_enum({DSHOW: [_laptop(0), _phone(1)], MSMF: []})
        self.assertIsNone(cameras.resolve_device(
            devices, identity="path:\\\\?\\nonexistent", name="No such cam"))


class PersistenceTests(unittest.TestCase):
    def test_saved_selection_roundtrip_and_resolution(self):
        phone = next(d for d in _run_enum({DSHOW: [_laptop(0), _phone(1)],
                                           MSMF: []})
                     if "Xiaomi" in d.name)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "camera.json"
            cameras.save_camera(phone, path)
            saved = cameras.load_saved_camera(path)
        self.assertEqual(saved["identity"], phone.identity)
        self.assertEqual(saved["name"], phone.name)
        self.assertEqual(saved["backend"], phone.backend)
        self.assertEqual(saved["backend_name"], "DSHOW")
        # A fresh (reordered) listing still resolves the saved identity.
        after = _run_enum({DSHOW: [_phone(0), _laptop(1)], MSMF: []})
        resolved = cameras.resolve_device(
            after, identity=saved["identity"], name=saved["name"])
        self.assertIsNotNone(resolved)
        self.assertIn("Xiaomi", resolved.name)

    def test_load_missing_or_corrupt_config_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "nope.json"
            self.assertIsNone(cameras.load_saved_camera(missing))
            bad = Path(tmp) / "bad.json"
            bad.write_text("{not json", encoding="utf-8")
            self.assertIsNone(cameras.load_saved_camera(bad))
            empty = Path(tmp) / "empty.json"
            empty.write_text("{}", encoding="utf-8")
            self.assertIsNone(cameras.load_saved_camera(empty))


class EnumerationErrorTests(unittest.TestCase):
    def test_all_backends_failing_raises_actionable_error(self):
        with mock.patch.object(cameras, "_backend_ids",
                               return_value=(("DSHOW", DSHOW),
                                             ("MSMF", MSMF))), \
                mock.patch.object(cameras, "_raw_enumerate",
                                  side_effect=RuntimeError("backend dead")):
            with self.assertRaises(RuntimeError) as ctx:
                cameras.enumerate_camera_devices()
        self.assertIn("cv2-enumerate-cameras", str(ctx.exception))

    def test_one_failing_backend_still_returns_the_other(self):
        def raw(api):
            if api == DSHOW:
                raise RuntimeError("dshow broken")
            return [_info(1, "Xiaomi 17 (connected camera)", PHONE_PATH_MSMF)]

        with mock.patch.object(cameras, "_backend_ids",
                               return_value=(("DSHOW", DSHOW),
                                             ("MSMF", MSMF))), \
                mock.patch.object(cameras, "_raw_enumerate",
                                  side_effect=raw):
            devices = cameras.enumerate_camera_devices()
        self.assertEqual(len(devices), 1)
        self.assertEqual(devices[0].backend, MSMF)


if __name__ == "__main__":
    unittest.main()
