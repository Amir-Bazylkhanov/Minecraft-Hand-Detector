"""Latency-focused tests for handcraft.tracker.CameraWorker.

Everything here runs on fake capture/model doubles injected via sys.modules;
no real camera, model file, input hooking, or network is touched. The fakes
prove the latency-oriented contract of the worker loop:

* a throttled/skipped preview still publishes every inference result,
* overlays are drawn on the final preview-sized (already resized + mirrored)
  frame while inference keeps the full camera resolution,
* detect_for_video timestamps are strictly increasing,
* an unsupported CAP_PROP_BUFFERSIZE never fails the camera.
"""

import queue
import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

from handcraft.tracker import CameraWorker


class FakeFrame:
    """Minimal stand-in for an OpenCV/numpy frame."""

    def __init__(self, width, height, flipped=False):
        self.shape = (height, width, 3)
        self.flipped = flipped

    def tobytes(self):
        h, w = self.shape[:2]
        return bytes(w * h * 3)


class FakeCapture:
    def __init__(self, frame, set_error=None):
        self.frame = frame
        self.set_error = set_error
        self.set_calls = []
        self.released = False

    def isOpened(self):
        return True

    def set(self, prop, value):
        self.set_calls.append((prop, value))
        if self.set_error is not None:
            raise self.set_error
        return True

    def read(self):
        return True, self.frame

    def release(self):
        self.released = True


class FakeModel:
    """Shared state for the fake HandLandmarker."""

    def __init__(self, stop_after, worker):
        self.stop_after = stop_after
        self.worker = worker
        self.calls = 0
        self.timestamps = []
        self.images = []
        self.result = SimpleNamespace(
            hand_landmarks=[
                [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(21)]
            ],
            handedness=[[SimpleNamespace(category_name="Left", score=0.9)]],
        )


def _make_fake_cv2(capture, call_log):
    cv2 = types.ModuleType("cv2")
    cv2.CAP_DSHOW = 700
    cv2.CAP_PROP_BUFFERSIZE = 38
    cv2.COLOR_BGR2RGB = 4
    cv2.VideoCapture = lambda index, backend: capture

    def cvt_color(frame, code):
        call_log.append(("cvtColor", frame.shape))
        return frame

    def flip(frame, axis):
        call_log.append(("flip", frame.shape))
        return FakeFrame(frame.shape[1], frame.shape[0], flipped=True)

    def resize(frame, size):
        call_log.append(("resize", frame.shape))
        w, h = size
        return FakeFrame(w, h, flipped=frame.flipped)

    cv2.cvtColor = cvt_color
    cv2.flip = flip
    cv2.resize = resize
    return cv2


def _make_fake_mediapipe(model):
    mp = types.ModuleType("mediapipe")
    tasks = types.ModuleType("mediapipe.tasks")
    python = types.ModuleType("mediapipe.tasks.python")
    vision = types.ModuleType("mediapipe.tasks.python.vision")

    class Image:
        def __init__(self, image_format=None, data=None):
            model.images.append(data)
            self.data = data

    class BaseOptions:
        def __init__(self, model_asset_path=None):
            self.model_asset_path = model_asset_path

    class HandLandmarkerOptions:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _Landmarker:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def detect_for_video(self, image, ts_ms):
            model.calls += 1
            model.timestamps.append(ts_ms)
            if model.calls >= model.stop_after:
                model.worker.stop()
            return model.result

    class HandLandmarker:
        @staticmethod
        def create_from_options(options):
            return _Landmarker()

    mp.Image = Image
    mp.ImageFormat = SimpleNamespace(SRGB="SRGB")
    mp.tasks = tasks
    tasks.python = python
    python.BaseOptions = BaseOptions
    python.vision = vision
    vision.HandLandmarkerOptions = HandLandmarkerOptions
    vision.RunningMode = SimpleNamespace(VIDEO="VIDEO")
    vision.HandLandmarker = HandLandmarker
    return {
        "mediapipe": mp,
        "mediapipe.tasks": tasks,
        "mediapipe.tasks.python": python,
        "mediapipe.tasks.python.vision": vision,
    }


def _drain(q):
    items = []
    while True:
        try:
            items.append(q.get_nowait())
        except queue.Empty:
            return items


class WorkerHarness:
    """Runs CameraWorker.run() synchronously against fake cv2/mediapipe."""

    def __init__(self, *, frames=5, preview_width=480, preview_max_fps=30.0,
                 frame_size=(640, 480), set_error=None, fake_time=None):
        self.frame_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.statuses = []
        self.worker = CameraWorker(
            0, "model.task", self.frame_queue, self.result_queue,
            self.statuses.append, preview_width=preview_width)
        self.worker.preview_max_fps = preview_max_fps
        self.model = FakeModel(stop_after=frames, worker=self.worker)
        self.capture = FakeCapture(
            FakeFrame(*frame_size), set_error=set_error)
        self.cv2_log = []
        self.drawn = []          # (which_overlay, shape, flipped)
        self.result_published_before_draw = []
        self.fake_time = fake_time

    def run(self):
        modules = {"cv2": _make_fake_cv2(self.capture, self.cv2_log)}
        modules.update(_make_fake_mediapipe(self.model))

        def fake_landmark_overlay(frame, hands):
            self.drawn.append(("landmark", frame.shape, frame.flipped))
            self.result_published_before_draw.append(
                not self.result_queue.empty())
            return frame

        def fake_control_overlay(frame, hands, state):
            self.drawn.append(("control", frame.shape, frame.flipped))
            self.result_published_before_draw.append(
                not self.result_queue.empty())

        with mock.patch.dict(sys.modules, modules), \
                mock.patch("handcraft.handmap.draw_landmark_overlay",
                           fake_landmark_overlay), \
                mock.patch("handcraft.control_overlay.draw_control_overlay",
                           fake_control_overlay):
            if self.fake_time is not None:
                with mock.patch("handcraft.tracker.time", self.fake_time):
                    self.worker.run()
            else:
                self.worker.run()
        self.results = _drain(self.result_queue)
        self.frames = _drain(self.frame_queue)
        return self


class PreviewThrottleTest(unittest.TestCase):
    def test_skipped_preview_still_publishes_every_inference(self):
        # Effectively "never preview again after the first frame".
        harness = WorkerHarness(frames=5, preview_max_fps=1e-9).run()

        self.assertEqual(self._errors(harness), [])
        self.assertEqual(harness.model.calls, 5)
        # Every inference produced a result packet...
        self.assertEqual(len(harness.results), 5)
        for hands, captured_at in harness.results:
            self.assertIsNotNone(hands)
            self.assertIn("left", hands)
        # ...but only the first cosmetic preview was generated.
        self.assertEqual(len(harness.frames), 1)
        self.assertEqual(len(harness.drawn), 2)  # landmark + control, once

    @staticmethod
    def _errors(harness):
        return [s for s in harness.statuses if s.startswith("ERROR")]


class PreviewGeometryTest(unittest.TestCase):
    def test_overlays_draw_on_final_preview_size_after_single_mirror(self):
        harness = WorkerHarness(frames=2, preview_width=320,
                                frame_size=(640, 480)).run()

        # Inference image kept the full camera resolution.
        self.assertTrue(harness.model.images)
        for image in harness.model.images:
            self.assertEqual(image.shape, (480, 640, 3))

        # Resize happened on the raw frame BEFORE flip and before drawing.
        kinds = [kind for kind, _ in harness.cv2_log]
        first_resize = kinds.index("resize")
        first_flip = kinds.index("flip")
        self.assertLess(first_resize, first_flip)
        resize_inputs = [shape for kind, shape in harness.cv2_log
                         if kind == "resize"]
        self.assertTrue(all(shape == (480, 640, 3) for shape in resize_inputs))
        flip_inputs = [shape for kind, shape in harness.cv2_log
                       if kind == "flip"]
        self.assertTrue(all(shape == (240, 320, 3) for shape in flip_inputs))

        # Both overlays operated on the final preview-sized, mirrored frame.
        self.assertTrue(harness.drawn)
        for _, shape, flipped in harness.drawn:
            self.assertEqual(shape, (240, 320, 3))
            self.assertTrue(flipped)  # mirrored exactly once

        # The encoded PPM carries the preview dimensions.
        ppm = harness.frames[-1]
        header = b"P6 320 240 255\n"
        self.assertTrue(ppm.startswith(header))
        self.assertEqual(len(ppm), len(header) + 320 * 240 * 3)

    def test_result_published_before_any_cosmetic_work(self):
        harness = WorkerHarness(frames=2).run()
        self.assertTrue(harness.drawn)
        self.assertTrue(all(harness.result_published_before_draw))


class TimestampTest(unittest.TestCase):
    def test_detect_timestamps_strictly_increasing(self):
        # Frozen clock: every capture lands on the same millisecond, so the
        # max(previous + 1, capture_ms) clamp is what must keep timestamps
        # strictly increasing for VIDEO running mode.
        fake_time = SimpleNamespace(
            monotonic=lambda: 1000.0,
            monotonic_ns=lambda: 1_000_000_000_000,
            sleep=lambda seconds: None,
        )
        harness = WorkerHarness(frames=4, fake_time=fake_time).run()

        timestamps = harness.model.timestamps
        self.assertEqual(len(timestamps), 4)
        capture_ms = int(1000.0 * 1000)
        self.assertEqual(
            timestamps,
            [capture_ms, capture_ms + 1, capture_ms + 2, capture_ms + 3])
        for earlier, later in zip(timestamps, timestamps[1:]):
            self.assertLess(earlier, later)


class BufferSizeTest(unittest.TestCase):
    def test_unsupported_buffersize_does_not_fail_camera(self):
        harness = WorkerHarness(
            frames=3, set_error=RuntimeError("property not supported")).run()

        self.assertEqual(
            [s for s in harness.statuses if s.startswith("ERROR")], [])
        self.assertIn("Tracking running.", harness.statuses)
        self.assertEqual(harness.model.calls, 3)
        self.assertEqual(len(harness.results), 3)
        self.assertTrue(harness.capture.released)
        self.assertIn((38, 1), harness.capture.set_calls)  # CAP_PROP_BUFFERSIZE


class TimingsTest(unittest.TestCase):
    def test_latest_timings_exposed_and_monotonic_measured(self):
        harness = WorkerHarness(frames=3).run()

        timings = harness.worker.latest_timings
        self.assertEqual(
            set(timings),
            {"capture_read_ms", "inference_ms", "preview_ms", "result_age_ms"})
        for value in timings.values():
            self.assertIsInstance(value, float)
            self.assertGreaterEqual(value, 0.0)


if __name__ == "__main__":
    unittest.main()
