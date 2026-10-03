"""Background camera + hand-tracking worker.

Runs on its own thread. The model file is downloaded with urllib the
first time the user explicitly presses Start (never at import, never at
GUI launch); the download is chunked, time-limited and cancellable via a
``stop_event`` so pressing Stop/F8 during preparation aborts it and the
camera never starts afterwards.

Frames are mirrored AFTER detection: inference runs on the raw camera
image and the frame is flipped only for the preview. The emitted
landmarks have their x coordinate mirrored back (``1.0 - x``) so they
line up with the mirrored preview, while MediaPipe's handedness
categories ("Left"/"Right") are preserved exactly as detected.
Results are emitted as a ``{"left": [...], "right": [...]}`` dict keyed by
those categories (never by detection list order), each value holding 21
``(x, y, z)`` landmark tuples. The monotonic capture timestamp is taken
right after ``cap.read()`` and before inference, so the app can discard
packets older than 300 ms before acting on them.

Frames go to a bounded "latest frame wins" queue so the UI can never fall
behind. cv2 and mediapipe are imported lazily inside the worker so that
importing this module (tests, GUI on machines without the deps) stays
cheap and the heavy native libraries only load when tracking is actually
started.
"""

from __future__ import annotations

import os
import queue
import threading
import time
import urllib.request
from typing import Callable, Dict, List, Optional, Tuple

from .geometry import Point

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_FILENAME = "hand_landmarker.task"

DOWNLOAD_TIMEOUT_S = 30.0
_DOWNLOAD_CHUNK = 1 << 16

# A detection result packet: (hands_dict_or_None, monotonic_capture_timestamp).
# hands dict: "left"/"right" -> 21 (x, y, z) tuples; None means no camera frame.
ResultPacket = Tuple[Optional[Dict[str, List[Point]]], float]


class DownloadCancelled(Exception):
    """Raised by ensure_model when the caller's stop_event fires."""


def ensure_model(models_dir: str,
                 status: Callable[[str], None] = lambda m: None,
                 stop_event: Optional[threading.Event] = None,
                 timeout: float = DOWNLOAD_TIMEOUT_S) -> str:
    """Return the local path of hand_landmarker.task, downloading it via
    urllib if missing. Only ever called on an explicit Start.

    The download is chunked and checks ``stop_event`` between chunks, so
    Stop/F8 during preparation cancels it (raising :class:`DownloadCancelled`)
    instead of completing and starting the camera behind the user's back.
    """
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, MODEL_FILENAME)
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return path
    if stop_event is not None and stop_event.is_set():
        raise DownloadCancelled()
    status(f"Downloading hand_landmarker.task (~8 MB) ...")
    tmp = path + ".part"
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=timeout) as response, \
                open(tmp, "wb") as out:
            while True:
                if stop_event is not None and stop_event.is_set():
                    raise DownloadCancelled()
                chunk = response.read(_DOWNLOAD_CHUNK)
                if not chunk:
                    break
                out.write(chunk)
        os.replace(tmp, path)
    except Exception:
        if os.path.isfile(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise
    status("Model downloaded.")
    return path


class CameraWorker(threading.Thread):
    """Capture -> HandLandmarker(detect_for_video, num_hands=2)
    -> mirror (preview + landmark x) -> result queue.

    The capture timestamp in each result packet is taken before inference.
    ``detect_for_video`` timestamps derive from that capture timestamp in
    milliseconds and are clamped to be strictly increasing
    (``max(previous + 1, capture_ms)``), as required by the VIDEO running
    mode. Inference always runs at full camera resolution; only the
    cosmetic preview is resized, and preview generation is throttled to
    ``preview_max_fps`` while inference and result publication never are.
    """

    def __init__(self, camera_index: int, model_path: str,
                 frame_queue: "queue.Queue[bytes]",
                 result_queue: "queue.Queue[ResultPacket]",
                 status: Callable[[str], None],
                 preview_width: int = 480,
                 camera_backend: Optional[int] = None,
                 camera_name: Optional[str] = None):
        super().__init__(daemon=True, name="handcraft-camera")
        self.camera_index = camera_index
        self.camera_backend = camera_backend  # None -> CAP_DSHOW at open time
        self.camera_name = camera_name        # for actionable status text
        self.model_path = model_path
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.status = status
        self.preview_width = preview_width
        self.preview_max_fps = 30.0  # cosmetic preview only; never inference
        self._stop_event = threading.Event()
        self._last_ts_ms: Optional[int] = None
        self._last_preview_at = 0.0
        # Latest per-frame timings in milliseconds, measured with
        # time.monotonic(); read-only snapshot for whoever wants it.
        self.latest_timings: Dict[str, float] = {
            "capture_read_ms": 0.0,
            "inference_ms": 0.0,
            "preview_ms": 0.0,
            "result_age_ms": 0.0,
        }
        # Latest control-zone snapshot assigned by the main app each tick;
        # read by reference at draw time, never mutated here.
        self.control_overlay_state: dict = {}

    def stop(self) -> None:
        self._stop_event.set()

    # ------------------------------------------------------------------ run

    def run(self) -> None:
        try:
            self._run()
        except Exception as exc:  # never let the thread die silently
            self.status(f"ERROR: tracking stopped: {exc}")

    def _run(self) -> None:
        import cv2  # lazy: heavy native deps only when actually tracking
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision

        from .handmap import draw_landmark_overlay
        from .control_overlay import draw_control_overlay

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        backend = self.camera_backend
        if backend is None:
            backend = cv2.CAP_DSHOW
        cap = cv2.VideoCapture(self.camera_index, backend)
        if not cap.isOpened():
            cap.release()
            label = self.camera_name or f"index {self.camera_index}"
            self.status(
                f"ERROR: cannot open camera {label!r} "
                f"(index {self.camera_index}, backend {int(backend)}). "
                "It may be disconnected or in use by another app - press "
                "Refresh, reselect the camera, then Start again.")
            return
        try:
            # Best effort: a driver/backend that does not support this
            # property must not fail an otherwise working camera.
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        self.status("Tracking running.")
        self._last_ts_ms = None
        # Prime so the first frame always yields a preview.
        self._last_preview_at = time.monotonic() - self._preview_interval()
        try:
            with vision.HandLandmarker.create_from_options(options) as landmarker:
                while not self._stop_event.is_set():
                    read_start = time.monotonic()
                    ok, frame = cap.read()
                    captured_at = time.monotonic()  # before any inference
                    capture_read_ms = (captured_at - read_start) * 1000.0
                    if not ok:
                        self._put_result(None, captured_at)
                        self._publish_timings(capture_read_ms, 0.0, 0.0,
                                              captured_at)
                        time.sleep(0.02)
                        continue

                    capture_ms = int(captured_at * 1000)
                    if self._last_ts_ms is None:
                        ts_ms = capture_ms
                    else:
                        ts_ms = max(self._last_ts_ms + 1, capture_ms)
                    self._last_ts_ms = ts_ms

                    # Detect on the raw (unmirrored) frame at FULL camera
                    # resolution; the preview is mirrored/resized only after
                    # the result has been published.
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB, data=rgb)
                    inference_start = time.monotonic()
                    result = landmarker.detect_for_video(mp_image, ts_ms)
                    inference_ms = (time.monotonic() - inference_start) * 1000.0

                    hands = self._hands_by_handedness(result)
                    published_at = time.monotonic()
                    self._put_result(hands, captured_at)

                    preview_ms = 0.0
                    now = time.monotonic()
                    if now - self._last_preview_at >= self._preview_interval():
                        self._last_preview_at = now
                        preview_start = now
                        # Shrink the raw RGB frame BEFORE flip/overlays so all
                        # cosmetic work runs at final preview dimensions.
                        rgb = self._resize_for_preview(rgb)
                        rgb = cv2.flip(rgb, 1)
                        if hands:
                            try:
                                draw_landmark_overlay(rgb, hands)
                            except Exception:
                                pass  # overlay is cosmetic; never drop a frame
                        try:
                            draw_control_overlay(
                                rgb, hands, self.control_overlay_state)
                        except Exception:
                            pass  # overlay is cosmetic; never drop a frame
                        self._put_frame(self._encode_ppm(rgb))
                        preview_ms = (time.monotonic() - preview_start) * 1000.0

                    self._publish_timings(capture_read_ms, inference_ms,
                                          preview_ms, captured_at,
                                          published_at)
        finally:
            cap.release()
            self.status("Tracking stopped.")

    # -------------------------------------------------------------- helpers

    def _preview_interval(self) -> float:
        fps = self.preview_max_fps
        return 1.0 / fps if fps > 0 else float("inf")

    def _publish_timings(self, capture_read_ms: float, inference_ms: float,
                         preview_ms: float, captured_at: float,
                         published_at: Optional[float] = None) -> None:
        if published_at is None:
            published_at = time.monotonic()
        self.latest_timings = {
            "capture_read_ms": capture_read_ms,
            "inference_ms": inference_ms,
            "preview_ms": preview_ms,
            "result_age_ms": (published_at - captured_at) * 1000.0,
        }

    def _resize_for_preview(self, rgb_frame):
        """Downscale raw RGB to preview_width before any cosmetic work."""
        import cv2
        h, w = rgb_frame.shape[:2]
        if w > self.preview_width:
            scale = self.preview_width / w
            return cv2.resize(
                rgb_frame, (self.preview_width, int(h * scale)))
        return rgb_frame

    @staticmethod
    def _hands_by_handedness(result) -> Dict[str, List[Point]]:
        """Build {"left"/"right": 21 points} from handedness categories,
        never from detection list order. On a duplicate label the higher
        confidence entry wins.

        Detection runs on the unmirrored frame, so each point's x is
        mirrored (``1.0 - lm.x``) to match the mirrored preview; y and z
        are kept as-is and labels are preserved unchanged."""
        hands: Dict[str, List[Point]] = {}
        scores: Dict[str, float] = {}
        if not result.hand_landmarks or not result.handedness:
            return hands
        for landmarks, categories in zip(result.hand_landmarks,
                                         result.handedness):
            if not categories:
                continue
            category = categories[0]
            label = str(getattr(category, "category_name", "")).strip().lower()
            if label not in ("left", "right"):
                continue
            score = float(getattr(category, "score", 0.0))
            if label in hands and scores[label] >= score:
                continue
            hands[label] = [(1.0 - lm.x, lm.y, lm.z) for lm in landmarks]
            scores[label] = score
        return hands

    def _encode_ppm(self, rgb_frame) -> bytes:
        import cv2
        h, w = rgb_frame.shape[:2]
        if w > self.preview_width:
            scale = self.preview_width / w
            rgb_frame = cv2.resize(
                rgb_frame, (self.preview_width, int(h * scale)))
            h, w = rgb_frame.shape[:2]
        header = f"P6 {w} {h} 255\n".encode("ascii")
        return header + rgb_frame.tobytes()

    def _put_latest(self, q: "queue.Queue", item) -> None:
        """Bounded latest-wins put: drop the oldest item if full."""
        try:
            q.put_nowait(item)
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            try:
                q.put_nowait(item)
            except queue.Full:
                pass

    def _put_result(self, hands: Optional[Dict[str, List[Point]]],
                    captured_at: float) -> None:
        self._put_latest(self.result_queue, (hands, captured_at))

    def _put_frame(self, ppm: bytes) -> None:
        self._put_latest(self.frame_queue, ppm)
