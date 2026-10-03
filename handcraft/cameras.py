"""Named camera discovery, selection and persistence (Windows).

Uses cv2-enumerate-cameras to list real device names and OS device paths
per backend (DirectShow and Media Foundation). Enumeration NEVER opens a
camera or captures a frame — it only queries the backend device registries.

Devices are identified by a stable identity derived from the OS device path
(never by numeric index, which drifts when devices are reordered or
re-attached, and never by backend+index alone). The same physical device
exposed by both DSHOW and MSMF is deduplicated (DirectShow preferred, since
it is the proven capture path here); MSMF entries are kept for devices that
DirectShow does not expose — e.g. Windows connected-phone virtual cameras.

The initial-launch default is a Xiaomi-family device matched by explicit
device name. There is deliberately NO positional fallback: a missing phone
must never silently select the integrated laptop webcam — the user picks a
named device from the full-name list instead.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

CAMERA_CONFIG_FILE_NAME = "handcraft_camera.json"

# Explicit-name keywords for the initial-launch default (Xiaomi family).
PREFERRED_NAME_KEYWORDS = ("xiaomi", "redmi", "poco")


def _backend_ids() -> Tuple[Tuple[str, int], ...]:
    """(name, OpenCV apiPreference) pairs to enumerate, in preference order.

    DirectShow first (proven capture path), then Media Foundation so
    connected-phone virtual cameras that only exist under MSMF still appear.
    """
    import cv2  # lazy: keeps module import cheap for tests/headless use
    return (("DSHOW", int(cv2.CAP_DSHOW)), ("MSMF", int(cv2.CAP_MSMF)))


def _raw_enumerate(api_preference: int) -> list:
    """One backend's CameraInfo list. Separated for tests to patch."""
    from cv2_enumerate_cameras import enumerate_cameras
    return list(enumerate_cameras(api_preference))


@dataclass
class CameraDevice:
    """One physical camera on one backend."""

    index: int
    backend: int
    backend_name: str
    name: str
    path: str = ""
    vid: Optional[int] = None
    pid: Optional[int] = None

    @property
    def identity(self) -> str:
        """Stable, backend-independent key for this physical device.

        The OS device path without its trailing interface GUID (the GUID
        differs between DSHOW and MSMF; the prefix is shared), else a
        VID/PID+name key, else the device name alone. Numeric index is
        never part of the identity.
        """
        base = self.path.split("{", 1)[0].strip().lower()
        if base:
            return "path:" + base
        if self.vid is not None and self.pid is not None:
            return "usb:%04x:%04x:%s" % (self.vid, self.pid,
                                         self.name.strip().lower())
        return "name:" + self.name.strip().lower()

    @property
    def display_name(self) -> str:
        return f"{self.name} [{self.backend_name}]"


def enumerate_camera_devices() -> List[CameraDevice]:
    """List cameras across DSHOW and MSMF, deduplicated per physical device.

    Never opens a camera. Raises RuntimeError with an actionable message
    when no backend could be queried at all (e.g. cv2-enumerate-cameras
    missing); a single failing backend is skipped so the other still works.
    """
    devices: List[CameraDevice] = []
    seen = set()
    errors = []
    queried = 0
    for backend_name, backend_id in _backend_ids():
        try:
            infos = _raw_enumerate(backend_id)
        except Exception as exc:
            errors.append(f"{backend_name}: {exc}")
            continue
        queried += 1
        for info in infos:
            device = CameraDevice(
                index=int(getattr(info, "index")),
                backend=backend_id,
                backend_name=backend_name,
                name=str(getattr(info, "name", "") or "") or
                f"Camera {getattr(info, 'index')}",
                path=str(getattr(info, "path", "") or ""),
                vid=getattr(info, "vid", None),
                pid=getattr(info, "pid", None),
            )
            if device.identity in seen:
                continue  # same physical device via another backend
            seen.add(device.identity)
            devices.append(device)
    if queried == 0:
        raise RuntimeError(
            "no camera backend could be enumerated ("
            + "; ".join(errors)
            + "). Re-run run.bat to install cv2-enumerate-cameras.")
    return devices


def resolve_device(devices: List[CameraDevice],
                   identity: Optional[str] = None,
                   name: Optional[str] = None) -> Optional[CameraDevice]:
    """Find a device by stable identity, then by exact (case-insensitive)
    name. Never matches on index/backend alone; returns None when the
    device is absent — callers must not fall back positionally."""
    if identity:
        for device in devices:
            if device.identity == identity:
                return device
    if name:
        wanted = name.strip().lower()
        for device in devices:
            if device.name.strip().lower() == wanted:
                return device
    return None


def prefer_device(devices: List[CameraDevice]) -> Optional[CameraDevice]:
    """Default choice: a Xiaomi-family device matched by explicit name.
    Returns None when no phone is present — never the first/laptop camera."""
    lowered = [(device, device.name.lower()) for device in devices]
    for keyword in PREFERRED_NAME_KEYWORDS:
        for device, lname in lowered:
            if keyword in lname:
                return device
    return None


def default_camera_config_path() -> Path:
    return Path.cwd() / CAMERA_CONFIG_FILE_NAME


def load_saved_camera(path=None) -> Optional[dict]:
    """Read the persisted selection: {identity, name, backend, backend_name}
    or None when absent/corrupt."""
    p = Path(path) if path is not None else default_camera_config_path()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict) or not data.get("identity"):
        return None
    return {
        "identity": str(data.get("identity")),
        "name": str(data.get("name", "")),
        "backend": data.get("backend"),
        "backend_name": str(data.get("backend_name", "")),
    }


def save_camera(device: CameraDevice, path=None) -> None:
    """Persist the chosen camera's stable identity + name + backend."""
    p = Path(path) if path is not None else default_camera_config_path()
    payload = {
        "version": 1,
        "identity": device.identity,
        "name": device.name,
        "backend": device.backend,
        "backend_name": device.backend_name,
    }
    tmp = p.with_name(p.name + ".part")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, p)
