"""
Animal detector: MegaDetector v6 weights run directly on Ultralytics YOLO, pinned to CUDA.

Why Ultralytics directly instead of the PytorchWildlife wrapper:
  MegaDetector v6 *is* an Ultralytics YOLO model -- PytorchWildlife runs these exact weights
  through Ultralytics internally. But importing PytorchWildlife eagerly pulls a heavy, fragile
  chain (bioacoustics/soundfile, classification/timm, the legacy yolov5, and a gradio web UI)
  that clashes with this project's "lean deps, no web servers" rule and is shaky on Python
  3.14. So we load the identical official MDV6 weights (Microsoft's Zenodo release) directly.
  Same model, same GPU inference, a fraction of the dependencies.

The first run downloads the chosen weight into weights/ (stdlib urllib); later runs reuse it.
"""
from __future__ import annotations

import hashlib
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import config


class CudaUnavailableError(RuntimeError):
    """Raised when the rig cannot actually run on the GPU. The message explains the fix."""


@dataclass
class Detection:
    """One detected object in a frame."""
    class_id: int                               # MegaDetector class id (0/1/2).
    class_name: str                             # 'animal' | 'person' | 'vehicle'.
    confidence: float                           # 0..1.
    bbox: tuple[float, float, float, float]     # (x1, y1, x2, y2), absolute pixels.


# Official MegaDetector v6 releases (Microsoft AI for Good Lab, Zenodo record 15398270).
# version -> (download URL, local filename, SHA-256). A YOLO .pt is a pickle that EXECUTES code when
# Ultralytics loads it (torch.load is called with weights_only=False), so a tampered weight = remote
# code execution. We verify every downloaded AND cached file against the pinned SHA-256 and refuse to
# load on a mismatch. sha256=None means "not pinned yet": it downloads as before but prints a note
# that integrity isn't verified. To pin one, download it once and run:
#   python -c "import hashlib;print(hashlib.sha256(open('weights/<file>','rb').read()).hexdigest())"
MDV6_WEIGHTS: dict[str, tuple[str, str, str | None]] = {
    "MDV6-yolov9-c":  ("https://zenodo.org/records/15398270/files/MDV6-yolov9-c.pt?download=1",       "MDV6-yolov9-c.pt", None),
    "MDV6-yolov9-e":  ("https://zenodo.org/records/15398270/files/MDV6-yolov9-e-1280.pt?download=1",  "MDV6-yolov9-e-1280.pt", None),
    "MDV6-yolov10-c": ("https://zenodo.org/records/15398270/files/MDV6-yolov10-c.pt?download=1",      "MDV6-yolov10-c.pt", "21ee78a2d4887128e2a4920937d3295b493f44d788d13e1a635378c16dd74ef7"),
    "MDV6-yolov10-e": ("https://zenodo.org/records/15398270/files/MDV6-yolov10-e-1280.pt?download=1", "MDV6-yolov10-e-1280.pt", None),
    "MDV6-rtdetr-c":  ("https://zenodo.org/records/15398270/files/MDV6-rtdetr-c.pt?download=1",       "MDV6-rtdetr-c.pt", None),
}

# MegaDetector's coarse classes. The weights carry model.names too; we prefer those at
# runtime and fall back to this mapping only if they're missing.
MD_CLASS_NAMES = {0: "animal", 1: "person", 2: "vehicle"}


def verify_cuda() -> str:
    """
    Confirm PyTorch can actually *compute* on the GPU, not merely see it.
    Returns the GPU name on success; raises CudaUnavailableError with a fix otherwise.
    """
    try:
        import torch
    except Exception as e:  # torch missing / broken install
        raise CudaUnavailableError(f"PyTorch is not importable: {e}") from e

    if not torch.cuda.is_available():
        raise CudaUnavailableError(
            "No usable NVIDIA GPU for PyTorch. Either install a CUDA build of torch:\n"
            "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130\n"
            "or run without a GPU by passing --device cpu (or --device auto to use the GPU only "
            "when it's available)."
        )

    # is_available() can be True while the wheel has no kernels for this GPU's compute
    # capability (exactly what bit us on the RTX 5050 / Blackwell sm_120). Prove it runs.
    try:
        a = torch.randn(64, 64, device="cuda")
        _ = (a @ a).sum().item()
        torch.cuda.synchronize()
    except Exception as e:
        cap = name = None
        try:
            cap = torch.cuda.get_device_capability(0)
            name = torch.cuda.get_device_name(0)
        except Exception:
            pass
        raise CudaUnavailableError(
            "Your installed torch can't run on this GPU -- it was built for a different GPU "
            "architecture. Install a matching CUDA build (for an NVIDIA RTX 50-series / Blackwell "
            "card use the CUDA 12.8+ cu130 wheels):\n"
            "  pip install torch==2.12.0 torchvision==0.27.0 "
            "--index-url https://download.pytorch.org/whl/cu130\n"
            "or pass --device cpu to run without the GPU.\n"
            f"(GPU seen: {name}, compute capability {cap}. Underlying error: {e})"
        ) from e

    return torch.cuda.get_device_name(0)


def resolve_device(device: str) -> tuple[str, str]:
    """Resolve a requested device into a concrete ``(device, label)`` pair, applying the rig's
    policy. ``device`` is one of:

      'cuda' (default) -- REQUIRE a working NVIDIA GPU; raise CudaUnavailableError with an
                          actionable fix if torch can't actually compute on it. Preserves the
                          original fail-loud behaviour, so a wrong-arch torch build (the
                          Blackwell sm_120 trap) never limps along silently on the main rig.
      'cpu'            -- force CPU inference. No GPU needed; slower per frame, but the motion
                          gate only wakes the detector on real motion (rate-limited), so a
                          backyard rig stays usable on a laptop CPU.
      'auto'           -- use the GPU when it genuinely computes, else fall back to CPU with a
                          one-line note (handy on a box where CUDA may or may not be set up).

    Returns (device, label): device is 'cuda' or 'cpu'; label is a human-readable name for the
    startup banner (the GPU model, or 'CPU').
    """
    device = (device or "cuda").strip().lower()
    if device == "cpu":
        return "cpu", "CPU"
    if device == "cuda":
        return "cuda", verify_cuda()          # raises with an actionable fix if it can't compute
    if device == "auto":
        try:
            return "cuda", verify_cuda()
        except CudaUnavailableError as e:
            print(f"  [device] no usable GPU ({str(e).splitlines()[0]}) -- running on CPU.")
            return "cpu", "CPU"
    raise ValueError(f"Unknown device '{device}'. Use 'cuda', 'cpu', or 'auto'.")


def build_with_fallback(make, device: str, *, what: str = "model") -> tuple:
    """Build a heavy ML model on the resolved device, degrading to CPU when the GPU can't run it.

    Routes `device` through resolve_device() -- the SAME real GPU compute-probe the live detector
    uses -- so a wrong-arch torch (the Blackwell sm_120 trap) is caught here, not at the first
    inference. Then 'cpu' / auto-without-a-GPU build on CPU; 'cuda' / auto-with-a-GPU build on the
    GPU, and if the model itself won't fit on the card it falls back to CPU with a note rather than
    crash. `make(dev)` builds and returns the model (any object) on the concrete 'cuda'/'cpu'
    string. Returns (model, concrete_device).

    Shared by classify.py / embed.py / clipfilter.py so every ML tool selects its device the one
    consistent way (it replaced three near-identical hand-rolled GPU->CPU fallback blocks)."""
    dev, _ = resolve_device(device)        # 'cuda' raises with an actionable fix if it can't compute
    try:
        return make(dev), dev
    except Exception as e:                  # the probe passed, but the full model wouldn't load
        if dev != "cuda":
            raise
        print(f"  [device] {what} would not load on the GPU ({str(e).splitlines()[0]}) -- using CPU.")
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        return make("cpu"), "cpu"


def _sha256(path: Path) -> str:
    """Streaming SHA-256 of a file (chunked so a ~100 MB weight isn't read whole into RAM)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_weights(version: str, weights_dir: Path) -> Path:
    """Return the local path to the weight file, downloading it once if missing and verifying its
    SHA-256 against the pinned digest (the file is loaded as a code-executing pickle, so integrity
    matters). A cached file that fails the check is discarded and re-downloaded; a download that
    fails the check is refused rather than loaded. Weights with no pinned hash (None) behave as
    before, with a one-line 'integrity not verified' note."""
    if version not in MDV6_WEIGHTS:
        raise ValueError(
            f"Unknown model_version '{version}'. Valid: {', '.join(MDV6_WEIGHTS)}"
        )
    url, fname, sha256 = MDV6_WEIGHTS[version]
    weights_dir.mkdir(parents=True, exist_ok=True)
    path = weights_dir / fname
    if path.exists() and path.stat().st_size > 0:
        if sha256 is None or _sha256(path) == sha256:
            return path
        print(f"  [weights] cached {path.name} failed its SHA-256 check -- re-downloading.")
        path.unlink(missing_ok=True)

    print(f"  downloading MegaDetector v6 weights '{version}' -> {path.name} (one time) ...")
    if sha256 is None:
        print(f"  [weights] note: no pinned checksum for '{version}' -- integrity is NOT verified.")
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "backyard-cam/1.0"})
        with urllib.request.urlopen(req) as resp, open(tmp, "wb") as f:
            expected = resp.getheader("Content-Length")
            shutil.copyfileobj(resp, f)
        expected = int(expected) if expected and expected.isdigit() else None
        got = tmp.stat().st_size
        # A dropped connection can leave a non-empty but TRUNCATED .part. Without this check it
        # gets cached as a valid weight (path.exists() and size > 0) and loads as a corrupt model
        # on every later run -- a confusing failure miles from its cause.
        if got == 0 or (expected is not None and got < expected):
            raise RuntimeError(
                f"incomplete download ({got}" + (f" of {expected}" if expected else "") + " bytes)")
        # Integrity gate: a YOLO .pt is unpickled with weights_only=False, so a substituted file is
        # code execution. Verify before we publish it to the cache, and refuse on mismatch.
        if sha256 is not None:
            actual = _sha256(tmp)
            if actual != sha256:
                raise RuntimeError(
                    f"SHA-256 mismatch (expected {sha256}, got {actual}) -- refusing to load a "
                    "weight that may have been tampered with")
        tmp.replace(path)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download weights from {url}: {e}") from e
    return path


class Detector:
    """Loads MegaDetector v6 (Ultralytics YOLO) on the GPU and runs it on single frames."""

    def __init__(
        self,
        model_version: str,
        device: str = "cuda",
        min_confidence: float = 0.25,
        weights_dir: Path | None = None,
        classes: tuple[str, ...] | None = None,
    ):
        # Resolve 'cuda' / 'cpu' / 'auto' to a concrete device. 'cuda' still fails loud if the
        # GPU can't compute; 'cpu' and 'auto' make the rig runnable without an NVIDIA GPU.
        self.device, self.device_name = resolve_device(device)
        self.min_confidence = float(min_confidence)

        weights_path = _ensure_weights(model_version, weights_dir or (config.ROOT / "weights"))

        # Imported here (not at module top) so a missing/broken torch surfaces via the clear
        # resolve_device()/verify_cuda() message above rather than an opaque import error.
        from ultralytics import YOLO

        self.model = YOLO(str(weights_path))
        self.model.to(self.device)  # park the weights on the chosen device (first detect() warm).

        # Prefer the class names baked into the weights; fall back to the MD defaults.
        names = getattr(self.model, "names", None)
        if names:
            self.class_names = {int(k): str(v) for k, v in dict(names).items()}
        else:
            self.class_names = dict(MD_CLASS_NAMES)

        # Optionally restrict which classes the detector REPORTS at all (cfg.detect_classes).
        # We resolve the requested names against the model's own class map and hand the ids to
        # Ultralytics, so the rest are never returned (never drawn, considered, or saved).
        # Unknown names are warned about and skipped; if that would leave nothing, fall back to
        # reporting every class rather than going silently blind. None/empty = report all.
        self.class_ids: list[int] | None = None
        if classes:
            name_to_id = {v: k for k, v in self.class_names.items()}
            ids = [name_to_id[n] for n in classes if n in name_to_id]
            unknown = [n for n in classes if n not in name_to_id]
            if unknown:
                print(f"  [detector] ignoring unknown detect_classes name(s): "
                      f"{', '.join(unknown)} (known: {', '.join(self.class_names.values())})")
            self.class_ids = ids or None

    def detect(self, frame_bgr) -> list[Detection]:
        """
        Run the detector on a single OpenCV BGR frame and return detections at or above
        min_confidence. Ultralytics expects BGR HWC numpy (OpenCV's native format), so the
        frame is passed through directly. Boxes come back in absolute pixel coordinates of
        the frame, matching the BGR frame used for drawing and cropping.
        """
        results = self.model.predict(
            frame_bgr,
            conf=self.min_confidence,
            classes=self.class_ids,   # None = all classes; a list restricts to those class ids
            device=self.device,
            verbose=False,
        )
        if not results:
            return []
        boxes = getattr(results[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy()

        out: list[Detection] = []
        for i in range(len(xyxy)):
            cid = int(clss[i])
            x1, y1, x2, y2 = (float(v) for v in xyxy[i][:4])
            out.append(
                Detection(
                    class_id=cid,
                    class_name=self.class_names.get(cid, str(cid)),
                    confidence=float(confs[i]),
                    bbox=(x1, y1, x2, y2),
                )
            )
        return out
