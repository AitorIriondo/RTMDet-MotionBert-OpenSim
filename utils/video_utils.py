"""
Video processing utilities for frame extraction.
"""

import math
import os
import struct
import subprocess
from pathlib import Path
from typing import Tuple, Optional, Generator
import cv2
import numpy as np
from tqdm import tqdm

# ExifTool path — bundled standalone executable (no Perl required)
_EXIFTOOL_PATH = Path(__file__).parent.parent / "tools" / "exiftool" / "exiftool-13.50_64" / "exiftool.exe"

# Known smartphone camera specs: model_pattern -> 35mm equivalent focal length (mm)
# Used when ExifTool identifies the device but no focal length is in metadata.
# Sources: GSMArena, manufacturer specs
_KNOWN_CAMERA_FOCAL_35MM = {
    # Samsung Galaxy S series (main wide camera)
    "SM-S901": 23,   # Galaxy S22
    "SM-S906": 23,   # Galaxy S22+
    "SM-S908": 23,   # Galaxy S22 Ultra (main)
    "SM-S911": 23,   # Galaxy S23
    "SM-S916": 23,   # Galaxy S23+
    "SM-S918": 23,   # Galaxy S23 Ultra (main)
    "SM-S921": 23,   # Galaxy S24
    "SM-S926": 23,   # Galaxy S24+
    "SM-S928": 23,   # Galaxy S24 Ultra (main)
    "SM-G99": 26,    # Galaxy S21 series
    "SM-G98": 26,    # Galaxy S20 series
    # Samsung Galaxy A series
    "SM-A5": 26,     # Galaxy A5x series
    "SM-A3": 26,     # Galaxy A3x series
    # Apple iPhones (main wide camera)
    "iPhone 15 Pro": 24,
    "iPhone 15": 26,
    "iPhone 14 Pro": 24,
    "iPhone 14": 26,
    "iPhone 13 Pro": 26,
    "iPhone 13": 26,
    "iPhone 12 Pro": 26,
    "iPhone 12": 26,
    "iPhone 11 Pro": 26,
    "iPhone 11": 26,
    "iPhone X": 28,
    "iPhone 8": 28,
    # Google Pixel
    "Pixel 9": 26.3,
    "Pixel 8": 25,
    "Pixel 7": 25,
    "Pixel 6": 25,
}


def extract_focal_length(video_path: str) -> Optional[float]:
    """Extract camera focal length in pixels from video metadata (best-effort).

    Strategy (in priority order):
    1. ExifTool: reads FocalLength/FocalLengthIn35mmFormat directly from metadata
    2. ExifTool: identifies device model, looks up known camera specs
    3. MP4 atom parsing: searches for Apple/Android metadata atoms

    Args:
        video_path: Path to MP4 video file

    Returns:
        Focal length in pixels, or None if not found
    """
    video_path = Path(video_path)
    if not video_path.exists():
        return None

    # Get video dimensions for mm-to-pixel conversion
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Strategy 1 & 2: Use ExifTool if available
    focal_px = _extract_with_exiftool(video_path, width, height)
    if focal_px is not None:
        return focal_px

    # Strategy 3: Manual MP4 atom parsing (fallback)
    try:
        focal_mm, focal_35mm = _parse_mp4_focal(video_path)
    except Exception:
        return None

    if focal_35mm is not None and focal_35mm > 0:
        return _focal_35mm_to_pixels(focal_35mm, width, height)

    if focal_mm is not None and focal_mm > 0:
        return focal_mm * width / 6.0  # assume ~6mm sensor width

    return None


def _focal_35mm_to_pixels(focal_35mm: float, width: int, height: int) -> float:
    """Convert 35mm-equivalent focal length to pixel focal length.

    Uses the horizontal FOV derived from the 35mm equivalent:
      FOV_h = 2 * atan(36 / (2 * f_35mm))
      f_px = width / (2 * tan(FOV_h / 2))
    """
    fov_h = 2 * math.atan(36.0 / (2 * focal_35mm))
    return width / (2 * math.tan(fov_h / 2))


def _extract_with_exiftool(video_path: Path, width: int, height: int) -> Optional[float]:
    """Use ExifTool to extract focal length or identify camera model."""
    exiftool = _EXIFTOOL_PATH
    if not exiftool.exists():
        # Try system-installed exiftool
        exiftool = "exiftool"

    try:
        result = subprocess.run(
            [str(exiftool), "-s",
             "-FocalLength", "-FocalLengthIn35mmFormat",
             "-Model", "-SamsungModel", "-Make",
             str(video_path)],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    tags = {}
    for line in result.stdout.strip().split("\n"):
        if ":" in line:
            key, _, value = line.partition(":")
            tags[key.strip()] = value.strip()

    # Priority 1: Direct focal length from metadata
    focal_35mm = tags.get("FocalLengthIn35mmFormat", "")
    if focal_35mm:
        try:
            val = float(focal_35mm.replace("mm", "").strip())
            if 5 < val < 500:
                return _focal_35mm_to_pixels(val, width, height)
        except ValueError:
            pass

    focal_mm = tags.get("FocalLength", "")
    if focal_mm:
        try:
            val = float(focal_mm.replace("mm", "").strip())
            if 1 < val < 500:
                return val * width / 6.0  # assume smartphone sensor ~6mm
        except ValueError:
            pass

    # Priority 2: Identify device model, look up known specs
    model = tags.get("SamsungModel") or tags.get("Model") or ""
    if model:
        for pattern, f35 in _KNOWN_CAMERA_FOCAL_35MM.items():
            if pattern in model:
                return _focal_35mm_to_pixels(f35, width, height)

    return None


def _parse_mp4_focal(video_path: Path):
    """Parse MP4 atoms for focal length metadata.

    Returns (focal_mm, focal_35mm_equiv) — either may be None.
    """
    focal_mm = None
    focal_35mm = None

    with open(video_path, "rb") as f:
        data = f.read(min(video_path.stat().st_size, 4_000_000))

    for key, is_35mm in [
        (b"com.apple.quicktime.camera.focal_length.35mm_equivalent", True),
        (b"com.apple.quicktime.camera.focal_length", False),
        (b"FocalLengthIn35mmFormat", True),
        (b"FocalLength", False),
    ]:
        idx = data.find(key)
        if idx < 0:
            continue

        search_start = idx + len(key)
        search_end = min(search_start + 64, len(data))
        chunk = data[search_start:search_end]

        val = _find_numeric_value(chunk)
        if val is not None and 1.0 < val < 500.0:
            if is_35mm:
                focal_35mm = val
            else:
                focal_mm = val

    return focal_mm, focal_35mm


def _find_numeric_value(chunk: bytes) -> Optional[float]:
    """Try to extract a numeric value from a chunk of bytes."""
    for offset in range(min(len(chunk) - 4, 32)):
        try:
            val = struct.unpack(">f", chunk[offset:offset + 4])[0]
            if 1.0 < val < 500.0:
                return val
        except Exception:
            pass
    for offset in range(min(len(chunk) - 2, 32)):
        try:
            val = struct.unpack(">H", chunk[offset:offset + 2])[0]
            if 10 < val < 500:
                return float(val)
        except Exception:
            pass
    return None


def get_video_info(video_path: str) -> dict:
    """
    Get video metadata.

    Args:
        video_path: Path to video file

    Returns:
        Dictionary with video info:
            - fps: Frames per second
            - frame_count: Total number of frames
            - width: Frame width in pixels
            - height: Frame height in pixels
            - duration: Duration in seconds
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    info["duration"] = info["frame_count"] / info["fps"] if info["fps"] > 0 else 0

    cap.release()
    return info


def extract_frames(
    video_path: str,
    output_dir: str,
    target_fps: Optional[float] = None,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    image_format: str = "jpg",
    quality: int = 95,
) -> Tuple[list, float]:
    """
    Extract frames from video to image files.

    Args:
        video_path: Path to input video
        output_dir: Directory to save extracted frames
        target_fps: Target FPS (None = use original FPS)
        start_frame: First frame to extract (0-indexed)
        end_frame: Last frame to extract (None = all frames)
        image_format: Output image format ('jpg' or 'png')
        quality: JPEG quality (1-100)

    Returns:
        Tuple of (list of frame paths, actual fps)
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if end_frame is None:
        end_frame = total_frames

    # Calculate frame interval for target FPS
    if target_fps is None or target_fps >= original_fps:
        frame_interval = 1
        actual_fps = original_fps
    else:
        frame_interval = round(original_fps / target_fps)
        actual_fps = original_fps / frame_interval

    # Set encoding parameters
    if image_format.lower() == "jpg":
        ext = ".jpg"
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    else:
        ext = ".png"
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]

    frame_paths = []
    frame_idx = 0
    saved_idx = 0

    # Set starting position
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_idx = start_frame

    pbar = tqdm(
        total=(end_frame - start_frame) // frame_interval,
        desc="Extracting frames",
        unit="frames",
    )

    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if (frame_idx - start_frame) % frame_interval == 0:
            # Save frame
            frame_filename = f"frame_{saved_idx:06d}{ext}"
            frame_path = output_dir / frame_filename
            cv2.imwrite(str(frame_path), frame, params)
            frame_paths.append(str(frame_path))
            saved_idx += 1
            pbar.update(1)

        frame_idx += 1

    pbar.close()
    cap.release()

    return frame_paths, actual_fps


def frame_generator(
    video_path: str,
    target_fps: Optional[float] = None,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Generator that yields frames from video without saving to disk.

    Args:
        video_path: Path to input video
        target_fps: Target FPS (None = use original FPS)
        start_frame: First frame to process
        end_frame: Last frame to process

    Yields:
        Tuple of (frame_index, frame_array in RGB format)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if end_frame is None:
        end_frame = total_frames

    # Calculate frame interval
    if target_fps is None or target_fps >= original_fps:
        frame_interval = 1
    else:
        frame_interval = round(original_fps / target_fps)

    # Set starting position
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_idx = start_frame
    output_idx = 0

    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if (frame_idx - start_frame) % frame_interval == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield output_idx, frame_rgb
            output_idx += 1

        frame_idx += 1

    cap.release()
