#!/usr/bin/env python3
"""
RTMPose3D Inference
====================

Runs RTMPose3D (RTMW3D) inference and saves results to video_outputs.json.
This is the slow step - run once, then iterate on export settings.

Usage:
    python run_inference.py --input video.mp4
    python run_inference.py --input video.mp4 --output my_output_dir --fps 30
    python run_inference.py --input video.mp4 --device cpu

Output:
    output_dir/
    ├── video_outputs.json    # RTMPose3D outputs (keypoints_3d, scores, bboxes)
    └── inference_meta.json   # Metadata (fps, timing, video info)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.video_utils import get_video_info
from utils.io_utils import load_config, get_output_dir


def parse_args():
    parser = argparse.ArgumentParser(description="RTMPose3D Inference")
    parser.add_argument("--input", "-i", required=True, help="Input video file")
    parser.add_argument("--output", "-o", help="Output directory (default: auto)")
    parser.add_argument("--fps", type=float, default=30.0, help="Target FPS")
    parser.add_argument("--device", default="cuda:0", help="Device (cuda:0 or cpu)")
    parser.add_argument("--config", help="Config file path")
    parser.add_argument("--person", type=int, default=0, help="Person index if multiple detected")
    parser.add_argument("--model", default=None, help="Model name (e.g. rtmpose3d-x for XL model)")
    return parser.parse_args()


def run_inference(
    input_path: str,
    output_dir: str,
    fps: float,
    device: str,
    config_path: str = None,
    person_idx: int = 0,
    model_override: str = None,
):
    """Run RTMPose3D inference and save to JSON."""
    start_time = time.time()

    # Load config
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        config = {"rtmpose3d": {"model_name": "rbarac/rtmpose3d", "device": device}}

    # Setup output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = model_override or config.get("rtmpose3d", {}).get("model_name", "rbarac/rtmpose3d")
    if device == "cuda:0":
        device = config.get("rtmpose3d", {}).get("device", device)

    print(f"\n{'='*60}")
    print("RTMPose3D Inference")
    print(f"{'='*60}")
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"{'='*60}\n")

    # Step 1: Open video and compute frame sampling
    print("[1/2] Reading video...")
    video_info = get_video_info(input_path)
    print(f"  Video: {video_info['width']}x{video_info['height']}, {video_info['fps']:.2f} FPS, {video_info['duration']:.1f}s")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"Error: Cannot open video: {input_path}")
        sys.exit(1)

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate frame interval for target FPS
    if fps is None or fps >= original_fps:
        frame_interval = 1
        actual_fps = original_fps
    else:
        frame_interval = round(original_fps / fps)
        actual_fps = original_fps / frame_interval

    expected_frames = total_frames // frame_interval
    print(f"  Sampling every {frame_interval} frames → ~{expected_frames} frames at {actual_fps:.2f} FPS")

    # Step 2: Run RTMPose3D (read frames directly from video, no disk I/O)
    print("\n[2/2] Running RTMPose3D inference...")

    from src.rtmpose3d_inference import RTMPose3DInference

    model = RTMPose3DInference(
        model_name=model_name,
        device=device,
    )

    all_outputs = []
    frame_idx = 0
    processed_idx = 0

    pbar = tqdm(total=expected_frames, desc="Processing", unit="frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            # frame is already BGR from cv2.VideoCapture, which RTMPose3D expects
            persons = model.process_frame(frame)

            frame_data = {
                "frame": f"frame_{processed_idx:06d}",
                "outputs": [],
            }

            for person in persons:
                person_data = {
                    "keypoints_3d": person["keypoints_3d"].tolist()
                        if isinstance(person["keypoints_3d"], np.ndarray)
                        else person["keypoints_3d"],
                    "keypoints_2d": person["keypoints_2d"].tolist()
                        if isinstance(person["keypoints_2d"], np.ndarray)
                        else person["keypoints_2d"],
                    "scores": person["scores"].tolist()
                        if isinstance(person["scores"], np.ndarray)
                        else person["scores"],
                    "bbox": person["bbox"].tolist()
                        if isinstance(person["bbox"], np.ndarray)
                        else person["bbox"],
                }
                frame_data["outputs"].append(person_data)

            all_outputs.append(frame_data)
            processed_idx += 1
            pbar.update(1)

        frame_idx += 1

    pbar.close()
    cap.release()

    num_frames = processed_idx
    print(f"  Processed {num_frames} frames")

    # Save to JSON
    json_path = output_dir / "video_outputs.json"
    with open(json_path, 'w') as f:
        json.dump(all_outputs, f, indent=2)

    # Try to extract focal length from video metadata
    from utils.video_utils import extract_focal_length
    focal_length_px = extract_focal_length(str(input_path))
    if focal_length_px is not None:
        print(f"  Focal length from metadata: {focal_length_px:.0f}px")

    # Save metadata
    meta_path = output_dir / "inference_meta.json"
    elapsed = time.time() - start_time
    meta = {
        "input_video": str(input_path),
        "fps": actual_fps,
        "num_frames": num_frames,
        "video_info": video_info,
        "model_name": model_name,
        "device": device,
        "inference_time": elapsed,
    }
    if focal_length_px is not None:
        meta["focal_length_px"] = focal_length_px
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*60}")
    print("Inference Complete!")
    print(f"{'='*60}")
    print(f"Time: {elapsed:.1f}s ({num_frames/elapsed:.1f} FPS)")
    print(f"Frames: {num_frames}")
    print(f"Output: {json_path}")
    print(f"{'='*60}\n")

    return str(json_path)


def main():
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input not found: {input_path}")
        sys.exit(1)

    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = get_output_dir(str(input_path))

    run_inference(
        input_path=str(input_path),
        output_dir=str(output_dir),
        fps=args.fps,
        device=args.device,
        config_path=args.config,
        person_idx=args.person,
        model_override=args.model,
    )


if __name__ == "__main__":
    main()
