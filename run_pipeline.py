#!/usr/bin/env python3
"""
Full Pipeline: Video -> OpenSim Motion Data
============================================

Runs both stages of the pipeline in a single command:
  1. RTMPose3D inference (slow, ~3 min)
  2. Hybrid export: MotionBERT 3D + OpenSim IK + GLB (fast, ~40 sec)

Usage:
    python run_pipeline.py --input videos/my_video.mp4 --height 1.69
    python run_pipeline.py --input videos/my_video.mp4 --height 1.69 --correct-lean
    python run_pipeline.py --input videos/my_video.mp4 --height 1.69 --correct-lean --single-level

For iterating on export settings without re-running inference, use the two-stage workflow:
    python run_inference.py --input videos/my_video.mp4
    python run_hybrid_pipeline.py --input videos/my_video.mp4 --height 1.69
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.io_utils import get_output_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Full Pipeline: Video -> OpenSim Motion Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --input videos/my_video.mp4 --height 1.69
  python run_pipeline.py --input videos/my_video.mp4 --height 1.69 --correct-lean --single-level
  python run_pipeline.py --input videos/my_video.mp4 --height 1.69 --device cpu
        """,
    )

    # Shared
    parser.add_argument("--input", "-i", required=True, help="Input video file (.mp4, .avi, .mov, .mkv)")
    parser.add_argument("--output", "-o", help="Output directory (default: auto-timestamped)")
    parser.add_argument("--device", default="cuda:0", help="Compute device (cuda:0 or cpu)")
    parser.add_argument("--person", type=int, default=0, help="Person index if multiple detected")

    # Inference stage
    inference = parser.add_argument_group("Inference (Stage 1)")
    inference.add_argument("--fps", type=float, default=30.0, help="Target FPS for frame extraction")
    inference.add_argument("--model", default=None, help="Model name override (e.g. rtmpose3d-x for XL)")

    # Export stage
    export = parser.add_argument_group("Export (Stage 2)")
    export.add_argument("--height", type=float, default=1.75, help="Subject height in meters")
    export.add_argument("--mass", type=float, default=70.0, help="Subject mass in kg")
    export.add_argument("--smooth", type=float, default=6.0, help="Smoothing cutoff Hz (0 to disable)")
    export.add_argument("--skip-ik", action="store_true", help="Skip OpenSim inverse kinematics")
    export.add_argument("--skip-glb", action="store_true", help="Skip GLB export")
    export.add_argument("--pose-model", default="COCO_17", choices=["COCO_17", "COCO_133"],
                        help="Pose model for IK: COCO_17 (22 markers) or COCO_133 (27 markers)")
    export.add_argument("--focal-length", type=float, default=None,
                        help="Camera focal length in pixels (default: auto-detect)")
    export.add_argument("--correct-lean", action="store_true",
                        help="Correct forward lean using ground-plane estimation from foot contacts")
    export.add_argument("--single-level", action="store_true",
                        help="Per-frame strict grounding: lowest foot = Y=0 every frame")

    return parser.parse_args()


def main():
    args = parse_args()
    pipeline_start = time.time()

    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input not found: {input_path}")
        sys.exit(1)

    if input_path.suffix.lower() not in (".mp4", ".avi", ".mov", ".mkv"):
        print(f"Error: Input must be a video file, got: {input_path}")
        sys.exit(1)

    # Output directory (shared between both stages)
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = get_output_dir(str(input_path))

    print(f"\n{'='*60}")
    print("Full Pipeline: Video -> OpenSim Motion Data")
    print(f"{'='*60}")
    print(f"Input:  {input_path}")
    print(f"Output: {output_dir}")
    print(f"Device: {args.device}")
    print(f"{'='*60}\n")

    # ── Stage 1: Inference ──────────────────────────────────────
    print(f"{'─'*60}")
    print("STAGE 1: RTMPose3D Inference")
    print(f"{'─'*60}\n")

    from run_inference import run_inference

    json_path = run_inference(
        input_path=str(input_path),
        output_dir=str(output_dir),
        fps=args.fps,
        device=args.device,
        person_idx=args.person,
        model_override=args.model,
    )

    # ── Stage 2: Hybrid Export ──────────────────────────────────
    print(f"\n{'─'*60}")
    print("STAGE 2: Hybrid Export (MotionBERT 3D + OpenSim IK + GLB)")
    print(f"{'─'*60}\n")

    from run_hybrid_pipeline import run_hybrid_pipeline

    run_hybrid_pipeline(
        json_path=json_path,
        output_dir=str(output_dir),
        subject_height=args.height,
        subject_mass=args.mass,
        fps=None,  # Use FPS from inference metadata
        smooth_cutoff=args.smooth,
        skip_ik=args.skip_ik,
        skip_glb=args.skip_glb,
        person_idx=args.person,
        device=args.device,
        pose_model=args.pose_model,
        focal_length=args.focal_length,
        correct_lean=args.correct_lean,
        single_level=args.single_level,
    )

    # ── Summary ─────────────────────────────────────────────────
    total_time = time.time() - pipeline_start
    minutes = int(total_time // 60)
    seconds = total_time % 60

    print(f"\n{'='*60}")
    print("Full Pipeline Complete!")
    print(f"{'='*60}")
    print(f"Total time: {minutes}m {seconds:.1f}s")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
