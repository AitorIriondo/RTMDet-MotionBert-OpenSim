#!/usr/bin/env python3
"""
Hybrid Pipeline: RTMW 2D + MotionBERT 3D -> OpenSim
=====================================================

Uses RTMW 2D keypoints (from run_inference.py) + MotionBERT temporal lifting
for body 3D, with hand/head/feet projected from 2D at body depth anchors.

Usage:
    python run_hybrid_pipeline.py --input test_output_hybrid/video_outputs.json --height 1.69
    python run_hybrid_pipeline.py --input video_outputs.json --height 1.69 --smooth 6.0
    python run_hybrid_pipeline.py --input video_outputs.json --height 1.69 --skip-ik
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Hybrid RTMW 2D + MotionBERT 3D Pipeline")
    parser.add_argument("--input", "-i", required=True,
                        help="Input video (.mp4) or video_outputs.json. "
                             "If a video is given, finds the latest inference output automatically.")
    parser.add_argument("--height", type=float, default=1.75, help="Subject height (m)")
    parser.add_argument("--mass", type=float, default=70.0, help="Subject mass (kg)")
    parser.add_argument("--output", "-o", help="Output directory (default: same as input)")
    parser.add_argument("--fps", type=float, help="Override FPS (default: from metadata)")
    parser.add_argument("--smooth", type=float, default=6.0, help="Smoothing cutoff Hz (0=disable)")
    parser.add_argument("--skip-ik", action="store_true", help="Skip OpenSim scaling + IK")
    parser.add_argument("--skip-glb", action="store_true", help="Skip GLB export")
    parser.add_argument("--person", type=int, default=0, help="Person index")
    parser.add_argument("--device", default="cuda:0", help="Device for MotionBERT (cuda:0 or cpu)")
    parser.add_argument("--pose-model", default="COCO_17", choices=["COCO_17", "COCO_133"],
                        help="Pose model for IK: COCO_17 (14 markers, body only) or COCO_133 (27 markers)")
    parser.add_argument("--focal-length", type=float, default=None,
                        help="Camera focal length in pixels (default: auto-estimate from resolution). "
                             "Corrects forward lean from FOV mismatch with MotionBERT training cameras.")
    parser.add_argument("--correct-lean", action="store_true",
                        help="Correct forward lean using ground-plane estimation from foot contacts. "
                             "Fits a plane to 3D foot positions during stance phases and rotates "
                             "the skeleton to make the ground horizontal.")
    parser.add_argument("--single-level", action="store_true",
                        help="Per-frame strict grounding: lowest foot = Y=0 every frame. "
                             "Removes all vertical translation (walking on perfectly flat ground).")
    return parser.parse_args()


def load_data(json_path: str, person_idx: int = 0):
    """Load 2D and 3D keypoints from video_outputs.json."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    T = len(data)
    kpts_2d = np.zeros((T, 133, 2), dtype=np.float32)
    kpts_3d = np.zeros((T, 133, 3), dtype=np.float32)
    scores = np.zeros((T, 133), dtype=np.float32)
    valid = np.zeros(T, dtype=bool)

    for i, frame in enumerate(data):
        outputs = frame.get("outputs", [])
        if len(outputs) > person_idx:
            p = outputs[person_idx]

            kp2d = p.get("keypoints_2d")
            if kp2d and len(kp2d) == 133:
                kpts_2d[i] = np.array(kp2d)

            kp3d = p.get("keypoints_3d")
            if kp3d and len(kp3d) == 133:
                kpts_3d[i] = np.array(kp3d)

            sc = p.get("scores")
            if sc and len(sc) == 133:
                scores[i] = np.array(sc)

            valid[i] = True

    return kpts_2d, kpts_3d, scores, valid, T


def run_hybrid_pipeline(
    json_path: str,
    output_dir: str,
    subject_height: float,
    subject_mass: float,
    fps: float,
    smooth_cutoff: float,
    skip_ik: bool,
    skip_glb: bool,
    person_idx: int,
    device: str,
    pose_model: str = "COCO_17",
    focal_length: float = None,
    correct_lean: bool = False,
    single_level: bool = False,
):
    """Run the hybrid RTMW 2D + MotionBERT 3D pipeline."""
    start_time = time.time()

    json_path = Path(json_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Hybrid Pipeline: RTMW 2D + MotionBERT 3D ({pose_model})")
    print(f"{'='*60}")
    print(f"Input: {json_path}")
    print(f"Output: {output_dir}")
    print(f"Subject: height={subject_height}m, mass={subject_mass}kg")
    print(f"{'='*60}\n")

    # ---- Step 1: Load data ----
    print("[1/7] Loading RTMW outputs...")
    kpts_2d, kpts_3d_rtm, scores, valid, T = load_data(str(json_path), person_idx)
    print(f"  {T} frames, {np.sum(valid)} valid")

    has_2d = np.any(kpts_2d != 0)
    if not has_2d:
        print("  ERROR: No 2D keypoints found. Re-run inference with updated run_inference.py.")
        sys.exit(1)

    # Load inference metadata
    meta_path = json_path.parent / "inference_meta.json"
    meta = {}
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            meta = json.load(f)

    if fps is None:
        fps = meta.get("fps", 30.0)
    print(f"  FPS: {fps}")

    image_width = meta.get("video_info", {}).get("width", 1920)
    image_height = meta.get("video_info", {}).get("height", 1080)
    print(f"  Image: {image_width}x{image_height}")

    # Auto-estimate focal length if not provided
    if focal_length is None:
        focal_length = meta.get("focal_length_px")
    if focal_length is None:
        # Try to extract from source video via ExifTool / MP4 atoms
        from utils.video_utils import extract_focal_length
        source_video = meta.get("input_video")
        if source_video and Path(source_video).exists():
            focal_length = extract_focal_length(source_video)
    if focal_length is not None:
        print(f"  Focal length: {focal_length:.0f}px")
    else:
        print("  Focal length: not set (no correction)")

    # ---- Step 2: MotionBERT 3D lifting ----
    print("\n[2/7] MotionBERT 3D body lifting...")
    from src.coco_h36m_converter import coco17_to_h36m, h36m_to_coco17
    from src.motionbert_inference import MotionBERTLifter

    # Extract body-17 2D keypoints
    body_2d = kpts_2d[:, :17].copy()
    body_scores = scores[:, :17].copy()

    # Convert COCO-17 → H36M-17
    h36m_2d, h36m_scores = coco17_to_h36m(body_2d, body_scores)
    print(f"  H36M 2D: {h36m_2d.shape}")

    # Run MotionBERT lifting
    lifter = MotionBERTLifter(device=device)
    body_3d_h36m = lifter.lift(
        h36m_2d, h36m_scores,
        image_width=image_width,
        image_height=image_height,
        focal_length=focal_length,
    )
    print(f"  3D output: {body_3d_h36m.shape}")

    # Check depth quality
    H36M_RHIP, H36M_RKNEE, H36M_LHIP, H36M_LKNEE = 1, 2, 4, 5
    hip_center = (body_3d_h36m[:, H36M_RHIP] + body_3d_h36m[:, H36M_LHIP]) / 2
    rknee_depth = (body_3d_h36m[:, H36M_RKNEE, 2] - hip_center[:, 2]) * 1000
    lknee_depth = (body_3d_h36m[:, H36M_LKNEE, 2] - hip_center[:, 2]) * 1000
    print(f"  Knee depth: R={rknee_depth.mean():.0f}mm, L={lknee_depth.mean():.0f}mm")

    # Normalize skeleton scale per frame (MotionBERT scale varies with distance)
    body_3d_h36m = _normalize_skeleton_scale(body_3d_h36m)

    # Normalize bone lengths to median
    body_3d_h36m = _normalize_h36m_bones(body_3d_h36m)

    # Smooth body 3D uniformly (all joints, all axes, same frequency)
    if smooth_cutoff > 0:
        print(f"  Smoothing body 3D: {smooth_cutoff}Hz (uniform)...")
        body_3d_h36m = _smooth_h36m_body(body_3d_h36m, fps, smooth_cutoff)

    # Convert H36M-17 → COCO-17 3D
    body_3d_coco17 = h36m_to_coco17(body_3d_h36m)
    print(f"  COCO-17 3D: {body_3d_coco17.shape}")

    if pose_model == "COCO_17":
        # ---- COCO_17 path: 14 body + 8 projected markers (22 total) ----
        # Body markers from MotionBERT 3D (high quality).
        # Eyes + hand fingertips projected from 2D at body depth anchors.
        print("\n[3/5] Projecting extra markers (eyes + hands)...")
        extra_markers = _project_extra_markers(kpts_2d, body_3d_coco17)
        print(f"  Projected 8 markers: LEye, REye, LThumb, LIndex, LPinky, RThumb, RIndex, RPinky")

        print("  Coordinate transform (22 markers)...")
        markers, marker_names = _extract_coco17_markers(
            body_3d_coco17, subject_height,
            extra_markers_h36m=extra_markers,
            correct_lean=correct_lean,
            fps=fps,
            single_level=single_level,
        )
        print(f"  {len(marker_names)} markers: {', '.join(marker_names)}")

        # TRC export
        print("\n[4/5] Exporting TRC...")
        from src.trc_exporter import TRCExporter

        video_name = json_path.parent.name
        if video_name in (".", ""):
            video_name = json_path.stem.replace("video_outputs", "hybrid")

        trc_exporter = TRCExporter(fps=fps, units="m")
        trc_path = output_dir / f"markers_{video_name}.trc"
        trc_exporter.export(markers, marker_names, str(trc_path))
        print(f"  TRC: {trc_path}")

        results = {"trc": trc_path, "mot": None, "glb": None}

        # Pose2Sim IK with COCO_17
        if not skip_ik:
            print("\n[5/5] Pose2Sim scaling + IK (COCO_17)...")
            from run_export import run_pose2sim_kinematics

            mot_path = run_pose2sim_kinematics(
                trc_path, output_dir, subject_height, subject_mass,
                pose_model="COCO_17",
            )
            if mot_path:
                mot_path_reg = _rerun_ik_light_regularization(
                    output_dir, trc_path
                )
                if mot_path_reg:
                    mot_path = mot_path_reg
                mot_path = _post_smooth_mot(mot_path, fps)
            results["mot"] = mot_path
        else:
            print("\nSkipping IK")

    else:
        # ---- COCO_133 path: 27 markers with projected hands/feet/face ----
        print("\n[3/7] Projecting hands to 3D...")
        from src.hand_projector import project_hands_to_3d

        hands_3d = project_hands_to_3d(kpts_2d, body_3d_coco17)
        print(f"  Hands 3D: {hands_3d.shape}")

        print("\n[4/7] Projecting head to 3D...")
        from src.head_projector import project_head_to_3d

        head_3d = project_head_to_3d(kpts_2d, body_3d_coco17)
        print(f"  Head 3D: {head_3d.shape}")

        print("\n[5/7] Merging into COCO-WholeBody 133...")
        from src.hybrid_merger import merge_hybrid_keypoints

        keypoints_133 = merge_hybrid_keypoints(
            body_3d_coco17, hands_3d, head_3d, kpts_2d,
        )
        print(f"  Merged: {keypoints_133.shape}")

        if smooth_cutoff > 0:
            depth_cutoff = min(smooth_cutoff, 1.0)
            print(f"  Butterworth smoothing: lateral/vertical={smooth_cutoff}Hz, depth={depth_cutoff}Hz")
            keypoints_133 = _smooth_keypoints_axis(
                keypoints_133, fps, smooth_cutoff, depth_cutoff, depth_axis=2
            )

        print("\n[6/7] Coordinate transform + marker extraction...")
        from src.coordinate_transform import CoordinateTransformer
        from src.keypoint_converter import KeypointConverter

        transformer = CoordinateTransformer(subject_height=subject_height, units="m")
        keypoints_opensim = transformer.transform_motionbert(
            keypoints_133,
            center_pelvis=True,
            align_to_ground=True,
            correct_lean=correct_lean,
            fps=fps,
            single_level=single_level,
        )
        print("  Transform: H36M camera -> OpenSim")

        converter = KeypointConverter()
        markers, marker_names = converter.convert(keypoints_opensim)
        print(f"  Markers: {len(marker_names)}")

        print("\n[7/7] Exporting TRC...")
        from src.trc_exporter import TRCExporter

        video_name = json_path.parent.name
        if video_name in (".", ""):
            video_name = json_path.stem.replace("video_outputs", "hybrid")

        trc_exporter = TRCExporter(fps=fps, units="m")
        trc_path = output_dir / f"markers_{video_name}.trc"
        trc_exporter.export(markers, marker_names, str(trc_path))
        print(f"  TRC: {trc_path}")

        results = {"trc": trc_path, "mot": None, "glb": None}

        if not skip_ik:
            print("\nRunning Pose2Sim scaling + IK...")
            from run_export import run_pose2sim_kinematics

            mot_path = run_pose2sim_kinematics(
                trc_path, output_dir, subject_height, subject_mass,
                pose_model="COCO_133",
            )
            if mot_path:
                mot_path_reg = _rerun_ik_light_regularization(
                    output_dir, trc_path
                )
                if mot_path_reg:
                    mot_path = mot_path_reg
                mot_path = _post_smooth_mot(mot_path, fps)
            results["mot"] = mot_path
        else:
            print("\nSkipping IK")

    # GLB export
    if not skip_glb and not skip_ik and results["mot"]:
        print("\nExporting GLB...")
        from run_export import run_glb_export
        glb_path = run_glb_export(results["mot"], output_dir)
        results["glb"] = glb_path

    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print("Hybrid Pipeline Complete!")
    print(f"{'='*60}")
    print(f"Time: {elapsed:.1f}s")
    for name, path in results.items():
        status = "OK" if path and Path(path).exists() else "SKIPPED"
        print(f"  [{status}] {name.upper()}: {path}")
    print(f"{'='*60}\n")

    return results


def _extract_coco17_markers(
    body_3d_coco17: np.ndarray, subject_height: float,
    extra_markers_h36m: np.ndarray = None,
    correct_lean: bool = False,
    fps: float = 30.0,
    single_level: bool = False,
) -> tuple:
    """
    Extract Pose2Sim COCO_17 markers from COCO-17 body 3D.

    When extra_markers_h36m is provided, outputs 22 markers (14 body + 8 extra).
    Otherwise outputs the original 14 body markers.

    Transforms from MotionBERT H36M camera coords to OpenSim coords,
    preserving global translation and facing direction.

    Args:
        body_3d_coco17: (T, 17, 3) COCO-17 3D in H36M camera coords
        subject_height: Subject height in meters
        extra_markers_h36m: (T, 8, 3) projected markers in H36M camera coords
            [LEye, REye, LThumb, LIndex, LPinky, RThumb, RIndex, RPinky]
        correct_lean: Correct forward lean via ground-plane estimation
        fps: Video frame rate (needed for foot velocity computation)

    Returns:
        markers: (T, 14, 3) or (T, 22, 3) markers in OpenSim coords (meters)
        marker_names: list of marker names
    """
    from scipy.ndimage import uniform_filter1d
    from src.coordinate_transform import CoordinateTransformer

    T = body_3d_coco17.shape[0]

    # Apply MB -> OpenSim rotation
    MB_TO_OPENSIM = CoordinateTransformer.MB_TO_OPENSIM
    transformed = body_3d_coco17.copy()
    for i in range(T):
        transformed[i] = transformed[i] @ MB_TO_OPENSIM.T

    # Transform extra markers with the same rotation
    extra_transformed = None
    if extra_markers_h36m is not None:
        extra_transformed = extra_markers_h36m.copy()
        for i in range(T):
            extra_transformed[i] = extra_transformed[i] @ MB_TO_OPENSIM.T

    # Scale to subject height (nose to ankle midpoint)
    heights = []
    for i in range(T):
        nose = transformed[i, 0]
        ankle_mid = (transformed[i, 15] + transformed[i, 16]) / 2
        h = np.linalg.norm(nose - ankle_mid)
        if h > 0.1:
            heights.append(h)
    if heights:
        avg_h = np.mean(heights)
        estimated_full = avg_h * 1.1
        scale = subject_height / estimated_full
        transformed *= scale
        if extra_transformed is not None:
            extra_transformed *= scale
        print(f"  Scale: avg nose-ankle={avg_h*scale:.3f}m, factor={scale:.3f}")

    # Ground-plane lean correction (after scaling, before centering)
    if correct_lean:
        ct = CoordinateTransformer(subject_height=subject_height, units="m")
        # COCO-17 only has ankles as foot keypoints (indices 15, 16)
        foot_indices = {"left": [15], "right": [16]}
        # Detect contacts and compute rotation from body keypoints
        contacts = ct._detect_ground_contacts(transformed, foot_indices, fps, 0.3)
        plane = ct._fit_ground_plane(contacts) if len(contacts) >= 10 else None
        if plane is not None:
            normal, centroid = plane
            R = ct._compute_ground_rotation(normal)
            total_angle = np.degrees(np.arccos(np.clip(
                np.dot(normal, np.array([0, 1, 0])), -1, 1)))
            sagittal = np.degrees(np.arctan2(normal[0], normal[1]))
            frontal = np.degrees(np.arctan2(normal[2], normal[1]))
            if total_angle >= 0.5:
                print(f"  Ground-plane correction: {total_angle:.1f}° "
                      f"(sagittal={sagittal:.1f}°, frontal={frontal:.1f}°, "
                      f"{len(contacts)} contact points)")
                pelvis_all = (transformed[:, 11] + transformed[:, 12]) / 2
                pelvis_center = np.mean(pelvis_all, axis=0)
                transformed -= pelvis_center
                transformed = transformed @ R.T
                transformed += pelvis_center
                if extra_transformed is not None:
                    extra_transformed -= pelvis_center
                    extra_transformed = extra_transformed @ R.T
                    extra_transformed += pelvis_center

    # Center at FIRST FRAME pelvis (preserves global translation)
    first_pelvis = (transformed[0, 11] + transformed[0, 12]) / 2
    transformed[:, :, 0] -= first_pelvis[0]
    transformed[:, :, 2] -= first_pelvis[2]
    if extra_transformed is not None:
        extra_transformed[:, :, 0] -= first_pelvis[0]
        extra_transformed[:, :, 2] -= first_pelvis[2]

    # Align to ground using ankles (Y axis in OpenSim = up)
    # COCO-17 only has ankle markers (not heels/toes), so we apply an offset
    # when single_level is True to account for the ankle-to-sole distance.
    ANKLE_TO_SOLE = 0.08  # meters — constant offset from ankle joint to foot sole
    ankle_min_y = np.zeros(T)
    for i in range(T):
        ankle_min_y[i] = min(transformed[i, 15, 1], transformed[i, 16, 1])
    if single_level:
        # Per-frame strict grounding: foot sole = Y=0 every frame
        ground_ref = ankle_min_y - ANKLE_TO_SOLE
    else:
        # Smoothed ground reference (15-frame window ≈ 0.5s at 30fps)
        if T > 15:
            ground_ref = uniform_filter1d(ankle_min_y, size=15)
        else:
            ground_ref = ankle_min_y
    for i in range(T):
        transformed[i, :, 1] -= ground_ref[i]
        if extra_transformed is not None:
            extra_transformed[i, :, 1] -= ground_ref[i]

    # Log global movement range
    pelvis_traj = np.zeros((T, 3))
    for i in range(T):
        pelvis_traj[i] = (transformed[i, 11] + transformed[i, 12]) / 2
    x_range = pelvis_traj[:, 0].max() - pelvis_traj[:, 0].min()
    z_range = pelvis_traj[:, 2].max() - pelvis_traj[:, 2].min()
    print(f"  Global movement: X={x_range:.2f}m, Z={z_range:.2f}m")

    # Base 14 markers for Pose2Sim COCO_17
    base_names = [
        "Nose", "Neck", "LShoulder", "RShoulder",
        "LElbow", "RElbow", "LWrist", "RWrist",
        "LHip", "RHip", "LKnee", "RKnee",
        "LAnkle", "RAnkle",
    ]
    base_markers = np.zeros((T, 14, 3), dtype=np.float32)
    base_markers[:, 0] = transformed[:, 0]     # Nose
    base_markers[:, 1] = (transformed[:, 5] + transformed[:, 6]) / 2  # Neck = shoulder midpoint
    base_markers[:, 2] = transformed[:, 5]     # LShoulder
    base_markers[:, 3] = transformed[:, 6]     # RShoulder
    base_markers[:, 4] = transformed[:, 7]     # LElbow
    base_markers[:, 5] = transformed[:, 8]     # RElbow
    base_markers[:, 6] = transformed[:, 9]     # LWrist
    base_markers[:, 7] = transformed[:, 10]    # RWrist
    base_markers[:, 8] = transformed[:, 11]    # LHip
    base_markers[:, 9] = transformed[:, 12]    # RHip
    base_markers[:, 10] = transformed[:, 13]   # LKnee
    base_markers[:, 11] = transformed[:, 14]   # RKnee
    base_markers[:, 12] = transformed[:, 15]   # LAnkle
    base_markers[:, 13] = transformed[:, 16]   # RAnkle

    if extra_transformed is not None:
        # 22 markers: 14 body + 8 extra
        extra_names = [
            "LEye", "REye",
            "LThumb", "LIndex", "LPinky",
            "RThumb", "RIndex", "RPinky",
        ]
        marker_names = base_names + extra_names
        markers = np.zeros((T, 22, 3), dtype=np.float32)
        markers[:, :14] = base_markers
        markers[:, 14:] = extra_transformed
    else:
        marker_names = base_names
        markers = base_markers

    return markers, marker_names


def _project_extra_markers(
    kpts_2d: np.ndarray, body_3d_coco17: np.ndarray,
) -> np.ndarray:
    """
    Project 8 extra markers from 2D to 3D using body depth anchors.

    Markers projected:
        0: LEye   (COCO-WB idx 1)  - nose depth, shoulder-width scale
        1: REye   (COCO-WB idx 2)  - nose depth, shoulder-width scale
        2: LThumb (COCO-WB idx 95) - LWrist depth, forearm-length scale
        3: LIndex (COCO-WB idx 99) - LWrist depth, forearm-length scale
        4: LPinky (COCO-WB idx 111)- LWrist depth, forearm-length scale
        5: RThumb (COCO-WB idx 116)- RWrist depth, forearm-length scale
        6: RIndex (COCO-WB idx 120)- RWrist depth, forearm-length scale
        7: RPinky (COCO-WB idx 132)- RWrist depth, forearm-length scale

    Args:
        kpts_2d: (T, 133, 2) RTMW 2D keypoints in pixel coords
        body_3d_coco17: (T, 17, 3) MotionBERT 3D in H36M camera coords

    Returns:
        extra_3d: (T, 8, 3) projected markers in H36M camera coords
    """
    T = kpts_2d.shape[0]
    extra_3d = np.zeros((T, 8, 3), dtype=np.float32)

    # COCO-WB 2D indices
    EYE_2D = [1, 2]       # LEye, REye
    HAND_2D_L = [95, 99, 111]   # LThumb tip, LIndex tip, LPinky tip
    HAND_2D_R = [116, 120, 132] # RThumb tip, RIndex tip, RPinky tip

    # COCO-17 body indices (in body_3d_coco17)
    NOSE, LSHOULDER, RSHOULDER = 0, 5, 6
    LELBOW, RELBOW, LWRIST, RWRIST = 7, 8, 9, 10

    for t in range(T):
        # --- Eyes: project at nose depth using shoulder-width scale ---
        nose_3d = body_3d_coco17[t, NOSE]
        nose_2d = kpts_2d[t, 0]

        lsh_2d = kpts_2d[t, 5]
        rsh_2d = kpts_2d[t, 6]
        shoulder_2d_len = np.linalg.norm(lsh_2d - rsh_2d)

        lsh_3d = body_3d_coco17[t, LSHOULDER]
        rsh_3d = body_3d_coco17[t, RSHOULDER]
        shoulder_3d_len = np.linalg.norm(lsh_3d[:2] - rsh_3d[:2])

        if shoulder_2d_len > 20.0 and shoulder_3d_len > 0.01:
            eye_scale = shoulder_3d_len / shoulder_2d_len
        else:
            eye_scale = 0.35 / max(shoulder_2d_len, 100.0)

        for i, idx_2d in enumerate(EYE_2D):
            delta = kpts_2d[t, idx_2d] - nose_2d
            extra_3d[t, i, 0] = nose_3d[0] + delta[0] * eye_scale
            extra_3d[t, i, 1] = nose_3d[1] + delta[1] * eye_scale
            extra_3d[t, i, 2] = nose_3d[2]  # anchored at nose depth

        # --- Left hand: project at LWrist depth using forearm scale ---
        lwrist_3d = body_3d_coco17[t, LWRIST]
        lelbow_3d = body_3d_coco17[t, LELBOW]
        lwrist_2d = kpts_2d[t, 9]
        lelbow_2d = kpts_2d[t, 7]

        forearm_2d_len = np.linalg.norm(lwrist_2d - lelbow_2d)
        forearm_3d_len = np.linalg.norm(lwrist_3d[:2] - lelbow_3d[:2])

        if forearm_2d_len > 10.0 and forearm_3d_len > 0.01:
            lhand_scale = forearm_3d_len / forearm_2d_len
        else:
            lhand_scale = 0.25 / max(forearm_2d_len, 50.0)

        for i, idx_2d in enumerate(HAND_2D_L):
            delta = kpts_2d[t, idx_2d] - lwrist_2d
            extra_3d[t, 2 + i, 0] = lwrist_3d[0] + delta[0] * lhand_scale
            extra_3d[t, 2 + i, 1] = lwrist_3d[1] + delta[1] * lhand_scale
            extra_3d[t, 2 + i, 2] = lwrist_3d[2]  # anchored at wrist depth

        # --- Right hand: project at RWrist depth using forearm scale ---
        rwrist_3d = body_3d_coco17[t, RWRIST]
        relbow_3d = body_3d_coco17[t, RELBOW]
        rwrist_2d = kpts_2d[t, 10]
        relbow_2d = kpts_2d[t, 8]

        forearm_2d_len = np.linalg.norm(rwrist_2d - relbow_2d)
        forearm_3d_len = np.linalg.norm(rwrist_3d[:2] - relbow_3d[:2])

        if forearm_2d_len > 10.0 and forearm_3d_len > 0.01:
            rhand_scale = forearm_3d_len / forearm_2d_len
        else:
            rhand_scale = 0.25 / max(forearm_2d_len, 50.0)

        for i, idx_2d in enumerate(HAND_2D_R):
            delta = kpts_2d[t, idx_2d] - rwrist_2d
            extra_3d[t, 5 + i, 0] = rwrist_3d[0] + delta[0] * rhand_scale
            extra_3d[t, 5 + i, 1] = rwrist_3d[1] + delta[1] * rhand_scale
            extra_3d[t, 5 + i, 2] = rwrist_3d[2]  # anchored at wrist depth

    return extra_3d


def _normalize_skeleton_scale(body_3d_h36m: np.ndarray) -> np.ndarray:
    """
    Normalize MotionBERT skeleton to consistent size per frame.

    MotionBERT global model outputs camera-frame 3D where the skeleton
    scale changes with person-camera distance. Normalize each frame so
    the hip-to-nose distance matches the median across the sequence.
    """
    H36M_HIP = 0  # hip center
    H36M_NECK_NOSE = 9

    result = body_3d_h36m.copy()
    T = result.shape[0]

    # Compute per-frame trunk height (hip center to nose)
    trunk_heights = np.zeros(T)
    for t in range(T):
        trunk_heights[t] = np.linalg.norm(result[t, H36M_NECK_NOSE] - result[t, H36M_HIP])

    valid = trunk_heights > 0.1
    if np.sum(valid) < 10:
        return result

    target = np.median(trunk_heights[valid])
    print(f"  Skeleton normalization: median trunk={target*100:.1f}cm, range={trunk_heights[valid].min()*100:.1f}-{trunk_heights[valid].max()*100:.1f}cm")

    for t in range(T):
        if trunk_heights[t] > 0.1:
            scale = target / trunk_heights[t]
            hip = result[t, H36M_HIP]
            result[t] = hip + (result[t] - hip) * scale

    return result


def _normalize_h36m_bones(body_3d: np.ndarray) -> np.ndarray:
    """Normalize H36M bone lengths to per-sequence median."""
    # H36M bone chain (proximal → distal)
    bone_chains = [
        (0, 1), (1, 2), (2, 3),    # hip→Rhip→Rknee→Rfoot
        (0, 4), (4, 5), (5, 6),    # hip→Lhip→Lknee→Lfoot
        (0, 7), (7, 8),            # hip→spine→thorax
        (8, 9), (9, 10),           # thorax→neck→head
        (8, 11), (11, 12), (12, 13),  # thorax→Lshoulder→Lelbow→Lwrist
        (8, 14), (14, 15), (15, 16),  # thorax→Rshoulder→Relbow→Rwrist
    ]

    downstream = {
        1: [2, 3], 2: [3],
        4: [5, 6], 5: [6],
        7: [8, 9, 10, 11, 12, 13, 14, 15, 16],
        8: [9, 10, 11, 12, 13, 14, 15, 16],
        9: [10],
        11: [12, 13], 12: [13],
        14: [15, 16], 15: [16],
    }

    result = body_3d.copy()
    T = result.shape[0]

    # Compute median bone lengths
    medians = {}
    for p, c in bone_chains:
        lengths = np.linalg.norm(result[:, c] - result[:, p], axis=1)
        valid = lengths > 0.001
        if np.any(valid):
            medians[(p, c)] = np.median(lengths[valid])

    # Normalize each frame
    for t in range(T):
        for p, c in bone_chains:
            if (p, c) not in medians:
                continue
            target_len = medians[(p, c)]
            vec = result[t, c] - result[t, p]
            current_len = np.linalg.norm(vec)
            if current_len > 0.001:
                new_pos = result[t, p] + vec * (target_len / current_len)
                delta = new_pos - result[t, c]
                result[t, c] = new_pos
                for d in downstream.get(c, []):
                    result[t, d] += delta

    return result



def _smooth_h36m_body(body_3d: np.ndarray, fps: float, cutoff: float = 6.0) -> np.ndarray:
    """Smooth H36M body uniformly at the given cutoff frequency.

    All joints, all axes, same frequency — like Sam3D's approach.
    This preserves facing direction and global translation naturally.
    """
    from scipy.signal import butter, filtfilt

    result = body_3d.copy()
    T = result.shape[0]
    nyquist = fps / 2
    order = 4

    if T < 3 * order + 1 or cutoff >= nyquist or cutoff <= 0:
        return result

    b, a = butter(order, cutoff / nyquist, btype="low")
    for k in range(result.shape[1]):
        for dim in range(3):
            try:
                result[:, k, dim] = filtfilt(b, a, result[:, k, dim])
            except Exception:
                pass

    return result


def _smooth_keypoints_axis(
    keypoints: np.ndarray, fps: float,
    cutoff: float, depth_cutoff: float, depth_axis: int = 2,
) -> np.ndarray:
    """Apply Butterworth filter with axis-specific cutoffs."""
    from scipy.signal import butter, filtfilt

    result = keypoints.copy()
    T, K, _ = result.shape
    nyquist = fps / 2
    order = 4

    if T < 3 * order + 1:
        return result

    cutoffs = [cutoff] * 3
    cutoffs[depth_axis] = depth_cutoff

    for dim in range(3):
        c = cutoffs[dim]
        if c >= nyquist or c <= 0:
            continue
        b, a = butter(order, c / nyquist, btype="low")
        for k in range(K):
            try:
                result[:, k, dim] = filtfilt(b, a, result[:, k, dim])
            except Exception:
                pass

    return result


def _rerun_ik_light_regularization(output_dir: Path, trc_path: Path):
    """
    Re-run IK with 22 markers + light pelvis regularization.

    1. Adds 8 extra markers (LEye, REye, LThumb, LIndex, LPinky,
       RThumb, RIndex, RPinky) to the scaled .osim model.
    2. Creates a custom IK setup with 22 marker tasks (body=original
       weights, eyes=0.05, hands=0.2).
    3. Adds pelvis regularization coordinate tasks.
    """
    import subprocess
    import xml.etree.ElementTree as ET

    output_dir = Path(output_dir).resolve()
    trc_path = Path(trc_path).resolve()
    kin_dir = output_dir / "kinematics"

    ik_setups = list(kin_dir.glob("*_ik_setup.xml")) if kin_dir.exists() else []
    if not ik_setups:
        return None

    ik_setup_path = ik_setups[0]

    # --- Step 1: Add extra markers to the scaled .osim model ---
    tree = ET.parse(str(ik_setup_path))
    root = tree.getroot()
    model_file_node = root.find('.//model_file')
    if model_file_node is None or not model_file_node.text:
        return None
    osim_path = Path(model_file_node.text)
    if not osim_path.exists():
        print(f"  WARNING: .osim model not found: {osim_path}")
        return None

    # Parse the .osim model and add markers
    osim_tree = ET.parse(str(osim_path))
    osim_root = osim_tree.getroot()
    marker_set = osim_root.find('.//MarkerSet/objects')
    if marker_set is None:
        print("  WARNING: No MarkerSet found in .osim model")
        return None

    # Check which extra markers already exist
    existing_markers = {m.get('name') for m in marker_set.findall('Marker')}

    # Extra marker definitions: (name, parent_frame, location)
    # Locations from Pose2Sim's Markers_Coco133.xml
    extra_marker_defs = [
        ("LEye", "/bodyset/head", "0.0781497 0.0467611 -0.0311069"),
        ("REye", "/bodyset/head", "0.0781497 0.0467611 0.0311069"),
        ("LThumb", "/bodyset/hand_l", "0.015900595366914239 -0.053530050744837965 -0.045219604622807212"),
        ("LIndex", "/bodyset/hand_l", "0.0031182929587494357 -0.082610892492083754 -0.016029259621825068"),
        ("LPinky", "/bodyset/hand_l", "0.0031131181410346542 -0.073810317144973658 0.032544599634169884"),
        ("RThumb", "/bodyset/hand_r", "0.015900600000000001 -0.053530099999999997 0.045219599999999999"),
        ("RIndex", "/bodyset/hand_r", "0.0031182900000000001 -0.082610900000000001 0.0160293"),
        ("RPinky", "/bodyset/hand_r", "0.00311312 -0.073810299999999995 -0.0325446"),
    ]

    added_count = 0
    for name, parent, location in extra_marker_defs:
        if name in existing_markers:
            continue
        marker_elem = ET.SubElement(marker_set, 'Marker')
        marker_elem.set('name', name)
        ET.SubElement(marker_elem, 'socket_parent_frame').text = parent
        ET.SubElement(marker_elem, 'location').text = location
        ET.SubElement(marker_elem, 'fixed').text = 'true'
        added_count += 1

    if added_count > 0:
        # Save modified .osim model
        modified_osim = kin_dir / (osim_path.stem + "_22markers.osim")
        osim_tree.write(str(modified_osim), xml_declaration=True, encoding='UTF-8')
        print(f"  Added {added_count} markers to model -> {modified_osim.name}")
    else:
        modified_osim = osim_path

    # --- Step 2: Build custom IK setup with 22 marker tasks ---
    task_set = root.find('.//IKTaskSet/objects')
    if task_set is None:
        return None

    # Remove all existing tasks to rebuild cleanly
    for child in list(task_set):
        task_set.remove(child)

    # 22 marker tasks with appropriate weights
    marker_weights = {
        # Body markers (original COCO_17 weights)
        "Nose": 0.1,
        "Neck": 0.2,
        "LShoulder": 2.0, "RShoulder": 2.0,
        "LElbow": 1.0, "RElbow": 1.0,
        "LWrist": 1.0, "RWrist": 1.0,
        "LHip": 2.0, "RHip": 2.0,
        "LKnee": 1.0, "RKnee": 1.0,
        "LAnkle": 1.0, "RAnkle": 1.0,
        # Projected eyes (low confidence)
        "LEye": 0.05, "REye": 0.05,
        # Projected hand tips (medium-low confidence)
        "LThumb": 0.2, "LIndex": 0.2, "LPinky": 0.2,
        "RThumb": 0.2, "RIndex": 0.2, "RPinky": 0.2,
    }

    for name, weight in marker_weights.items():
        task = ET.SubElement(task_set, 'IKMarkerTask')
        task.set('name', name)
        ET.SubElement(task, 'apply').text = 'true'
        ET.SubElement(task, 'weight').text = str(weight)

    # Coordinate regularization
    coord_reg = ET.SubElement(task_set, 'IKCoordinateTask')
    coord_reg.set('name', 'L5_S1_Flex_Ext')
    ET.SubElement(coord_reg, 'apply').text = 'true'
    ET.SubElement(coord_reg, 'weight').text = '0.1'
    ET.SubElement(coord_reg, 'value_type').text = 'default_value'
    ET.SubElement(coord_reg, 'value').text = '0'

    # Pelvis regularization
    pelvis_coords = [
        ('pelvis_tilt', 0.05),
        ('pelvis_list', 0.005),
        ('hip_adduction_r', 0.005),
        ('hip_adduction_l', 0.005),
    ]
    for coord_name, weight in pelvis_coords:
        task = ET.SubElement(task_set, 'IKCoordinateTask')
        task.set('name', coord_name)
        ET.SubElement(task, 'apply').text = 'true'
        ET.SubElement(task, 'weight').text = str(weight)
        ET.SubElement(task, 'value_type').text = 'default_value'
        ET.SubElement(task, 'value').text = '0'

    # Point to modified model and TRC
    model_file_node.text = str(modified_osim)

    marker_file_node = root.find('.//marker_file')
    if marker_file_node is not None:
        marker_file_node.text = str(trc_path)

    mot_stem = trc_path.stem
    new_mot_path = kin_dir / f"{mot_stem}.mot"
    mot_node = root.find('.//output_motion_file')
    if mot_node is not None:
        mot_node.text = str(new_mot_path)

    modified_setup = kin_dir / f"{mot_stem}_ik_setup_pass2.xml"
    tree.write(str(modified_setup), xml_declaration=True, encoding='UTF-8')

    # --- Step 3: Run IK ---
    POSE2SIM_PYTHON = os.environ.get("POSE2SIM_PYTHON", r"C:\ProgramData\anaconda3\envs\Pose2Sim\python.exe")
    ik_script = f'''
import opensim
tool = opensim.InverseKinematicsTool(r"{modified_setup}")
tool.run()
print("SUCCESS: Pass 2 IK complete (22 markers)")
'''
    script_path = output_dir / "_run_pass2_ik.py"
    with open(script_path, 'w') as f:
        f.write(ik_script)

    result = subprocess.run(
        [POSE2SIM_PYTHON, str(script_path)],
        cwd=str(kin_dir),
        capture_output=True,
        text=True,
    )

    if result.stdout:
        for line in result.stdout.split('\n'):
            if line.strip():
                print(f"  {line}")
    if result.stderr:
        for line in result.stderr.split('\n'):
            if line.strip() and 'info' not in line.lower():
                print(f"  {line}")

    script_path.unlink(missing_ok=True)

    if result.returncode == 0 and new_mot_path.exists():
        print(f"  Pass 2 MOT: {new_mot_path}")
        # Remove Pass 1 .osim model (only keep the 22-marker model)
        if osim_path.exists() and modified_osim != osim_path:
            osim_path.unlink()
            print(f"  Removed Pass 1 model: {osim_path.name}")
        return new_mot_path
    else:
        print("  Pass 2 IK failed, using Pose2Sim result")
        return None


def _post_smooth_mot(mot_path: Path, fps: float) -> Path:
    """Post-process joint angles in the MOT file.

    - hip_rotation: zero (COCO markers can't constrain thigh rotation)
    """
    mot_path = Path(mot_path)
    with open(mot_path, 'r') as f:
        lines = f.readlines()

    # Find header
    header_end = 0
    for i, line in enumerate(lines):
        if line.strip() == 'endheader':
            header_end = i + 1
            break

    header_lines = lines[:header_end + 1]  # includes column names row
    col_names = lines[header_end].strip().split('\t')
    data_lines = lines[header_end + 1:]

    # Parse data
    data = []
    for line in data_lines:
        vals = line.strip().split('\t')
        if len(vals) == len(col_names):
            data.append([float(v) for v in vals])
    data = np.array(data)
    T = data.shape[0]

    if T < 13:
        return mot_path

    modified = 0

    # Zero hip rotation (unconstrained by COCO markers)
    zeroed = ['hip_rotation_r', 'hip_rotation_l']
    for joint in zeroed:
        if joint in col_names:
            idx = col_names.index(joint)
            data[:, idx] = 0.0
            modified += 1

    if modified == 0:
        return mot_path

    # Write back
    out_lines = header_lines[:]
    for row in data:
        out_lines.append('\t'.join(f'{v:.6f}' for v in row) + '\n')

    with open(mot_path, 'w') as f:
        f.writelines(out_lines)

    print(f"  Post-process: zeroed hip_rotation")
    return mot_path


def _find_inference_output(video_path: Path) -> Path:
    """Find the latest inference output folder for a given video.

    Looks for output_*_{video_stem}/video_outputs.json in the project root,
    sorted by modification time (newest first).
    """
    video_stem = video_path.stem
    candidates = sorted(
        PROJECT_ROOT.glob(f"output_*_{video_stem}/video_outputs.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    return None


def main():
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input not found: {input_path}")
        sys.exit(1)

    # Accept either a video file or video_outputs.json
    if input_path.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"):
        json_path = _find_inference_output(input_path)
        if json_path is None:
            print(f"Error: No inference output found for {input_path.name}")
            print(f"  Run inference first: python run_inference.py --input \"{input_path}\"")
            sys.exit(1)
        print(f"Found inference output: {json_path.parent.name}/")
    elif input_path.name == "video_outputs.json":
        json_path = input_path
    else:
        # Try as a directory containing video_outputs.json
        candidate = input_path / "video_outputs.json"
        if candidate.exists():
            json_path = candidate
        else:
            json_path = input_path  # let it fail below with a clear error

    if not json_path.exists():
        print(f"Error: video_outputs.json not found: {json_path}")
        sys.exit(1)

    if args.output:
        output_dir = Path(args.output)
    else:
        # Output to the same folder as video_outputs.json
        output_dir = json_path.parent

    run_hybrid_pipeline(
        json_path=str(json_path),
        output_dir=str(output_dir),
        subject_height=args.height,
        subject_mass=args.mass,
        fps=args.fps,
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


if __name__ == "__main__":
    main()
