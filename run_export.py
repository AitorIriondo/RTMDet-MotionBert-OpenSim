#!/usr/bin/env python3
"""
Export RTMPose3D Results to OpenSim
====================================

Reads video_outputs.json and exports to TRC/MOT/FBX.
Uses Pose2Sim's kinematics() for proper model scaling + IK.

Usage:
    python run_export.py --input output_dir/video_outputs.json --height 1.69
    python run_export.py --input video_outputs.json --height 1.69 --skip-ik
    python run_export.py --input video_outputs.json --height 1.69 --smooth 6.0
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Export RTMPose3D outputs to OpenSim")
    parser.add_argument("--input", "-i", required=True, help="video_outputs.json file")
    parser.add_argument("--height", type=float, default=1.75, help="Subject height (m)")
    parser.add_argument("--mass", type=float, default=70.0, help="Subject mass (kg)")
    parser.add_argument("--output", "-o", help="Output directory (default: same as input)")
    parser.add_argument("--fps", type=float, help="Override FPS (default: from metadata)")
    parser.add_argument("--skip-ik", action="store_true", help="Skip OpenSim scaling + IK")
    parser.add_argument("--skip-fbx", action="store_true", help="Skip FBX export")
    parser.add_argument("--person", type=int, default=0, help="Person index")
    parser.add_argument("--smooth", type=float, default=6.0, help="Smoothing cutoff Hz (0=disable)")
    return parser.parse_args()


def load_rtmpose3d_outputs(json_path: str) -> list:
    """Load RTMPose3D outputs from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected list of frame outputs")
    return data


def extract_keypoints(data: list, person_idx: int = 0) -> tuple:
    """Extract keypoints and scores from RTMPose3D JSON data."""
    num_frames = len(data)
    keypoints_3d = np.zeros((num_frames, 133, 3), dtype=np.float32)
    scores = np.zeros((num_frames, 133), dtype=np.float32)
    valid_frames = np.zeros(num_frames, dtype=bool)

    for i, frame_data in enumerate(data):
        outputs = frame_data.get("outputs", [])
        if len(outputs) > person_idx:
            person = outputs[person_idx]
            kp3d = person.get("keypoints_3d", [])
            if len(kp3d) == 133:
                keypoints_3d[i] = np.array(kp3d)
                valid_frames[i] = True
            sc = person.get("scores", [])
            if len(sc) == 133:
                scores[i] = np.array(sc)

    return keypoints_3d, scores, valid_frames


def run_export(
    json_path: str,
    output_dir: str,
    subject_height: float,
    subject_mass: float,
    fps: float,
    skip_ik: bool,
    skip_fbx: bool,
    person_idx: int,
    smooth_cutoff: float = 6.0,
):
    """Export RTMPose3D outputs to TRC/MOT/FBX."""
    start_time = time.time()

    json_path = Path(json_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("RTMPose3D Export to OpenSim")
    print(f"{'='*60}")
    print(f"Input: {json_path}")
    print(f"Output: {output_dir}")
    print(f"Subject: height={subject_height}m, mass={subject_mass}kg")
    print(f"{'='*60}\n")

    # Load data
    print("[1/5] Loading RTMPose3D outputs...")
    data = load_rtmpose3d_outputs(str(json_path))
    keypoints_3d, scores, valid_frames = extract_keypoints(data, person_idx)
    print(f"  Loaded {len(data)} frames, {np.sum(valid_frames)} valid detections")

    # Get FPS from metadata
    meta_path = json_path.parent / "inference_meta.json"
    if fps is None:
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            fps = meta.get("fps", 30.0)
            print(f"  FPS from metadata: {fps}")
        else:
            fps = 30.0
            print(f"  Using default FPS: {fps}")

    # Post-process: interpolation + optional smoothing
    print("\n[2/5] Post-processing keypoints...")
    from src.post_processing import PostProcessor
    from src.coordinate_transform import CoordinateTransformer

    use_smoothing = smooth_cutoff > 0
    if use_smoothing:
        print(f"  Smoothing: {smooth_cutoff} Hz Butterworth filter")
    post_processor = PostProcessor(
        smooth_filter=use_smoothing,
        filter_cutoff=smooth_cutoff,
        normalize_bones=True,  # Enforce consistent bone lengths across frames
    )
    keypoints_processed = post_processor.process(keypoints_3d, fps=fps)

    # Transform coordinates: RTMPose3D -> OpenSim
    # Use meters for Pose2Sim compatibility (scaling computes TRC/model ratio)
    transformer = CoordinateTransformer(subject_height=subject_height, units="m")
    keypoints_opensim = transformer.transform(
        keypoints_processed,
        center_pelvis=True,
        align_to_ground=True,
        correct_lean=False,
        depth_factor=1.0,  # v19: preserve full depth signal
    )
    print("  Coordinate transformation complete")

    # Convert to Pose2Sim COCO_133 markers
    print("\n[3/5] Converting to Pose2Sim COCO_133 markers...")
    from src.keypoint_converter import KeypointConverter

    converter = KeypointConverter()
    markers, marker_names = converter.convert(keypoints_opensim)
    print(f"  Generated {len(marker_names)} markers")

    # Export TRC
    print("\n[4/5] Exporting TRC file...")
    from src.trc_exporter import TRCExporter

    video_name = json_path.parent.name
    if video_name in (".", ""):
        video_name = json_path.stem.replace("video_outputs", "export")

    trc_exporter = TRCExporter(fps=fps, units="m")
    trc_path = output_dir / f"markers_{video_name}.trc"
    trc_exporter.export(markers, marker_names, str(trc_path))
    print(f"  Saved: {trc_path}")

    results = {"trc": trc_path, "mot": None, "fbx": None}

    # Run Pose2Sim scaling + IK
    if not skip_ik:
        print("\n[5/5] Running Pose2Sim scaling + IK...")
        mot_path = run_pose2sim_kinematics(
            trc_path, output_dir, subject_height, subject_mass
        )
        if mot_path:
            # Re-run IK with pelvis regularization for stability
            mot_path = rerun_ik_with_pelvis_regularization(
                output_dir, trc_path
            ) or mot_path
        results["mot"] = mot_path
    else:
        print("\n[5/5] Skipping scaling + IK")

    # Export FBX
    if not skip_fbx and not skip_ik and results["mot"]:
        print("\nExporting FBX...")
        fbx_path = run_fbx_export(results["mot"], output_dir)
        results["fbx"] = fbx_path

    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print("Export Complete!")
    print(f"{'='*60}")
    print(f"Time: {elapsed:.1f}s")
    print(f"\nOutput files:")
    for name, path in results.items():
        status = "OK" if path and Path(path).exists() else "SKIPPED"
        print(f"  [{status}] {name.upper()}: {path}")
    print(f"{'='*60}\n")

    return results


def _check_pose2sim_marker_bug(pose2sim_python: str):
    """Check for and auto-fix the Pose2Sim Markers_Coco17.xml corruption bug."""
    fix_script = Path(__file__).parent / "fix_pose2sim.py"
    if not fix_script.exists():
        return

    import subprocess as sp
    result = sp.run(
        [pose2sim_python, str(fix_script), "--check"],
        capture_output=True, text=True,
    )
    if result.returncode != 0 and "Corruption detected" in (result.stdout + result.stderr):
        print("  WARNING: Pose2Sim Markers_Coco17.xml bug detected! Auto-fixing...")
        fix_result = sp.run(
            [pose2sim_python, str(fix_script)],
            capture_output=True, text=True,
        )
        for line in (fix_result.stdout or "").split("\n"):
            if line.strip():
                print(f"  {line}")


def run_pose2sim_kinematics(trc_path: Path, output_dir: Path,
                             height: float, mass: float,
                             pose_model: str = "COCO_133"):
    """
    Run Pose2Sim scaling + IK using the Pose2Sim environment.

    Pose2Sim's kinematics() does:
    1. Model scaling (per-segment, using trimmed marker distances)
    2. Inverse kinematics (on scaled model)

    This replicates the approach used in MotionBERT-OpenSim.
    """
    import subprocess

    trc_path = Path(trc_path).resolve()
    output_dir = Path(output_dir).resolve()

    # Find Python with OpenSim + Pose2Sim
    POSE2SIM_PYTHON = r"C:\Users\iria\AppData\Local\anaconda3\envs\Pose2Sim\python.exe"
    if not Path(POSE2SIM_PYTHON).exists():
        # Try current env
        try:
            import opensim
            POSE2SIM_PYTHON = sys.executable
        except ImportError:
            print("  ERROR: Cannot find Pose2Sim/OpenSim environment")
            return None
    print(f"  Using: {POSE2SIM_PYTHON}")

    # Check for Pose2Sim marker bug (auto-fixes if detected)
    _check_pose2sim_marker_bug(POSE2SIM_PYTHON)

    # Pose2Sim expects TRC in {project_dir}/pose-3d/
    pose3d_dir = output_dir / "pose-3d"
    pose3d_dir.mkdir(parents=True, exist_ok=True)

    # Copy TRC to pose-3d directory
    trc_dest = pose3d_dir / trc_path.name
    shutil.copy2(str(trc_path), str(trc_dest))
    print(f"  TRC copied to: {trc_dest}")

    # Build the Pose2Sim kinematics script
    kin_script = '''
import sys
import json
from pathlib import Path

# Pose2Sim kinematics config (same approach as MotionBERT-OpenSim)
config_dict = {{
    'project': {{
        'project_dir': r'{project_dir}',
        'multi_person': False,
        'participant_height': {height},
        'participant_mass': {mass},
        'frame_rate': 'auto',
        'frame_range': 'all',
    }},
    'pose': {{
        'pose_model': '{pose_model}',
    }},
    'markerAugmentation': {{
        'feet_on_floor': False,
        'make_c3d': False,
    }},
    'kinematics': {{
        'use_augmentation': False,
        'use_simple_model': True,
        'right_left_symmetry': True,
        'default_height': {height},
        'remove_individual_scaling_setup': False,
        'remove_individual_ik_setup': False,
        'fastest_frames_to_remove_percent': 0.1,
        'close_to_zero_speed_m': 0.2,
        'large_hip_knee_angles': 45,
        'trimmed_extrema_percent': 0.5,
    }},
    'logging': {{
        'use_custom_logging': False,
    }},
}}

print("Running Pose2Sim kinematics (scaling + IK)...")
from Pose2Sim import Pose2Sim as P2S
P2S.kinematics(config_dict)
print("SUCCESS: Pose2Sim kinematics complete")
'''.format(
        project_dir=str(output_dir),
        height=height,
        mass=mass,
        pose_model=pose_model,
    )

    kin_script_path = output_dir / "_run_pose2sim_kin.py"
    with open(kin_script_path, 'w') as f:
        f.write(kin_script)

    result = subprocess.run(
        [POSE2SIM_PYTHON, str(kin_script_path)],
        cwd=str(output_dir),
        capture_output=True,
        text=True,
    )

    # Print output
    if result.stdout:
        for line in result.stdout.split('\n'):
            if line.strip():
                print(f"  {line}")
    if result.stderr:
        for line in result.stderr.split('\n'):
            if line.strip() and not line.startswith('[info]'):
                print(f"  {line}")

    if result.returncode != 0:
        print(f"  ERROR: Pose2Sim kinematics failed (exit code {result.returncode})")
        # Fall back to basic IK if Pose2Sim fails
        print("  Falling back to basic OpenSim IK...")
        kin_script_path.unlink(missing_ok=True)
        return run_basic_opensim_ik(trc_path, output_dir, height, mass)

    kin_script_path.unlink(missing_ok=True)

    # Find output MOT file
    # Pose2Sim puts results in {project_dir}/kinematics/
    kin_dir = output_dir / "kinematics"
    mot_files = list(kin_dir.glob("*.mot")) if kin_dir.exists() else []
    if not mot_files:
        # Also check output_dir directly
        mot_files = list(output_dir.glob("*_ik.mot"))

    if mot_files:
        mot_path = mot_files[0]
        print(f"  MOT file: {mot_path}")
        return mot_path
    else:
        print("  WARNING: No MOT file found after Pose2Sim kinematics")
        return None


def rerun_ik_with_pelvis_regularization(output_dir: Path, trc_path: Path):
    """
    Two-pass IK to stabilize pelvis rotation.

    Pass 1: Pose2Sim's IK (already done) — pelvis_rotation is noisy/flipping
    Pass 2: Re-run IK with pelvis_rotation constrained to smoothed trajectory
    from pass 1. This prevents 180° flips while preserving natural rotation.
    """
    import subprocess
    import xml.etree.ElementTree as ET

    output_dir = Path(output_dir).resolve()
    trc_path = Path(trc_path).resolve()
    kin_dir = output_dir / "kinematics"

    # Find the Pose2Sim-generated IK setup and MOT
    ik_setups = list(kin_dir.glob("*_ik_setup.xml"))
    mot_files = list(kin_dir.glob("*.mot"))
    if not ik_setups or not mot_files:
        print("  No IK setup/MOT found, skipping pelvis regularization")
        return None

    ik_setup_path = ik_setups[0]
    pass1_mot = mot_files[0]

    # Read pass 1 pelvis_rotation and smooth it
    print("  Pass 2: Stabilizing pelvis rotation...")
    pelvis_rot_smooth = _smooth_pelvis_rotation(pass1_mot)
    if pelvis_rot_smooth is None:
        return None

    # Write smoothed pelvis rotation as coordinate file
    coord_file = kin_dir / "_pelvis_rotation_smooth.sto"
    _write_coordinate_file(coord_file, pass1_mot, pelvis_rot_smooth)

    # Modify IK setup: add pelvis_rotation from_file constraint
    tree = ET.parse(str(ik_setup_path))
    root = tree.getroot()

    task_set = root.find('.//IKTaskSet/objects')
    if task_set is None:
        return None

    # Add pelvis coordinate tasks — gentle regularization toward default
    # pelvis_rotation: prevents 180° facing flips from flat depth
    # pelvis_list: reduces frontal plane wobble from L/R asymmetry
    # pelvis_tilt: keeps forward lean centered
    pelvis_coords = [
        ('pelvis_rotation', 0.005),
        ('pelvis_list', 0.005),
        ('pelvis_tilt', 0.005),
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

    # Update output path
    mot_stem = trc_path.stem
    new_mot_path = kin_dir / f"{mot_stem}.mot"
    mot_node = root.find('.//output_motion_file')
    if mot_node is not None:
        mot_node.text = str(new_mot_path)

    modified_setup = kin_dir / f"{mot_stem}_ik_setup_pass2.xml"
    tree.write(str(modified_setup), xml_declaration=True, encoding='UTF-8')

    # Run pass 2 IK
    POSE2SIM_PYTHON = r"C:\Users\iria\AppData\Local\anaconda3\envs\Pose2Sim\python.exe"
    ik_script = f'''
import opensim
tool = opensim.InverseKinematicsTool(r"{modified_setup}")
tool.run()
print("SUCCESS: Pass 2 IK complete")
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
        return new_mot_path
    else:
        print("  Pass 2 IK failed, using Pose2Sim result")
        return None


def _smooth_pelvis_rotation(mot_path: Path):
    """Extract and smooth pelvis_rotation from pass 1 MOT file."""
    with open(mot_path) as f:
        lines = f.readlines()

    for i, l in enumerate(lines):
        if l.startswith('time'):
            header = l.strip().split('\t')
            data_start = i + 1
            break
    else:
        return None

    data = []
    for l in lines[data_start:]:
        vals = l.strip().split('\t')
        if len(vals) == len(header):
            data.append([float(v) for v in vals])
    data = np.array(data)

    if 'pelvis_rotation' not in header:
        return None

    rot_idx = header.index('pelvis_rotation')
    rot = data[:, rot_idx].copy()

    # Unwrap large jumps (>90° between consecutive frames = likely flip)
    for i in range(1, len(rot)):
        diff = rot[i] - rot[i-1]
        if abs(diff) > 90:
            # Snap to nearest equivalent angle
            rot[i] -= round(diff / 180) * 180

    # Heavy smoothing with median filter + lowpass
    from scipy.ndimage import median_filter, uniform_filter1d
    if len(rot) > 15:
        rot = median_filter(rot, size=15)
        rot = uniform_filter1d(rot.astype(float), size=31)

    return rot


def _write_coordinate_file(path: Path, mot_path: Path, pelvis_rot: np.ndarray):
    """Write an OpenSim .sto coordinate file with smoothed pelvis rotation."""
    with open(mot_path) as f:
        lines = f.readlines()

    for i, l in enumerate(lines):
        if l.startswith('time'):
            header = l.strip().split('\t')
            data_start = i + 1
            break

    data = []
    for l in lines[data_start:]:
        vals = l.strip().split('\t')
        if len(vals) == len(header):
            data.append([float(v) for v in vals])
    data = np.array(data)

    times = data[:, header.index('time')]

    with open(path, 'w') as f:
        f.write("Coordinates\n")
        f.write("version=1\n")
        f.write(f"nRows={len(times)}\n")
        f.write("nColumns=2\n")
        f.write("inDegrees=yes\n")
        f.write("endheader\n")
        f.write("time\tpelvis_rotation\n")
        for t, r in zip(times, pelvis_rot):
            f.write(f"{t}\t{r}\n")


def run_basic_opensim_ik(trc_path: Path, output_dir: Path,
                          height: float, mass: float):
    """Fallback: basic OpenSim IK without Pose2Sim scaling."""
    import subprocess

    POSE2SIM_PYTHON = r"C:\Users\iria\AppData\Local\anaconda3\envs\Pose2Sim\python.exe"
    trc_path = Path(trc_path).resolve()
    output_dir = Path(output_dir).resolve()

    # Find Pose2Sim setup path
    pose2sim_setup = Path(r"C:\Users\iria\AppData\Local\anaconda3\envs\Pose2Sim\Lib\site-packages\Pose2Sim\OpenSim_Setup")

    ik_script = '''
import opensim as osim
from pathlib import Path

trc_path = Path(r"{trc_path}")
output_dir = Path(r"{output_dir}")
pose2sim_setup = Path(r"{pose2sim_setup}")

model_path = pose2sim_setup / "Model_Pose2Sim_simple.osim"
marker_path = pose2sim_setup / "Markers_Coco133.xml"

def get_trc_time_range(trc_file):
    with open(trc_file, 'r') as f:
        lines = f.readlines()
    header_values = lines[2].split("\\t")
    data_rate = float(header_values[0])
    num_frames = int(header_values[2])
    return (0.0, (num_frames - 1) / data_rate)

time_range = get_trc_time_range(str(trc_path))
stem = trc_path.stem
mot_path = output_dir / f"{{stem}}_ik.mot"

model = osim.Model(str(model_path))

# Add markers from Pose2Sim's marker set
marker_set = osim.MarkerSet(str(marker_path))
for i in range(marker_set.getSize()):
    marker = marker_set.get(i).clone()
    model.addMarker(marker)

model.finalizeConnections()
model.initSystem()

ik_tool = osim.InverseKinematicsTool()
ik_tool.setModel(model)
ik_tool.setMarkerDataFileName(str(trc_path))
ik_tool.setStartTime(time_range[0])
ik_tool.setEndTime(time_range[1])
ik_tool.setOutputMotionFileName(str(mot_path))
ik_tool.setResultsDir(str(output_dir))

try:
    ik_tool.run()
    print(f"SUCCESS: {{mot_path}}")
except Exception as e:
    print(f"ERROR: {{e}}")
    raise
'''.format(trc_path=trc_path, output_dir=output_dir, pose2sim_setup=pose2sim_setup)

    ik_script_path = output_dir / "_run_ik_fallback.py"
    with open(ik_script_path, 'w') as f:
        f.write(ik_script)

    result = subprocess.run(
        [POSE2SIM_PYTHON, str(ik_script_path)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )

    print(result.stdout)
    if result.stderr:
        for line in result.stderr.split('\n'):
            if line and not line.startswith('[info]'):
                print(line)

    ik_script_path.unlink(missing_ok=True)

    mot_files = list(output_dir.glob("*_ik.mot"))
    return mot_files[0] if mot_files else None


def run_fbx_export(mot_path: Path, output_dir: Path):
    """Export FBX + GLB using Blender with skeleton template.

    The Blender script exports both formats:
    - FBX: works in Blender
    - GLB (binary glTF): works in all external viewers (quaternion-native)
    """
    import subprocess

    BLENDER_PATH = r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe"
    blend_template = PROJECT_ROOT / "Import_OS4_Patreon_Aitor_Skely.blend"
    blender_script = PROJECT_ROOT / "scripts" / "export_fbx_skely.py"

    mot_path = Path(mot_path).resolve()
    fbx_path = output_dir / f"{mot_path.stem.replace('_ik', '')}.fbx"
    glb_path = fbx_path.with_suffix('.glb')

    if not Path(BLENDER_PATH).exists():
        print(f"  Blender not found: {BLENDER_PATH}")
        return None

    if not blend_template.exists():
        print(f"  Skeleton template not found: {blend_template}")
        return None

    cmd = [
        BLENDER_PATH, "--background", str(blend_template),
        "--python", str(blender_script),
        "--", "--mot", str(mot_path), "--output", str(fbx_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"  FBX: {fbx_path}")
        if glb_path.exists():
            print(f"  GLB: {glb_path}")
        return fbx_path
    else:
        if result.stderr:
            for line in result.stderr.split('\n'):
                if line and 'Error' in line:
                    print(f"  {line}")
        return None


def main():
    args = parse_args()

    json_path = Path(args.input)
    if not json_path.exists():
        print(f"Error: Input not found: {json_path}")
        sys.exit(1)

    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = json_path.parent

    run_export(
        json_path=str(json_path),
        output_dir=str(output_dir),
        subject_height=args.height,
        subject_mass=args.mass,
        fps=args.fps,
        skip_ik=args.skip_ik,
        skip_fbx=args.skip_fbx,
        person_idx=args.person,
        smooth_cutoff=args.smooth,
    )


if __name__ == "__main__":
    main()
