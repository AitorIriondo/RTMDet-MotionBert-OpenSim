# How to Run RTMToOpenSim

Monocular video to OpenSim motion data (TRC / MOT / GLB / FBX).

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [MotionBERT Model Download](#motionbert-model-download)
4. [Running the Pipeline](#running-the-pipeline)
5. [Command-Line Reference](#command-line-reference)
6. [Output Files](#output-files)
7. [Viewing Results](#viewing-results)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Hardware

- **GPU**: NVIDIA GPU with CUDA support (recommended). CPU works but is ~10x slower for inference.
- **VRAM**: 4 GB minimum, 8 GB recommended.
- **RAM**: 8 GB minimum.
- **Disk**: ~2 GB for models and environments.

### Software

| Software | Version | Required |
|----------|---------|----------|
| Anaconda / Miniconda | any | Yes |
| CUDA Toolkit | 12.1+ | Yes (for GPU) |
| Blender | 5.0+ | Optional (GLB/FBX export only) |

---

## Environment Setup

The pipeline uses **two conda environments**:

| Environment | Purpose | Python |
|-------------|---------|--------|
| `mmpose` | Main pipeline (inference, MotionBERT, export) | 3.10 |
| `Pose2Sim` | OpenSim inverse kinematics (runs as subprocess) | 3.12 |

You only interact with `mmpose` directly. The `Pose2Sim` environment is called automatically as a subprocess for the IK step.

### 1. Create the `mmpose` environment

```bash
conda create -n mmpose python=3.10 -y
conda activate mmpose
```

### 2. Install PyTorch with CUDA

```bash
# For CUDA 12.1 (adjust URL for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Verify:
```bash
python -c "import torch; print(torch.cuda.is_available())"
# Should print: True
```

### 3. Install the mmpose ecosystem

```bash
pip install mmengine>=0.7.0
pip install mmcv>=2.0.0
pip install mmdet>=3.0.0
pip install mmpose>=1.0.0
```

> **Note (Windows):** Pre-built `mmcv` wheels with CUDA ops may not be available for Windows. Check [mmcv installation docs](https://mmcv.readthedocs.io/en/latest/get_started/installation.html) for platform-specific instructions. You may need to build from source.

### 4. Install project dependencies

```bash
cd C:\RTMToOpenSim
pip install -r requirements.txt
```

### 5. Install rtmpose3d

```bash
pip install rtmpose3d
# Or from source:
pip install git+https://github.com/b-arac/rtmpose3d.git
```

### 6. Install xtcocotools (if needed)

Must be built from source for numpy compatibility:
```bash
pip install --no-binary xtcocotools --no-build-isolation xtcocotools
```

### 7. Create the `Pose2Sim` environment (for IK)

```bash
conda create -n Pose2Sim python=3.12 -y
conda activate Pose2Sim
conda install -c opensim-org opensim=4.5.2
pip install pose2sim>=0.10.0
```

> **Important:** OpenSim is NOT available via pip. It must be installed via conda from the `opensim-org` channel.

Switch back to the main environment:
```bash
conda activate mmpose
```

### 8. (Optional) Install Blender for GLB/FBX export

Download and install [Blender 5.0+](https://www.blender.org/download/) to the default location:
```
C:\Program Files\Blender Foundation\Blender 5.0\blender.exe
```

The pipeline will skip GLB/FBX export automatically if Blender is not found.

---

## MotionBERT Model Download

The hybrid pipeline requires a pre-trained MotionBERT checkpoint.

### Download the checkpoint

1. Download `FT_MB_lite_MB_ft_h36m_global_lite` from the [MotionBERT releases](https://github.com/Walter0807/MotionBERT/releases) or the Google Drive link in the MotionBERT README.

2. Place it at:
   ```
   models/MotionBERT/checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin
   ```

### Verify the model

```bash
python -c "from src.motionbert_inference import MotionBERTLifter; print('OK')"
```

---

## Running the Pipeline

### Activate the environment

```bash
conda activate mmpose
```

On Windows, all commands use the full Python path to avoid PATH issues:
```
C:/ProgramData/anaconda3/envs/mmpose/python.exe
```

Or if you activated the environment, just use `python`.

### Recommended: Two-stage workflow

The pipeline is split into two stages so you can run the slow inference once, then iterate quickly on export settings.

#### Stage 1: Inference (slow, run once per video)

```bash
python run_inference.py --input path/to/video.mp4
```

This creates:
```
test_output_<video_name>/
    frames/                   # Extracted video frames
    video_outputs.json        # 2D + 3D keypoints per frame
    inference_meta.json       # Video metadata (FPS, resolution)
```

#### Stage 2: Hybrid export (fast, iterate settings)

```bash
python run_hybrid_pipeline.py ^
    --input test_output_<video_name>/video_outputs.json ^
    --height 1.69
```

This creates TRC, MOT, GLB, and FBX output files (see [Output Files](#output-files)).

### Quick one-liner

```bash
python run_inference.py --input video.mp4 && ^
python run_hybrid_pipeline.py --input test_output_video/video_outputs.json --height 1.69
```

---

## Command-Line Reference

### run_inference.py

Runs RTMW inference on a video. Produces 2D and 3D keypoints.

```bash
python run_inference.py [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input`, `-i` | **required** | Input video file path |
| `--output`, `-o` | auto | Output directory. Default: auto-generated from video name |
| `--fps` | `30.0` | Target FPS for frame extraction |
| `--device` | `cuda:0` | Compute device. Use `cpu` if no GPU |
| `--config` | None | Optional YAML config file |
| `--person` | `0` | Person index (0 = first detected person) |
| `--model` | None | Model name override (e.g. `rtmpose3d-x` for XL) |

**Examples:**

```bash
# Basic usage
python run_inference.py --input walk.mp4

# Custom output directory, CPU mode
python run_inference.py --input walk.mp4 --output my_output --device cpu

# Extract at 15 FPS instead of 30
python run_inference.py --input walk.mp4 --fps 15
```

---

### run_hybrid_pipeline.py (Recommended)

Converts inference output to OpenSim motion data using MotionBERT 3D lifting.

```bash
python run_hybrid_pipeline.py [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input`, `-i` | **required** | Path to `video_outputs.json` from Stage 1 |
| `--output`, `-o` | same as input dir | Output directory |
| `--height` | `1.75` | Subject height in meters |
| `--mass` | `70.0` | Subject mass in kilograms |
| `--fps` | from metadata | Override FPS. Default: read from `inference_meta.json` |
| `--smooth` | `6.0` | Butterworth low-pass filter cutoff in Hz. `0` = disable smoothing |
| `--skip-ik` | false | Skip OpenSim scaling + inverse kinematics. Outputs TRC only |
| `--skip-fbx` | false | Skip Blender GLB/FBX export |
| `--person` | `0` | Person index (0 = first person) |
| `--device` | `cuda:0` | Device for MotionBERT inference. Use `cpu` if no GPU |
| `--pose-model` | `COCO_17` | Pose model for IK. `COCO_17` = 22 markers (recommended), `COCO_133` = 27 markers |

**Examples:**

```bash
# Standard usage with subject measurements
python run_hybrid_pipeline.py ^
    --input test_output/video_outputs.json ^
    --height 1.69 ^
    --mass 65

# Output to a specific directory
python run_hybrid_pipeline.py ^
    --input test_output/video_outputs.json ^
    --height 1.69 ^
    --output my_results

# Skip IK (TRC only, for debugging markers)
python run_hybrid_pipeline.py ^
    --input test_output/video_outputs.json ^
    --height 1.69 ^
    --skip-ik

# Stronger smoothing (3 Hz) for noisy video
python run_hybrid_pipeline.py ^
    --input test_output/video_outputs.json ^
    --height 1.69 ^
    --smooth 3.0

# No smoothing
python run_hybrid_pipeline.py ^
    --input test_output/video_outputs.json ^
    --height 1.69 ^
    --smooth 0

# CPU only (no GPU)
python run_hybrid_pipeline.py ^
    --input test_output/video_outputs.json ^
    --height 1.69 ^
    --device cpu

# COCO_133 mode (27 markers, includes projected hands/feet/face)
python run_hybrid_pipeline.py ^
    --input test_output/video_outputs.json ^
    --height 1.69 ^
    --pose-model COCO_133
```

---

### run_export.py (Legacy)

Legacy export using RTMW3D direct 3D (without MotionBERT). Use `run_hybrid_pipeline.py` instead for better results.

```bash
python run_export.py [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input`, `-i` | **required** | Path to `video_outputs.json` |
| `--output`, `-o` | same as input dir | Output directory |
| `--height` | `1.75` | Subject height in meters |
| `--mass` | `70.0` | Subject mass in kg |
| `--fps` | from metadata | Override FPS |
| `--smooth` | `6.0` | Butterworth cutoff Hz (0 = disable) |
| `--skip-ik` | false | Skip OpenSim IK |
| `--skip-fbx` | false | Skip GLB/FBX export |
| `--person` | `0` | Person index |

---

## Output Files

After running both stages, you will have:

### From Stage 1 (run_inference.py)

| File | Description |
|------|-------------|
| `video_outputs.json` | Raw RTMW keypoints: 133 COCO-WholeBody 2D + 3D per frame per person |
| `inference_meta.json` | Video metadata: FPS, resolution, frame count, inference time |
| `frames/` | Extracted video frames (JPEG) |

### From Stage 2 (run_hybrid_pipeline.py)

| File | Description |
|------|-------------|
| `markers_*.trc` | 22 OpenSim markers in meters (tab-separated text) |
| `kinematics/*.osim` | Scaled OpenSim model with marker definitions |
| `kinematics/*_22markers.osim` | Model with extra eye + hand markers for Pass 2 IK |
| `kinematics/*.mot` | 40 joint angles in degrees (from inverse kinematics) |
| `kinematics/*_ik_setup*.xml` | IK solver configuration files |
| `*.glb` | Skeleton animation for universal viewers — Three.js, Unity, Unreal, web (quaternion-native) |
| `*.fbx` | Skeleton animation for Blender (optional) |

### Joint angles in the MOT file

The `.mot` file contains 40 degrees of freedom:

| Group | Joints |
|-------|--------|
| Pelvis (6 DOF) | tilt, list, rotation, tx, ty, tz |
| Right leg (7) | hip_flexion, hip_adduction, hip_rotation, knee_angle, ankle_angle, subtalar_angle, mtp_angle |
| Left leg (7) | same as right |
| Spine (3) | L5_S1_Flex_Ext, L5_S1_Lat_Bending, L5_S1_axial_rotation |
| Neck (3) | neck_flexion, neck_bending, neck_rotation |
| Right arm (7) | arm_flex, arm_add, arm_rot, elbow_flex, pro_sup, wrist_flex, wrist_dev |
| Left arm (7) | same as right |

---

## Viewing Results

### OpenSim GUI

1. Open the scaled model: `kinematics/*.osim`
2. Load motion data: `kinematics/*.mot`
3. Click Play to view the animation

### 3D Viewers (Three.js, Unity, Unreal, web)

Import the `.glb` file directly. GLB uses quaternions natively, so there are no Euler angle wrapping artifacts.

### Blender

1. Import the `.fbx` or `.glb` file directly
2. Or open `Import_OS4_Patreon_Aitor_Skely.blend` and load the `.mot` via the included script

### TRC inspection

The `.trc` file is plain text (tab-separated). Open it in any text editor or spreadsheet. Columns:

```
Frame#  Time  Nose_X  Nose_Y  Nose_Z  Neck_X  ...  RPinky_X  RPinky_Y  RPinky_Z
```

Coordinate system: OpenSim (X = forward, Y = up, Z = right). Units: meters.

---

## Troubleshooting

### "No 2D keypoints found"

Re-run `run_inference.py` with the latest version. Older inference outputs may not contain 2D keypoints needed by the hybrid pipeline.

### "Cannot find Pose2Sim/OpenSim environment"

The IK step needs the `Pose2Sim` conda environment at:
```
C:\ProgramData\anaconda3\envs\Pose2Sim\python.exe
```

If your Pose2Sim environment is at a different path, the pipeline will try `sys.executable` as a fallback (works if OpenSim is installed in the current env).

### "Blender not found"

GLB/FBX export requires Blender 5.0+ installed at:
```
C:\Program Files\Blender Foundation\Blender 5.0\blender.exe
```

Use `--skip-fbx` to skip GLB/FBX export if Blender is not installed. The TRC and MOT files will still be generated.

### CUDA out of memory

- Use `--device cpu` for both inference and hybrid pipeline
- Or reduce input video resolution before running

### High IK marker errors

Typical RMS marker errors for markerless pose estimation:
- Body markers: 30-50 mm (normal)
- Hand markers: 60-100 mm (normal, projected from 2D)
- If errors are consistently > 100 mm for body markers, check that `--height` matches the actual subject

### "mmcv" or "mmdet" import errors

The mmpose ecosystem requires compatible versions. Install in order:
```bash
pip install mmengine mmcv mmdet mmpose
```

On Windows, if `mmcv` CUDA ops fail, you may need to build from source or use CPU-only mode.

---

## Architecture

```
Video (.mp4)
    |
    v
Stage 1: run_inference.py
    RTMW inference (133 COCO-WholeBody 2D + 3D keypoints)
    |
    v
video_outputs.json
    |
    v
Stage 2: run_hybrid_pipeline.py
    |
    +-- Extract body-17 2D keypoints
    +-- COCO-17 to H36M conversion
    +-- MotionBERT 3D lifting (temporal, 243-frame window)
    +-- Skeleton normalization + bone length normalization
    +-- 6 Hz Butterworth smoothing
    +-- H36M to COCO-17 back-conversion
    +-- Project extra markers (eyes at nose depth, hands at wrist depth)
    +-- Coordinate transform (MotionBERT camera -> OpenSim world)
    +-- Center at first-frame pelvis + ground alignment
    +-- TRC export (22 markers)
    +-- Pose2Sim scaling + Pass 1 IK (14 body markers)
    +-- Pass 2 IK (22 markers + pelvis regularization)
    +-- Post-process MOT (zero hip_rotation)
    +-- GLB/FBX export via Blender (optional)
    |
    v
TRC + MOT + GLB + FBX
```

### Coordinate Systems

```
MotionBERT (H36M camera):       OpenSim:
    Y (down)                        Y (up)
    |                               |
    |                               |
    +--- X (right)                  +--- Z (right)
   /                               /
  Z (forward)                     X (forward)

Transform: X_osim = -Z_mb,  Y_osim = -Y_mb,  Z_osim = -X_mb
```

### 22 Markers (COCO_17 mode)

| # | Marker | Source | IK Weight |
|---|--------|--------|-----------|
| 1 | Nose | MotionBERT 3D | 0.1 |
| 2 | Neck | Shoulder midpoint | 0.2 |
| 3 | LShoulder | MotionBERT 3D | 2.0 |
| 4 | RShoulder | MotionBERT 3D | 2.0 |
| 5 | LElbow | MotionBERT 3D | 1.0 |
| 6 | RElbow | MotionBERT 3D | 1.0 |
| 7 | LWrist | MotionBERT 3D | 1.0 |
| 8 | RWrist | MotionBERT 3D | 1.0 |
| 9 | LHip | MotionBERT 3D | 2.0 |
| 10 | RHip | MotionBERT 3D | 2.0 |
| 11 | LKnee | MotionBERT 3D | 1.0 |
| 12 | RKnee | MotionBERT 3D | 1.0 |
| 13 | LAnkle | MotionBERT 3D | 1.0 |
| 14 | RAnkle | MotionBERT 3D | 1.0 |
| 15 | LEye | Projected (nose depth) | 0.05 |
| 16 | REye | Projected (nose depth) | 0.05 |
| 17 | LThumb | Projected (wrist depth) | 0.2 |
| 18 | LIndex | Projected (wrist depth) | 0.2 |
| 19 | LPinky | Projected (wrist depth) | 0.2 |
| 20 | RThumb | Projected (wrist depth) | 0.2 |
| 21 | RIndex | Projected (wrist depth) | 0.2 |
| 22 | RPinky | Projected (wrist depth) | 0.2 |

---

## License

All components are Apache 2.0 licensed:
- RTMDet, RTMW, mmpose (OpenMMLab)
- MotionBERT (Walter0807)
- Pose2Sim (perfanalytics)
- rtmpose3d (b-arac)
