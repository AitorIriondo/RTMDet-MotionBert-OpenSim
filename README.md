# RTMDet-MotionBert-OpenSim

Convert monocular video to OpenSim motion data using RTMDet + RTMW for 2D pose estimation and MotionBERT for 3D lifting.

## Overview

This pipeline uses [RTMW](https://github.com/open-mmlab/mmpose) (OpenMMLab) for real-time whole-body 2D pose estimation and [MotionBERT](https://github.com/Walter0807/MotionBERT) for temporal 2D-to-3D lifting, producing OpenSim-compatible motion files for biomechanical analysis.

**Pipeline Flow:**
```
Video → RTMDet → RTMW 2D (133 keypoints) → MotionBERT 3D Lifting → 22 OpenSim Markers → IK → Joint Angles (.mot)
              ↓                                     ↓                                             ↓
         Person bbox                          H36M camera 3D                              Blender → GLB Animation
```

## Features

- **133 COCO-WholeBody keypoints** including body, hands, feet, and face
- **RTMDet-m**: Real-time person detection (Apache 2.0, no GPL)
- **MotionBERT DSTformer**: Temporal transformer for accurate depth estimation (243-frame window)
- **22 OpenSim markers**: 14 body (MotionBERT 3D) + 8 projected (eyes + hand fingertips)
- **Global translation tracking** from MotionBERT camera-frame output
- **Per-frame ground alignment**: Feet always touch the floor
- **Eye markers for head rotation**: Better neck flexion/rotation tracking
- **Hand markers for forearm rotation**: Better pronation/supination tracking
- **Butterworth smoothing**: Configurable low-pass filter to reduce jitter (default 6 Hz)
- **OpenSim IK** with 40 DOF using Pose2Sim model (two-pass with pelvis regularization)
- **GLB export** via Blender with rigged skeleton template (quaternion-native, universal viewer compatibility)
- **Two-stage workflow**: Separate inference (slow) from export (fast) for rapid iteration

## Performance

Tested on NVIDIA RTX GPU with 1136 frames (37.8 sec video, 1920x1080):

| Stage | Time | Speed |
|-------|------|-------|
| RTMW Inference | ~89 sec | ~12.7 frames/sec |
| Hybrid Export (MotionBERT + IK + GLB) | ~17 sec | ~67 frames/sec |
| OpenSim IK (Pass 1 + Pass 2) | ~6 sec | ~189 frames/sec |

## Quick Start

```bash
cd C:\RTMDetMotionBertOpenSim
conda activate mmpose

# Single command (runs both stages)
python run_pipeline.py --input videos\aitor_garden_walk.mp4 --height 1.69 --correct-lean
```

## Usage

### Single Command

Runs inference + export in one go:
```bash
python run_pipeline.py --input videos\aitor_garden_walk.mp4 --height 1.69 --correct-lean
```

### Two-Stage Workflow (for iterating on export settings)

**Stage 1: Inference** (slow, run once)
```bash
python run_inference.py --input videos\aitor_garden_walk.mp4
```

**Stage 2: Hybrid Export** (fast, iterate on settings)

Pass the same video path — the pipeline auto-discovers the latest inference output.
```bash
python run_hybrid_pipeline.py --input videos\aitor_garden_walk.mp4 --height 1.69 --correct-lean
```

You can also pass the inference output directly:
```bash
python run_hybrid_pipeline.py --input output_*_aitor_garden_walk/video_outputs.json --height 1.69
```

### CPU Mode

```bash
python run_inference.py --input videos\aitor_garden_walk.mp4 --device cpu
python run_hybrid_pipeline.py --input videos\aitor_garden_walk.mp4 --height 1.69 --device cpu
```

## Arguments Reference

### run_pipeline.py (combined)

All arguments from both stages. Shared args (`--device`, `--person`) apply to both.

| Argument | Description | Default |
|----------|-------------|---------|
| `--input, -i` | Input video file | Required |
| `--output, -o` | Output directory | Auto |
| `--device` | Compute device (cuda:0 or cpu) | cuda:0 |
| `--person` | Person index | 0 |
| `--fps` | Target FPS for frame extraction | 30.0 |
| `--model` | Model name override | rtmpose3d |
| `--height` | Subject height (meters) | 1.75 |
| `--mass` | Subject mass (kg) | 70.0 |
| `--smooth` | Smoothing cutoff Hz (0 to disable) | 6.0 |
| `--skip-ik` | Skip OpenSim IK | false |
| `--skip-glb` | Skip GLB export | false |
| `--pose-model` | COCO_17 (22 markers) or COCO_133 (27 markers) | COCO_17 |
| `--focal-length` | Camera focal length in pixels | Auto |
| `--correct-lean` | Ground-plane lean correction | false |
| `--single-level` | Per-frame strict grounding | false |

### run_inference.py

| Argument | Description | Default |
|----------|-------------|---------|
| `--input, -i` | Input video file | Required |
| `--output, -o` | Output directory | Auto |
| `--fps` | Target FPS for frame extraction | 30.0 |
| `--device` | Compute device (cuda:0 or cpu) | cuda:0 |
| `--person` | Person index (0 = first detected) | 0 |
| `--model` | Model name override | rtmpose3d |

### run_hybrid_pipeline.py

| Argument | Description | Default |
|----------|-------------|---------|
| `--input, -i` | Input video (.mp4) or video_outputs.json | Required |
| `--height` | Subject height (meters) | 1.75 |
| `--mass` | Subject mass (kg) | 70.0 |
| `--output, -o` | Output directory | Same as input |
| `--smooth` | Smoothing cutoff frequency in Hz (0 to disable) | 6.0 |
| `--device` | Device for MotionBERT (cuda:0 or cpu) | cuda:0 |
| `--pose-model` | IK marker set: **COCO_17** (22 markers) or COCO_133 (27 markers) | COCO_17 |
| `--skip-ik` | Skip OpenSim inverse kinematics | false |
| `--skip-glb` | Skip GLB export | false |
| `--correct-lean` | Ground-plane lean correction from foot contacts | false |
| `--single-level` | Per-frame strict grounding (lowest foot = 0 every frame) | false |
| `--person` | Person index | 0 |
| `--fps` | Override FPS (default: from metadata) | Auto |

## Output Files

```
output_dir/
├── video_outputs.json                # RTMW outputs (2D + 3D keypoints, scores)
├── inference_meta.json               # Video metadata (FPS, dimensions)
├── markers_videoname.trc             # OpenSim marker trajectories (22 markers, meters)
├── kinematics/
│   ├── markers_videoname_22markers.osim  # Scaled OpenSim model (22 markers)
│   ├── markers_videoname.mot         # Joint angles (40 DOF)
│   └── *_ik_setup*.xml              # IK solver configuration
└── markers_videoname.glb             # Animated skeleton (quaternion-native, universal viewers)
```

## Pipeline Stages

1. **Frame Reading**: Video frames read directly into memory at target FPS
2. **Person Detection**: RTMDet-m bounding box detection
3. **2D Pose Estimation**: RTMW3D-Large → 133 COCO-WholeBody 2D keypoints
4. **COCO-17 to H36M Conversion**: Reformat body joints for MotionBERT
5. **3D Lifting**: MotionBERT DSTformer → 17 body joints in 3D (camera coordinates)
6. **Skeleton Normalization**: Per-frame scale + bone length normalization
7. **Smoothing**: 6 Hz Butterworth low-pass filter (all joints, all axes)
8. **Marker Projection**: Project eyes + hand fingertips from 2D at body depth anchors
9. **Coordinate Transform**: H36M camera → OpenSim world coordinates
10. **TRC Export**: 22 markers in OpenSim format
11. **OpenSim IK**: Two-pass inverse kinematics → 40 DOF joint angles
12. **GLB Export**: Blender animated skeleton (quaternion-native)

## Documentation

- [HOW_TO_RUN.md](HOW_TO_RUN.md) - Installation and usage guide
- [PIPELINE_EXPLANATION.md](PIPELINE_EXPLANATION.md) - Full technical reference

## Requirements

- Windows 10/11
- Python 3.10
- CUDA-capable GPU (4GB+ VRAM recommended)
- PyTorch 2.0+
- MMPose ecosystem (mmcv, mmdet, mmpose)
- rtmpose3d
- MotionBERT (bundled, checkpoint downloaded separately)
- OpenSim 4.5+ (via Pose2Sim, in separate conda environment)
- Blender 5.0+ (optional, for GLB export)

See [requirements.txt](requirements.txt) for Python packages.

## Project Structure

```
RTMDet-MotionBert-OpenSim/
├── config/                    # Pipeline configuration
├── models/MotionBERT/         # MotionBERT model code (DSTformer)
├── src/                       # Source modules
├── utils/                     # Utility functions
├── scripts/                   # Blender export script
├── run_pipeline.py            # Full pipeline (both stages in one command)
├── run_inference.py           # Stage 1: RTMW inference
├── run_hybrid_pipeline.py     # Stage 2: Hybrid export (recommended)
├── run_export.py              # Stage 2: Legacy export (RTMW3D direct)
└── requirements.txt           # Python dependencies
```

## License

Apache 2.0. All components are Apache 2.0 licensed.

## Acknowledgments

- **RTMW / RTMDet**: OpenMMLab - https://github.com/open-mmlab/mmpose
- **MotionBERT**: Walter0807 - https://github.com/Walter0807/MotionBERT
- **Pose2Sim**: perfanalytics - https://github.com/perfanalytics/pose2sim
- **OpenSim**: Stanford - https://opensim.stanford.edu/
- **rtmpose3d**: b-arac - https://github.com/b-arac/rtmpose3d
