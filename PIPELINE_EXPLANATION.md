# RTMToOpenSim Pipeline: Full Technical Explanation

This document explains every stage of the RTMToOpenSim pipeline, from raw video input to final OpenSim joint angles and FBX animation. It describes what each component does, why it exists, and what data flows between stages.

---

## Overview

The goal is to take an ordinary monocular video of a person and produce biomechanically valid joint angles that OpenSim can use. The pipeline has two main stages:

```
Stage 1 (run_inference.py):   Video  -->  2D + 3D keypoints (JSON)
Stage 2 (run_hybrid_pipeline.py):  JSON  -->  TRC markers  -->  MOT joint angles  -->  FBX animation
```

Stage 1 is slow (GPU inference on every frame). Stage 2 is fast (~18 seconds for 1136 frames) and can be re-run with different settings without repeating inference.

---

## Stage 1: Video to Keypoints (`run_inference.py`)

### 1.1 Frame Extraction

The input video (e.g., 1920x1080 @ 30 FPS) is decoded frame-by-frame using OpenCV. Each frame is saved as a JPEG image in an output directory. The target FPS can be set (default 30); if the video has a different FPS, frames are sub-sampled accordingly.

### 1.2 Person Detection (RTMDet)

Each frame is fed to **RTMDet-m**, a real-time object detector from the OpenMMLab ecosystem. RTMDet detects all people in the frame and returns bounding boxes. The model is:

- **RTMDet-Medium** (`rtmdet_m_8xb32-100e_coco-obj365-person`)
- Trained on COCO + Objects365, person class only
- Checkpoint: ~99 MB, auto-downloaded on first run
- License: Apache 2.0

RTMDet is used instead of YOLOv8 to avoid GPL licensing. It provides the bounding box that crops the image for the pose estimator.

### 1.3 2D and 3D Pose Estimation (RTMW)

The cropped person image is fed to **RTMW3D-Large**, a whole-body 3D pose estimation model from the RTMPose family. It simultaneously predicts:

- **133 2D keypoints** in pixel coordinates (x, y per joint)
- **133 3D keypoints** in meters (x, y, z per joint, root-relative)
- **133 confidence scores** (0 to 1 per joint)

The 133 COCO-WholeBody keypoints are organized as:

| Range | Count | Body Part |
|-------|-------|-----------|
| 0-16 | 17 | Body (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles) |
| 17-22 | 6 | Feet (big toe, small toe, heel, left and right) |
| 23-90 | 68 | Face (facial landmarks) |
| 91-111 | 21 | Left hand (wrist, each finger: CMC/MCP/IP/DIP/tip) |
| 112-132 | 21 | Right hand (same structure) |

The model is:

- **RTMW3D-Large** (`rtmw3d-l_8xb64_cocktail14-384x288`)
- Input resolution: 384x288 pixels
- Trained on the "cocktail14" dataset (14 combined datasets)
- Checkpoint: ~231 MB, auto-downloaded
- License: Apache 2.0

**Important limitation**: RTMW3D's direct 3D output has unreliable depth. The Z-axis (depth) is effectively constant or noisy across frames. This is why the pipeline uses MotionBERT for 3D lifting in Stage 2, using only the 2D keypoints from RTMW.

### 1.4 Output

Stage 1 produces two files:

- **`video_outputs.json`**: Array of per-frame results. Each frame contains `keypoints_2d` (133x2), `keypoints_3d` (133x3), `scores` (133), and `bbox` (4) per detected person.
- **`inference_meta.json`**: Video metadata (FPS, resolution, frame count, inference time).

---

## Stage 2: Keypoints to Motion Data (`run_hybrid_pipeline.py`)

This is where the core processing happens. The pipeline is called "hybrid" because it combines RTMW's high-quality 2D keypoints with MotionBERT's temporal 3D lifting, rather than using RTMW's direct 3D output.

### 2.1 Load Data

The pipeline reads `video_outputs.json` and extracts:
- `kpts_2d`: (T, 133, 2) — all 133 whole-body 2D keypoints in pixel coordinates
- `scores`: (T, 133) — per-joint confidence scores
- Video metadata (FPS, image dimensions) from `inference_meta.json`

### 2.2 Extract Body-17 Keypoints

Only the first 17 keypoints (body joints) are used for 3D lifting. These are the standard COCO body keypoints:

```
0: Nose          5: LShoulder     10: RWrist      15: LAnkle
1: LEye          6: RShoulder     11: LHip        16: RAnkle
2: REye          7: LElbow        12: RHip
3: LEar          8: RElbow        13: LKnee
4: REar          9: LWrist        14: RKnee
```

The remaining 116 keypoints (feet, face, hands) are NOT used for 3D lifting — they are used later for marker projection.

### 2.3 COCO-17 to H36M-17 Conversion

MotionBERT expects keypoints in the **Human3.6M (H36M) 17-joint format**, which is different from COCO-17. The conversion (`coco17_to_h36m`) maps joints and derives virtual joints:

| H36M Joint | Source |
|------------|--------|
| 0: Hip (pelvis center) | Midpoint of LHip + RHip |
| 1: RHip | COCO RHip |
| 2: RKnee | COCO RKnee |
| 3: RFoot | COCO RAnkle |
| 4: LHip | COCO LHip |
| 5: LKnee | COCO LKnee |
| 6: LFoot | COCO LAnkle |
| 7: Spine | Interpolated (50% between hip center and thorax) |
| 8: Thorax | Interpolated (75% from hip center toward shoulder center) |
| 9: Neck/Nose | COCO Nose |
| 10: Head | Mean of LEye, REye, LEar, REar |
| 11: LShoulder | COCO LShoulder |
| 12: LElbow | COCO LElbow |
| 13: LWrist | COCO LWrist |
| 14: RShoulder | COCO RShoulder |
| 15: RElbow | COCO RElbow |
| 16: RWrist | COCO RWrist |

### 2.4 MotionBERT 3D Lifting

This is the core of the pipeline. **MotionBERT** is a transformer-based model that takes a sequence of 2D poses and predicts 3D joint positions with proper depth.

**Model architecture:**
- **DSTformer** (Dual-Stream Spatial-Temporal Transformer)
- 16 million parameters
- Trained on Human3.6M with global (camera-frame) 3D prediction
- Temporal window: 243 frames (~8 seconds at 30 FPS)
- Input: normalized 2D keypoints + confidence scores
- Output: 3D keypoints in meters, in the H36M camera coordinate system

**How it works:**

1. **2D Normalization**: The 2D pixel coordinates are normalized by centering at the image center and dividing by `min(width, height) / 2`. This preserves the aspect ratio — for a 1920x1080 video, X ranges from -1.78 to +1.78 and Y from -1 to +1. This is critical: per-axis normalization would distort the skeleton proportions and break depth estimation.

2. **Input preparation**: The normalized (x, y) is stacked with the confidence score to form a (T, 17, 3) input tensor: [x_norm, y_norm, confidence].

3. **Sliding window inference**: The sequence is processed in overlapping windows of 243 frames with a stride of 81 frames. Each window produces a 3D prediction. Where windows overlap, predictions are averaged, which acts as a natural smoothing mechanism.

4. **Flip augmentation**: Each clip is also run with horizontal flipping (X coordinates negated, left/right joints swapped). The two predictions are averaged. This reduces left-right bias and improves accuracy.

5. **Output**: (T, 17, 3) 3D keypoints in meters, in the **H36M camera coordinate system**:
   - X = right
   - Y = down (gravity direction)
   - Z = forward (toward the camera)

**Why MotionBERT instead of RTMW3D?**

RTMW3D's direct 3D output has a known depth limitation — the Z-axis is nearly constant across frames, meaning the person appears flat (no depth variation between left and right sides of the body). MotionBERT, trained specifically on H36M with temporal context, produces meaningful depth that tracks which side of the body is closer to the camera. This is essential for deriving pelvis rotation (facing direction) during inverse kinematics.

### 2.5 Skeleton Scale Normalization

MotionBERT's "global" model outputs camera-frame 3D where the skeleton size changes with distance from the camera — a person walking away gets smaller in 3D. To produce consistent biomechanical data, we normalize the skeleton to a consistent size.

**Per-frame scale normalization**: For each frame, the trunk height (hip center to nose distance) is computed. All frames are scaled so the trunk height matches the median across the sequence. The hip center is used as the pivot point so that global translation is preserved.

### 2.6 Bone Length Normalization

Even after scale normalization, individual bone lengths can vary frame-to-frame due to estimation noise. This step enforces consistent bone lengths:

1. For each bone in the H36M skeleton tree (16 bones total), the median length across all frames is computed.
2. For each frame, each bone is rescaled to match its median length.
3. Rescaling propagates downstream — if the upper arm is adjusted, the elbow, forearm, and wrist all shift together to maintain the kinematic chain.

This produces a rigid skeleton with fixed segment lengths, which is what OpenSim expects.

### 2.7 Butterworth Low-Pass Smoothing

A 4th-order zero-phase Butterworth filter is applied uniformly to all joints and all axes (X, Y, Z) at the same cutoff frequency (default: 6 Hz). This removes high-frequency jitter from the pose estimation while preserving natural movement.

**Why uniform smoothing?** Earlier versions used different cutoffs for depth vs. lateral axes, but this caused artifacts. The simple uniform approach (same as the Sam3DBodyToOpenSim reference project) preserves facing direction and global translation naturally.

**Why 6 Hz?** Human voluntary movement rarely exceeds 6 Hz. Walking gait has primary frequencies of 1-2 Hz. A 6 Hz cutoff removes estimation noise while keeping all movement signal intact.

### 2.8 H36M-17 to COCO-17 Back-Conversion

After all processing in H36M space, the 3D keypoints are converted back to COCO-17 ordering (`h36m_to_coco17`). This reverses the mapping from step 2.3. Eyes and ears (not present in H36M) are approximated from the Head joint position — they will be overridden by the projected markers in the next step.

### 2.9 Extra Marker Projection (2D to 3D)

The 14 body markers from MotionBERT give good body tracking but lack head orientation (only Nose) and hand information (only wrists). To constrain head rotation and forearm rotation in IK, 8 additional markers are projected from 2D to 3D:

| Marker | 2D Source (COCO-WB index) | Depth Anchor | Purpose |
|--------|--------------------------|--------------|---------|
| LEye | 1 (left eye) | Nose depth | Head rotation |
| REye | 2 (right eye) | Nose depth | Head rotation |
| LThumb | 95 (left thumb tip) | LWrist depth | Forearm rotation |
| LIndex | 99 (left index tip) | LWrist depth | Forearm rotation |
| LPinky | 111 (left pinky tip) | LWrist depth | Forearm rotation |
| RThumb | 116 (right thumb tip) | RWrist depth | Forearm rotation |
| RIndex | 120 (right index tip) | RWrist depth | Forearm rotation |
| RPinky | 132 (right pinky tip) | RWrist depth | Forearm rotation |

**Projection method:**

For eyes, the pixel-to-meter scale is computed from the shoulder width:
```
scale = ||LShoulder_3D - RShoulder_3D||_XY  /  ||LShoulder_2D - RShoulder_2D||
```
Then the eye 3D position is:
```
eye_3D.X = nose_3D.X + (eye_2D.x - nose_2D.x) * scale
eye_3D.Y = nose_3D.Y + (eye_2D.y - nose_2D.y) * scale
eye_3D.Z = nose_3D.Z    (anchored at nose depth)
```

For hands, the same approach uses the forearm length (elbow-to-wrist) as the scale reference, and the wrist depth as the anchor.

These projected markers have no independent depth information — they live on the depth plane of their anchor (nose or wrist). But they provide lateral and vertical offset information that constrains head and forearm rotation in IK.

### 2.10 Coordinate Transform (MotionBERT to OpenSim)

The 3D markers are now in MotionBERT's H36M camera coordinate system and need to be converted to OpenSim's biomechanical coordinate system:

```
MotionBERT (H36M camera):       OpenSim (biomechanical):
    Y (down)                        Y (up)
    |                               |
    +--- X (right)                  +--- Z (right)
   /                               /
  Z (forward, toward camera)      X (forward, anterior)
```

The person faces the camera, meaning the person's forward direction is **-Z** in camera space (toward the camera). The rotation matrix that maps camera coords to OpenSim coords is:

```
X_opensim = -Z_motionbert    (person forward)
Y_opensim = -Y_motionbert    (up = opposite of gravity)
Z_opensim = -X_motionbert    (person right)
```

As a matrix:
```
        [ 0   0  -1 ]
R =     [ 0  -1   0 ]
        [-1   0   0 ]
```

All axes are negated because the camera sees the person from the front — the person's right is the camera's left, the person's up is opposite to Y-down, and the person's forward is away from the camera.

### 2.11 Scale to Subject Height

The MotionBERT output is in arbitrary metric units (roughly correct scale but not exact). The skeleton is scaled to match the actual subject height:

1. For each frame, measure the nose-to-ankle-midpoint distance
2. Average across all frames
3. Multiply by 1.1 to estimate full body height (head crown is above the nose)
4. Compute scale factor: `subject_height / estimated_full_height`
5. Multiply all marker positions by this scale factor

### 2.12 First-Frame Pelvis Centering

The skeleton is centered in the XZ plane (horizontal) at the **first frame's** pelvis position. This is important — it preserves global translation throughout the sequence. If the person walks 3 meters during the video, the pelvis will move 3 meters in the output. Only the starting position is zeroed.

### 2.13 Ground Alignment

The Y axis (vertical in OpenSim) is aligned so that the feet touch the ground (Y=0):

1. For each frame, find the minimum ankle height (left or right)
2. Smooth this ground reference with a 15-frame moving average to prevent vertical jitter
3. Subtract the smoothed ground height from all markers per frame

### 2.14 Marker Assembly

The 22 markers are assembled from two sources:

**14 body markers** (from MotionBERT 3D, through the coordinate transform):

| # | Marker | Source (COCO-17 index) |
|---|--------|----------------------|
| 1 | Nose | 0 |
| 2 | Neck | Midpoint of LShoulder (5) + RShoulder (6) |
| 3 | LShoulder | 5 |
| 4 | RShoulder | 6 |
| 5 | LElbow | 7 |
| 6 | RElbow | 8 |
| 7 | LWrist | 9 |
| 8 | RWrist | 10 |
| 9 | LHip | 11 |
| 10 | RHip | 12 |
| 11 | LKnee | 13 |
| 12 | RKnee | 14 |
| 13 | LAnkle | 15 |
| 14 | RAnkle | 16 |

**8 projected markers** (from 2D projection at body depth):

| # | Marker |
|---|--------|
| 15 | LEye |
| 16 | REye |
| 17 | LThumb |
| 18 | LIndex |
| 19 | LPinky |
| 20 | RThumb |
| 21 | RIndex |
| 22 | RPinky |

### 2.15 TRC Export

The 22 markers are written to a **TRC (Track Row Column)** file, the standard marker trajectory format for OpenSim. The TRC file is plain text with tab-separated values:

```
Header lines (file version, marker count, FPS, units)
Column names: Frame#  Time  Nose  Neck  LShoulder  ...  RPinky
              (blank)       X1 Y1 Z1  X2 Y2 Z2    ...
Data rows:    1  0.000  0.012  1.542  -0.003  ...
```

Units are meters. Coordinate system is OpenSim (X=forward, Y=up, Z=right).

---

## Stage 2 continued: Inverse Kinematics

### 2.16 Pose2Sim Scaling + Pass 1 IK

The TRC file is passed to **Pose2Sim**, which handles OpenSim model scaling and inverse kinematics. This runs as a **subprocess** in a separate Python environment (`Pose2Sim` conda env) because OpenSim is not compatible with the mmpose environment.

**What Pose2Sim does:**

1. **Model selection**: Loads `Model_Pose2Sim_simple.osim`, a 40-DOF whole-body musculoskeletal model with 22 body segments (pelvis, torso, head, upper/lower arms, hands, upper/lower legs, feet).

2. **Scaling**: Adjusts the generic model's segment dimensions to match the subject. It computes scale factors from the marker distances in the TRC file:
   - Measures distances between marker pairs (e.g., LShoulder-RShoulder for torso width)
   - Trims outlier frames (fastest 0.1%, frames near zero speed)
   - Computes median marker distances
   - Scales each body segment independently

3. **Pass 1 IK**: Runs OpenSim's InverseKinematicsTool on the scaled model with the original 14 COCO_17 marker tasks. Marker weights are set by Pose2Sim:
   - Shoulders, Hips: weight 2.0 (proximal joints, highest confidence)
   - Elbows, Knees, Wrists, Ankles: weight 1.0
   - Neck: weight 0.2
   - Nose: weight 0.1

   The IK solver minimizes the weighted sum of squared distances between the TRC markers and the model's marker positions by adjusting the 40 joint angles per frame.

   The result is a scaled `.osim` model and a `.mot` file with joint angles.

### 2.17 Pass 2 IK (22 Markers + Regularization)

Pass 1 uses only 14 body markers. Pass 2 re-runs IK with the full 22 markers and adds regularization for joints that are underconstrained by marker data alone.

**Step 1: Add markers to the model**

The 8 extra markers (LEye, REye, LThumb, LIndex, LPinky, RThumb, RIndex, RPinky) are added to the scaled `.osim` model by inserting XML `<Marker>` elements into the model's MarkerSet. Each marker is attached to its anatomical body:
- Eyes on `head`
- Left hand markers on `hand_l`
- Right hand markers on `hand_r`

The marker positions within each body segment come from Pose2Sim's COCO_133 marker definitions (anatomically correct offsets).

**Step 2: Build 22-marker IK setup**

A custom IK setup XML is created with 22 marker tasks and coordinate regularization tasks:

| Marker | Weight | Rationale |
|--------|--------|-----------|
| Shoulders, Hips | 2.0 | High-confidence anchor joints |
| Elbows, Knees, Wrists, Ankles | 1.0 | Standard limb joints |
| Neck | 0.2 | Derived (shoulder midpoint) |
| Nose | 0.1 | Head tracking, but noisy |
| LEye, REye | 0.05 | Projected from 2D, low confidence |
| Hand markers (6) | 0.2 | Projected from 2D, moderate confidence |

Coordinate regularization tasks pull specific joint angles toward their default (zero) value:

| Coordinate | Weight | Purpose |
|------------|--------|---------|
| L5_S1_Flex_Ext | 0.1 | Prevent spine flex/extension from drifting |
| pelvis_tilt | 0.05 | Prevent forward/backward tilt oscillation |
| pelvis_list | 0.005 | Light lateral tilt constraint |
| hip_adduction_r/l | 0.005 | Prevent hip abduction noise |

Note: **pelvis_rotation is NOT regularized**. The depth differences between left and right markers naturally drive pelvis rotation through IK. Regularizing it would suppress the facing direction signal.

**Step 3: Run IK**

OpenSim's InverseKinematicsTool runs with the modified model, 22-marker TRC, and the custom IK setup. The solver finds joint angles that best fit all 22 markers while respecting the regularization constraints.

### 2.18 Post-Processing: Zero Hip Rotation

The hip rotation DOF (internal/external rotation of the thigh) cannot be constrained by the COCO markers we have — there are no markers on the thigh or shin that differentiate internal from external rotation. As a result, the IK solver produces random noise in `hip_rotation_r` and `hip_rotation_l` (up to 45 degrees of random oscillation).

Rather than adding regularization (which was found to cause side effects in knee and ankle angles), the hip rotation values are simply set to zero in the final MOT file. This is biomechanically reasonable for most activities (walking, standing) where hip rotation is small.

### 2.19 Output: MOT File

The final `.mot` file contains 40 joint angles per frame:

**Pelvis** (6 DOF):
- `pelvis_tilt`: forward/backward tilt (degrees)
- `pelvis_list`: lateral tilt (degrees)
- `pelvis_rotation`: facing direction / yaw (degrees)
- `pelvis_tx`, `pelvis_ty`, `pelvis_tz`: global translation (meters)

**Legs** (7 DOF each, left and right):
- `hip_flexion`: forward/backward leg swing
- `hip_adduction`: lateral leg movement
- `hip_rotation`: internal/external thigh rotation (zeroed)
- `knee_angle`: knee bend
- `ankle_angle`: dorsiflexion/plantarflexion
- `subtalar_angle`: foot inversion/eversion
- `mtp_angle`: toe bend

**Spine** (3 DOF):
- `L5_S1_Flex_Ext`: spine forward/backward bend
- `L5_S1_Lat_Bending`: spine lateral bend
- `L5_S1_axial_rotation`: spine twist

**Neck** (3 DOF):
- `neck_flexion`: head nod
- `neck_bending`: head lateral tilt
- `neck_rotation`: head turn

**Arms** (7 DOF each, left and right):
- `arm_flex`: shoulder forward/backward
- `arm_add`: shoulder lateral
- `arm_rot`: shoulder internal/external rotation
- `elbow_flex`: elbow bend
- `pro_sup`: forearm pronation/supination
- `wrist_flex`: wrist flex/extension
- `wrist_dev`: wrist radial/ulnar deviation

---

## Stage 2 final: FBX Export (Optional)

### 2.20 Blender FBX Export

If Blender is installed, the MOT joint angles are applied to a pre-rigged skeleton template and exported as an FBX file. This runs as a subprocess:

1. Blender opens in background mode with the skeleton template (`Import_OS4_Patreon_Aitor_Skely.blend`)
2. A Python script (`scripts/export_fbx_skely.py`) reads the MOT file, maps joint angles to the Blender armature, and keyframes each frame
3. The animation is exported as `.fbx`

The FBX file can be imported into any 3D application (Unity, Unreal Engine, Blender, Maya, etc.) for visualization or game development.

---

## Data Flow Summary

```
Video (1920x1080, 30 FPS)
    |
    |  [RTMDet-m: person detection]
    |  [RTMW3D-L: 2D + 3D pose estimation]
    v
video_outputs.json
    |  133 keypoints x 2D pixel coords
    |  133 keypoints x 3D coords (unused for hybrid)
    |  133 confidence scores
    v
Extract body-17 2D keypoints (indices 0-16)
    |
    |  [COCO-17 -> H36M-17 format conversion]
    v
H36M 2D keypoints (T, 17, 2)
    |
    |  [MotionBERT DSTformer: 2D-to-3D temporal lifting]
    |  [243-frame sliding window, flip augmentation]
    v
H36M 3D keypoints (T, 17, 3) in camera coords
    |
    |  [Skeleton scale normalization]
    |  [Bone length normalization to median]
    |  [6 Hz Butterworth smoothing]
    v
Smoothed H36M 3D keypoints
    |
    |  [H36M-17 -> COCO-17 back-conversion]
    v
COCO-17 3D body (T, 17, 3)
    |
    |  [Project eyes from 2D at nose depth]
    |  [Project hand tips from 2D at wrist depth]
    v
8 extra markers (T, 8, 3) in camera coords
    |
    |  [MB->OpenSim coordinate rotation]
    |  [Scale to subject height]
    |  [Center at first-frame pelvis]
    |  [Ground alignment via ankle heights]
    v
22 OpenSim markers (T, 22, 3) in meters
    |
    |  [TRC file export]
    v
markers.trc
    |
    |  [Pose2Sim: model scaling from marker distances]
    |  [Pass 1 IK: 14 body markers, L5_S1 regularization]
    v
Scaled .osim model + Pass 1 .mot
    |
    |  [Add 8 extra markers to .osim model]
    |  [Pass 2 IK: 22 markers + pelvis regularization]
    |  [Post-process: zero hip_rotation]
    v
Final .mot (40 joint angles per frame)
    |
    |  [Blender: apply to skeleton, export]
    v
.fbx animation (optional)
```

---

## Why This Architecture?

### Why not use RTMW3D's 3D directly?

RTMW3D produces 133 keypoints in 3D, which seems ideal. But its depth axis (Z) is unreliable — it outputs nearly constant depth across frames, making the person appear flat. The person's facing direction and lateral depth (which shoulder is closer) are lost. This breaks pelvis rotation estimation in IK.

### Why MotionBERT?

MotionBERT was trained specifically on Human3.6M motion capture data with a temporal transformer architecture. Its 243-frame temporal window allows it to reason about depth from motion parallax — if a person is turning, the shoulder moving faster in 2D is the one coming toward the camera. This produces meaningful depth that tracks with the person's actual orientation.

### Why two IK passes?

Pass 1 uses Pose2Sim's standard COCO_17 pipeline for model scaling. Pose2Sim selects setup files based on the `pose_model` name, so we can't easily inject custom markers into its workflow. Pass 2 re-runs IK on the already-scaled model with all 22 markers and custom weights. This cleanly separates "scaling with known markers" from "IK with extra markers."

### Why project eyes and hands instead of using MotionBERT for them?

MotionBERT only lifts the 17 H36M body joints. Eyes are not in H36M, and hand joints would require a separate hand-specific model. Projection from 2D is simple and sufficient — the eyes only need to constrain head rotation (not depth), and the hand fingertips only need to constrain forearm pronation/supination. For these purposes, 2D position on the correct depth plane is enough.

### Why zero hip rotation instead of regularizing it?

Hip rotation (internal/external thigh rotation) needs markers on the thigh or shin to be observable. With only hip, knee, and ankle markers, the IK solver has no information about thigh twist and produces random noise. Regularization (pulling toward zero) was tried but caused side effects — the solver compensated by distorting knee and ankle angles. Zeroing hip rotation in post-processing is cleaner and doesn't affect any other joints.

---

## Technologies Used

| Component | Technology | License |
|-----------|-----------|---------|
| Person detection | RTMDet-m (OpenMMLab) | Apache 2.0 |
| 2D + 3D pose estimation | RTMW3D-Large (OpenMMLab) | Apache 2.0 |
| 2D-to-3D lifting | MotionBERT DSTformer | Apache 2.0 |
| Model scaling + IK | Pose2Sim + OpenSim 4.5.2 | Apache 2.0 |
| FBX export | Blender 5.0 (optional) | GPL (standalone, not linked) |
| rtmpose3d wrapper | b-arac/rtmpose3d | Apache 2.0 |
