"""
Merge MotionBERT body 3D, projected hands, projected head, and projected feet
into a full COCO-WholeBody 133-keypoint array.

All inputs must be in the same coordinate system (H36M camera: X=right, Y=down, Z=forward).
"""

import numpy as np

# COCO-WholeBody ranges
BODY_RANGE = slice(0, 17)
FEET_RANGE = slice(17, 23)
FACE_RANGE = slice(23, 91)
LEFT_HAND_RANGE = slice(91, 112)
RIGHT_HAND_RANGE = slice(112, 133)

# Body anchor indices for feet projection
COCO_LEFT_ANKLE = 15
COCO_RIGHT_ANKLE = 16
COCO_LEFT_KNEE = 13
COCO_RIGHT_KNEE = 14

# Foot keypoint indices within COCO-WholeBody
FEET_INDICES = [17, 18, 19, 20, 21, 22]  # LBigToe, LSmallToe, LHeel, RBigToe, RSmallToe, RHeel
LEFT_FOOT_INDICES = [17, 18, 19]   # LBigToe, LSmallToe, LHeel
RIGHT_FOOT_INDICES = [20, 21, 22]  # RBigToe, RSmallToe, RHeel


def merge_hybrid_keypoints(
    body_3d_coco17: np.ndarray,
    hands_3d: np.ndarray,
    head_3d: np.ndarray,
    keypoints_2d_133: np.ndarray,
    min_ankle_2d: float = 10.0,
) -> np.ndarray:
    """
    Merge all 3D components into a COCO-WholeBody 133-keypoint array.

    Args:
        body_3d_coco17: (T, 17, 3) body 3D from MotionBERT (COCO-17 order)
        hands_3d: (T, 42, 3) hands from hand_projector (21 left + 21 right)
        head_3d: (T, 4, 3) from head_projector [LEye, REye, LEar, REar]
        keypoints_2d_133: (T, 133, 2) RTMW 2D keypoints for foot projection
        min_ankle_2d: Min ankle-to-foot 2D distance for scale validity

    Returns:
        keypoints_133: (T, 133, 3) full COCO-WholeBody 3D keypoints
    """
    T = body_3d_coco17.shape[0]
    result = np.zeros((T, 133, 3), dtype=np.float32)

    # 1. Body (0-16): directly from MotionBERT
    result[:, BODY_RANGE] = body_3d_coco17

    # 2. Head details: override eyes and ears with projected values
    # Nose (index 0) already set from body_3d_coco17
    result[:, 1] = head_3d[:, 0]  # LEye
    result[:, 2] = head_3d[:, 1]  # REye
    result[:, 3] = head_3d[:, 2]  # LEar
    result[:, 4] = head_3d[:, 3]  # REar

    # 3. Feet (17-22): project from 2D at ankle depth
    _project_feet(result, body_3d_coco17, keypoints_2d_133, min_ankle_2d)

    # 4. Face (23-90): leave as zeros (not used for OpenSim markers)

    # 5. Hands (91-132): from hand projector
    result[:, LEFT_HAND_RANGE] = hands_3d[:, :21]
    result[:, RIGHT_HAND_RANGE] = hands_3d[:, 21:]

    return result


def _project_feet(
    result: np.ndarray,
    body_3d: np.ndarray,
    kpts_2d: np.ndarray,
    min_dist: float,
) -> None:
    """Project 2D foot keypoints to 3D at ankle depth (in-place)."""
    T = body_3d.shape[0]

    for t in range(T):
        # Left foot: anchor at left ankle
        _project_foot_side(
            result, t,
            foot_indices=LEFT_FOOT_INDICES,
            ankle_idx=COCO_LEFT_ANKLE,
            knee_idx=COCO_LEFT_KNEE,
            body_3d=body_3d,
            kpts_2d=kpts_2d,
            min_dist=min_dist,
        )

        # Right foot: anchor at right ankle
        _project_foot_side(
            result, t,
            foot_indices=RIGHT_FOOT_INDICES,
            ankle_idx=COCO_RIGHT_ANKLE,
            knee_idx=COCO_RIGHT_KNEE,
            body_3d=body_3d,
            kpts_2d=kpts_2d,
            min_dist=min_dist,
        )


def _project_foot_side(
    result: np.ndarray,
    t: int,
    foot_indices: list,
    ankle_idx: int,
    knee_idx: int,
    body_3d: np.ndarray,
    kpts_2d: np.ndarray,
    min_dist: float,
) -> None:
    """Project one foot's keypoints to 3D at ankle depth."""
    ankle_3d = body_3d[t, ankle_idx]
    ankle_2d = kpts_2d[t, ankle_idx]
    knee_3d = body_3d[t, knee_idx]
    knee_2d = kpts_2d[t, knee_idx]

    # Scale from shin (knee→ankle)
    shin_2d_len = np.linalg.norm(ankle_2d - knee_2d)
    shin_3d_xy_len = np.linalg.norm(ankle_3d[:2] - knee_3d[:2])

    if shin_2d_len > min_dist and shin_3d_xy_len > 0.01:
        scale = shin_3d_xy_len / shin_2d_len
    else:
        scale = 0.4 / max(shin_2d_len, 50.0)

    for foot_coco_idx in foot_indices:
        foot_2d = kpts_2d[t, foot_coco_idx]
        delta_2d = foot_2d - ankle_2d
        result[t, foot_coco_idx, 0] = ankle_3d[0] + delta_2d[0] * scale
        result[t, foot_coco_idx, 1] = ankle_3d[1] + delta_2d[1] * scale
        result[t, foot_coco_idx, 2] = ankle_3d[2]  # ankle depth
