"""
COCO-17 <-> H36M-17 keypoint format conversion.

Ported from MotionBERT-OpenSim (Apache 2.0).
"""

import numpy as np
from typing import Optional, Tuple


# COCO-17 body joint indices (from COCO-WholeBody indices 0-16)
COCO_NOSE = 0
COCO_LEFT_EYE = 1
COCO_RIGHT_EYE = 2
COCO_LEFT_EAR = 3
COCO_RIGHT_EAR = 4
COCO_LEFT_SHOULDER = 5
COCO_RIGHT_SHOULDER = 6
COCO_LEFT_ELBOW = 7
COCO_RIGHT_ELBOW = 8
COCO_LEFT_WRIST = 9
COCO_RIGHT_WRIST = 10
COCO_LEFT_HIP = 11
COCO_RIGHT_HIP = 12
COCO_LEFT_KNEE = 13
COCO_RIGHT_KNEE = 14
COCO_LEFT_ANKLE = 15
COCO_RIGHT_ANKLE = 16

# H36M-17 joint indices
H36M_HIP = 0        # pelvis center (derived)
H36M_RHIP = 1
H36M_RKNEE = 2
H36M_RFOOT = 3
H36M_LHIP = 4
H36M_LKNEE = 5
H36M_LFOOT = 6
H36M_SPINE = 7      # derived
H36M_THORAX = 8     # derived
H36M_NECK_NOSE = 9
H36M_HEAD = 10      # derived from eyes/ears
H36M_LSHOULDER = 11
H36M_LELBOW = 12
H36M_LWRIST = 13
H36M_RSHOULDER = 14
H36M_RELBOW = 15
H36M_RWRIST = 16


def coco17_to_h36m(
    coco_kpts: np.ndarray,
    coco_scores: Optional[np.ndarray] = None,
    spine_ratio: float = 0.5,
    thorax_ratio: float = 0.75,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Convert COCO-17 keypoints to H36M-17 format.

    Args:
        coco_kpts: (T, 17, D) or (17, D) keypoints (D=2 for 2D, D=3 for 3D)
        coco_scores: (T, 17) or (17,) confidence scores
        spine_ratio: interpolation ratio for spine joint
        thorax_ratio: interpolation ratio for thorax joint

    Returns:
        h36m_kpts: (T, 17, D) H36M keypoints
        h36m_scores: (T, 17) scores or None
    """
    single = coco_kpts.ndim == 2
    if single:
        coco_kpts = coco_kpts[np.newaxis]
        if coco_scores is not None:
            coco_scores = coco_scores[np.newaxis]

    T, _, D = coco_kpts.shape
    h36m = np.zeros((T, 17, D), dtype=np.float32)

    # Direct mappings
    h36m[:, H36M_RHIP] = coco_kpts[:, COCO_RIGHT_HIP]
    h36m[:, H36M_RKNEE] = coco_kpts[:, COCO_RIGHT_KNEE]
    h36m[:, H36M_RFOOT] = coco_kpts[:, COCO_RIGHT_ANKLE]
    h36m[:, H36M_LHIP] = coco_kpts[:, COCO_LEFT_HIP]
    h36m[:, H36M_LKNEE] = coco_kpts[:, COCO_LEFT_KNEE]
    h36m[:, H36M_LFOOT] = coco_kpts[:, COCO_LEFT_ANKLE]
    h36m[:, H36M_LSHOULDER] = coco_kpts[:, COCO_LEFT_SHOULDER]
    h36m[:, H36M_LELBOW] = coco_kpts[:, COCO_LEFT_ELBOW]
    h36m[:, H36M_LWRIST] = coco_kpts[:, COCO_LEFT_WRIST]
    h36m[:, H36M_RSHOULDER] = coco_kpts[:, COCO_RIGHT_SHOULDER]
    h36m[:, H36M_RELBOW] = coco_kpts[:, COCO_RIGHT_ELBOW]
    h36m[:, H36M_RWRIST] = coco_kpts[:, COCO_RIGHT_WRIST]
    h36m[:, H36M_NECK_NOSE] = coco_kpts[:, COCO_NOSE]

    # Derived joints
    hip_center = (coco_kpts[:, COCO_LEFT_HIP] + coco_kpts[:, COCO_RIGHT_HIP]) / 2.0
    h36m[:, H36M_HIP] = hip_center

    shoulder_center = (coco_kpts[:, COCO_LEFT_SHOULDER] + coco_kpts[:, COCO_RIGHT_SHOULDER]) / 2.0
    thorax = hip_center * (1 - thorax_ratio) + shoulder_center * thorax_ratio
    h36m[:, H36M_THORAX] = thorax
    h36m[:, H36M_SPINE] = hip_center * (1 - spine_ratio) + thorax * spine_ratio

    # Head = mean of eyes and ears
    head = (coco_kpts[:, COCO_LEFT_EYE] + coco_kpts[:, COCO_RIGHT_EYE] +
            coco_kpts[:, COCO_LEFT_EAR] + coco_kpts[:, COCO_RIGHT_EAR]) / 4.0
    h36m[:, H36M_HEAD] = head

    # Scores
    h36m_scores = None
    if coco_scores is not None:
        h36m_scores = np.zeros((T, 17), dtype=np.float32)
        h36m_scores[:, H36M_RHIP] = coco_scores[:, COCO_RIGHT_HIP]
        h36m_scores[:, H36M_RKNEE] = coco_scores[:, COCO_RIGHT_KNEE]
        h36m_scores[:, H36M_RFOOT] = coco_scores[:, COCO_RIGHT_ANKLE]
        h36m_scores[:, H36M_LHIP] = coco_scores[:, COCO_LEFT_HIP]
        h36m_scores[:, H36M_LKNEE] = coco_scores[:, COCO_LEFT_KNEE]
        h36m_scores[:, H36M_LFOOT] = coco_scores[:, COCO_LEFT_ANKLE]
        h36m_scores[:, H36M_LSHOULDER] = coco_scores[:, COCO_LEFT_SHOULDER]
        h36m_scores[:, H36M_LELBOW] = coco_scores[:, COCO_LEFT_ELBOW]
        h36m_scores[:, H36M_LWRIST] = coco_scores[:, COCO_LEFT_WRIST]
        h36m_scores[:, H36M_RSHOULDER] = coco_scores[:, COCO_RIGHT_SHOULDER]
        h36m_scores[:, H36M_RELBOW] = coco_scores[:, COCO_RIGHT_ELBOW]
        h36m_scores[:, H36M_RWRIST] = coco_scores[:, COCO_RIGHT_WRIST]
        h36m_scores[:, H36M_NECK_NOSE] = coco_scores[:, COCO_NOSE]
        h36m_scores[:, H36M_HIP] = np.minimum(coco_scores[:, COCO_LEFT_HIP],
                                                coco_scores[:, COCO_RIGHT_HIP])
        h36m_scores[:, H36M_THORAX] = h36m_scores[:, H36M_HIP]
        h36m_scores[:, H36M_SPINE] = h36m_scores[:, H36M_HIP]
        h36m_scores[:, H36M_HEAD] = np.minimum(coco_scores[:, COCO_LEFT_EYE],
                                                coco_scores[:, COCO_RIGHT_EYE])

    if single:
        h36m = h36m[0]
        if h36m_scores is not None:
            h36m_scores = h36m_scores[0]

    return h36m, h36m_scores


def h36m_to_coco17(h36m_kpts: np.ndarray) -> np.ndarray:
    """
    Convert H36M-17 3D keypoints back to COCO-17 ordering.

    Args:
        h36m_kpts: (T, 17, 3) H36M 3D keypoints

    Returns:
        coco_kpts: (T, 17, 3) COCO-17 3D keypoints
        Note: eyes and ears are approximated from Head joint.
    """
    single = h36m_kpts.ndim == 2
    if single:
        h36m_kpts = h36m_kpts[np.newaxis]

    T, _, D = h36m_kpts.shape
    coco = np.zeros((T, 17, D), dtype=np.float32)

    # Direct mappings
    coco[:, COCO_NOSE] = h36m_kpts[:, H36M_NECK_NOSE]
    coco[:, COCO_LEFT_SHOULDER] = h36m_kpts[:, H36M_LSHOULDER]
    coco[:, COCO_RIGHT_SHOULDER] = h36m_kpts[:, H36M_RSHOULDER]
    coco[:, COCO_LEFT_ELBOW] = h36m_kpts[:, H36M_LELBOW]
    coco[:, COCO_RIGHT_ELBOW] = h36m_kpts[:, H36M_RELBOW]
    coco[:, COCO_LEFT_WRIST] = h36m_kpts[:, H36M_LWRIST]
    coco[:, COCO_RIGHT_WRIST] = h36m_kpts[:, H36M_RWRIST]
    coco[:, COCO_LEFT_HIP] = h36m_kpts[:, H36M_LHIP]
    coco[:, COCO_RIGHT_HIP] = h36m_kpts[:, H36M_RHIP]
    coco[:, COCO_LEFT_KNEE] = h36m_kpts[:, H36M_LKNEE]
    coco[:, COCO_RIGHT_KNEE] = h36m_kpts[:, H36M_RKNEE]
    coco[:, COCO_LEFT_ANKLE] = h36m_kpts[:, H36M_LFOOT]
    coco[:, COCO_RIGHT_ANKLE] = h36m_kpts[:, H36M_RFOOT]

    # Eyes and ears approximated from Head position
    # (will be overridden by projected 2D positions in hybrid pipeline)
    coco[:, COCO_LEFT_EYE] = h36m_kpts[:, H36M_HEAD]
    coco[:, COCO_RIGHT_EYE] = h36m_kpts[:, H36M_HEAD]
    coco[:, COCO_LEFT_EAR] = h36m_kpts[:, H36M_HEAD]
    coco[:, COCO_RIGHT_EAR] = h36m_kpts[:, H36M_HEAD]

    if single:
        coco = coco[0]

    return coco
