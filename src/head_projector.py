"""
Project 2D head keypoints (eyes, ears) to 3D using head depth from MotionBERT.

Eyes and ears are not in H36M-17, so they're not directly available from
MotionBERT. We back-project their 2D pixel positions to 3D using the
nose depth and a local scale from the shoulder width.
"""

import numpy as np


# COCO-WholeBody head keypoint indices
COCO_NOSE = 0
COCO_LEFT_EYE = 1
COCO_RIGHT_EYE = 2
COCO_LEFT_EAR = 3
COCO_RIGHT_EAR = 4

# COCO body anchors for scaling
COCO_LEFT_SHOULDER = 5
COCO_RIGHT_SHOULDER = 6

# Head keypoints we project (5 total)
HEAD_KEYPOINT_INDICES = [
    COCO_LEFT_EYE,
    COCO_RIGHT_EYE,
    COCO_LEFT_EAR,
    COCO_RIGHT_EAR,
]


def project_head_to_3d(
    keypoints_2d_133: np.ndarray,
    body_3d_coco17: np.ndarray,
    min_shoulder_2d: float = 20.0,
) -> np.ndarray:
    """
    Project 2D eye and ear keypoints to 3D using nose/head depth.

    Uses shoulder width as reference for pixel-to-meter scale.
    All projected keypoints are placed at the nose depth plane.

    Args:
        keypoints_2d_133: (T, 133, 2) RTMW 2D keypoints in pixel coords
        body_3d_coco17: (T, 17, 3) MotionBERT 3D body in H36M camera coords
        min_shoulder_2d: Minimum shoulder width in pixels for valid scaling

    Returns:
        head_3d: (T, 4, 3) projected [LEye, REye, LEar, REar] in 3D
    """
    T = keypoints_2d_133.shape[0]
    head_3d = np.zeros((T, 4, 3), dtype=np.float32)

    for t in range(T):
        # Anchor: nose 3D from MotionBERT
        nose_3d = body_3d_coco17[t, COCO_NOSE]
        nose_2d = keypoints_2d_133[t, COCO_NOSE]

        # Scale from shoulder width
        lshoulder_2d = keypoints_2d_133[t, COCO_LEFT_SHOULDER]
        rshoulder_2d = keypoints_2d_133[t, COCO_RIGHT_SHOULDER]
        shoulder_2d_width = np.linalg.norm(lshoulder_2d - rshoulder_2d)

        lshoulder_3d = body_3d_coco17[t, COCO_LEFT_SHOULDER]
        rshoulder_3d = body_3d_coco17[t, COCO_RIGHT_SHOULDER]
        shoulder_3d_width_xy = np.linalg.norm(
            lshoulder_3d[:2] - rshoulder_3d[:2]
        )

        if shoulder_2d_width > min_shoulder_2d and shoulder_3d_width_xy > 0.01:
            scale = shoulder_3d_width_xy / shoulder_2d_width
        else:
            # Fallback: ~0.35m shoulder width at typical 2D size
            scale = 0.35 / max(shoulder_2d_width, 100.0)

        # Project each head keypoint relative to nose
        for i, coco_idx in enumerate(HEAD_KEYPOINT_INDICES):
            kp_2d = keypoints_2d_133[t, coco_idx]
            delta_2d = kp_2d - nose_2d
            head_3d[t, i, 0] = nose_3d[0] + delta_2d[0] * scale  # X
            head_3d[t, i, 1] = nose_3d[1] + delta_2d[1] * scale  # Y
            head_3d[t, i, 2] = nose_3d[2]  # Z = nose depth

    return head_3d
