"""
Project 2D hand keypoints to 3D using wrist depth from MotionBERT.

Uses a local scale factor from the forearm (elbow→wrist) to convert
2D pixel offsets to 3D meter offsets, avoiding the need for focal length
estimation. All hand keypoints are placed at the wrist depth plane.
"""

import numpy as np


# COCO-WholeBody hand keypoint ranges
LEFT_HAND_START = 91
LEFT_HAND_END = 112  # exclusive
RIGHT_HAND_START = 112
RIGHT_HAND_END = 133  # exclusive
HAND_NUM_JOINTS = 21

# COCO body anchors
COCO_LEFT_ELBOW = 7
COCO_LEFT_WRIST = 9
COCO_RIGHT_ELBOW = 8
COCO_RIGHT_WRIST = 10


def project_hands_to_3d(
    keypoints_2d_133: np.ndarray,
    body_3d_coco17: np.ndarray,
    min_forearm_2d: float = 10.0,
) -> np.ndarray:
    """
    Project 2D hand keypoints to 3D using wrist depth and forearm scale.

    For each hand keypoint, computes a pixel-to-meter scale from the
    forearm (elbow→wrist), then projects the 2D hand-to-wrist offset
    into 3D at the wrist depth.

    Args:
        keypoints_2d_133: (T, 133, 2) RTMW 2D keypoints in pixel coords
        body_3d_coco17: (T, 17, 3) MotionBERT 3D body in H36M camera coords
            (X=right, Y=down, Z=forward)
        min_forearm_2d: Minimum forearm length in pixels to use for scaling.
            Below this, a default scale is used.

    Returns:
        hands_3d: (T, 42, 3) 3D hand keypoints (21 left + 21 right)
            in the same coordinate system as body_3d_coco17
    """
    T = keypoints_2d_133.shape[0]
    hands_3d = np.zeros((T, 42, 3), dtype=np.float32)

    for t in range(T):
        # Left hand
        hands_3d[t, :21] = _project_single_hand(
            hand_2d=keypoints_2d_133[t, LEFT_HAND_START:LEFT_HAND_END],
            wrist_2d=keypoints_2d_133[t, COCO_LEFT_WRIST],
            elbow_2d=keypoints_2d_133[t, COCO_LEFT_ELBOW],
            wrist_3d=body_3d_coco17[t, COCO_LEFT_WRIST],
            elbow_3d=body_3d_coco17[t, COCO_LEFT_ELBOW],
            min_forearm_2d=min_forearm_2d,
        )

        # Right hand
        hands_3d[t, 21:] = _project_single_hand(
            hand_2d=keypoints_2d_133[t, RIGHT_HAND_START:RIGHT_HAND_END],
            wrist_2d=keypoints_2d_133[t, COCO_RIGHT_WRIST],
            elbow_2d=keypoints_2d_133[t, COCO_RIGHT_ELBOW],
            wrist_3d=body_3d_coco17[t, COCO_RIGHT_WRIST],
            elbow_3d=body_3d_coco17[t, COCO_RIGHT_ELBOW],
            min_forearm_2d=min_forearm_2d,
        )

    return hands_3d


def _project_single_hand(
    hand_2d: np.ndarray,
    wrist_2d: np.ndarray,
    elbow_2d: np.ndarray,
    wrist_3d: np.ndarray,
    elbow_3d: np.ndarray,
    min_forearm_2d: float,
) -> np.ndarray:
    """
    Project a single hand's 2D keypoints to 3D.

    Args:
        hand_2d: (21, 2) hand keypoints in pixels
        wrist_2d: (2,) wrist position in pixels
        elbow_2d: (2,) elbow position in pixels
        wrist_3d: (3,) wrist position in meters (H36M camera coords)
        elbow_3d: (3,) elbow position in meters
        min_forearm_2d: Minimum forearm length for valid scaling

    Returns:
        hand_3d: (21, 3) hand keypoints in meters
    """
    hand_3d = np.zeros((21, 3), dtype=np.float32)

    # Compute pixel-to-meter scale from forearm
    forearm_2d = wrist_2d - elbow_2d
    forearm_2d_len = np.linalg.norm(forearm_2d)

    forearm_3d = wrist_3d - elbow_3d
    # Use XY components only (lateral + vertical in camera space)
    forearm_3d_xy_len = np.linalg.norm(forearm_3d[:2])

    if forearm_2d_len > min_forearm_2d and forearm_3d_xy_len > 0.01:
        scale = forearm_3d_xy_len / forearm_2d_len
    else:
        # Fallback: assume ~0.25m forearm at typical 2D length
        scale = 0.25 / max(forearm_2d_len, 50.0)

    # Project each hand keypoint
    for j in range(21):
        delta_2d = hand_2d[j] - wrist_2d
        hand_3d[j, 0] = wrist_3d[0] + delta_2d[0] * scale  # X (right)
        hand_3d[j, 1] = wrist_3d[1] + delta_2d[1] * scale  # Y (down)
        hand_3d[j, 2] = wrist_3d[2]  # Z = wrist depth

    return hand_3d
