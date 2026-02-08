"""
COCO-WholeBody to Pose2Sim COCO_133 marker conversion.

Outputs marker names that match Pose2Sim's COCO_133 configuration,
enabling direct use of Pose2Sim.kinematics() for scaling + IK.

Pose2Sim COCO_133 expects 27 markers:
  Head:  Nose, LEye, REye
  Body:  LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
  Hands: LThumb, LIndex, LPinky, RThumb, RIndex, RPinky
  Hips:  LHip, RHip
  Legs:  LKnee, RKnee, LAnkle, RAnkle
  Feet:  LBigToe, LSmallToe, LHeel, RBigToe, RSmallToe, RHeel
"""

from typing import Dict, List, Tuple
import numpy as np


class KeypointConverter:
    """
    Converts COCO-WholeBody 133 keypoints to Pose2Sim COCO_133 markers.
    """

    # COCO-WholeBody index -> Pose2Sim COCO_133 marker name
    # Matches Pose2Sim's Markers_Coco133.xml and Scaling_Setup_Pose2Sim_Coco133.xml
    MARKER_MAPPING = {
        # Head
        "Nose":       0,
        "LEye":       1,
        "REye":       2,
        # Body
        "LShoulder":  5,
        "RShoulder":  6,
        "LElbow":     7,
        "RElbow":     8,
        "LWrist":     9,
        "RWrist":     10,
        # Pelvis
        "LHip":       11,
        "RHip":       12,
        # Legs
        "LKnee":      13,
        "RKnee":      14,
        "LAnkle":     15,
        "RAnkle":     16,
        # Feet
        "LBigToe":    17,
        "LSmallToe":  18,
        "LHeel":      19,
        "RBigToe":    20,
        "RSmallToe":  21,
        "RHeel":      22,
        # Hands (finger tips from COCO-WholeBody hand keypoints)
        # Left hand: 91=wrist, 92-95=thumb, 96-99=index, 100-103=middle,
        #            104-107=ring, 108-111=pinky
        "LThumb":     95,   # left thumb tip
        "LIndex":     99,   # left index tip
        "LPinky":     111,  # left pinky tip
        # Right hand: 112=wrist, 113-116=thumb, 117-120=index, 121-124=middle,
        #             125-128=ring, 129-132=pinky
        "RThumb":     116,  # right thumb tip
        "RIndex":     120,  # right index tip
        "RPinky":     132,  # right pinky tip
    }

    def __init__(self):
        """Initialize keypoint converter."""
        pass

    def convert(
        self,
        keypoints_3d: np.ndarray,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Convert COCO-WholeBody 133 keypoints to Pose2Sim COCO_133 markers.

        Args:
            keypoints_3d: (N, 133, 3) or (133, 3) array

        Returns:
            Tuple of:
                - markers: (N, 27, 3) or (27, 3) array
                - marker_names: List of 27 marker names
        """
        single_frame = keypoints_3d.ndim == 2
        if single_frame:
            keypoints_3d = keypoints_3d[np.newaxis, ...]

        marker_names = list(self.MARKER_MAPPING.keys())
        marker_indices = list(self.MARKER_MAPPING.values())

        markers = keypoints_3d[:, marker_indices, :]

        if single_frame:
            markers = markers[0]

        return markers, marker_names

    def get_marker_names(self) -> List[str]:
        """Get list of marker names in order."""
        return list(self.MARKER_MAPPING.keys())
