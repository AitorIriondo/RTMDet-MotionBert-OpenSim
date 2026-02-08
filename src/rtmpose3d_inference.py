"""
RTMPose3D inference wrapper.

This module wraps the standalone RTMPose3D (RTMW3D) package
for 3D whole-body pose estimation from monocular images.

RTMPose3D outputs 133 COCO-WholeBody keypoints in 3D:
  - Body (0-16): 17 standard body joints
  - Feet (17-22): 6 foot landmarks
  - Face (23-90): 68 face landmarks
  - Left Hand (91-111): 21 hand joints
  - Right Hand (112-132): 21 hand joints

Output is root-relative (hip midpoint as origin), Z-up coordinate system.
Units are meters.

NOTE: RTMPose3D does NOT provide:
  - Camera translation (cam_t) - no global position tracking
  - Focal length estimation
  - Body mesh or shape parameters
"""

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm


class RTMPose3DInference:
    """
    Wrapper for RTMPose3D (RTMW3D) model inference.

    Uses the standalone b-arac/rtmpose3d package with
    HuggingFace-style API for model loading.
    """

    NUM_KEYPOINTS = 133

    def __init__(
        self,
        model_name: str = "rbarac/rtmpose3d",
        device: str = "cuda:0",
        confidence_threshold: float = 0.3,
    ):
        """
        Initialize RTMPose3D model.

        Args:
            model_name: HuggingFace model name for from_pretrained()
            device: Device string ('cuda:0' or 'cpu')
            confidence_threshold: Min confidence to consider a keypoint valid
        """
        from rtmpose3d import RTMPose3D

        self.model = RTMPose3D.from_pretrained(model_name, device=device)
        self.device = device
        self.confidence_threshold = confidence_threshold

    def process_frame(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Process a single frame.

        Args:
            image: BGR numpy array (H, W, 3), uint8 — as returned by cv2.imread()

        Returns:
            List of dicts per detected person, each containing:
                - keypoints_3d: (133, 3) 3D keypoints in meters, root-relative
                - scores: (133,) confidence per keypoint [0, 1]
                - bbox: (4,) detection bounding box [x1, y1, x2, y2]
        """
        results = self.model(image)

        persons = []
        num_persons = results["keypoints_3d"].shape[0]

        for p in range(num_persons):
            person = {
                "keypoints_3d": results["keypoints_3d"][p],  # (133, 3)
                "keypoints_2d": results["keypoints_2d"][p],  # (133, 2) pixel coords
                "scores": results["scores"][p],               # (133,)
                "bbox": results["bboxes"][p],                  # (4,)
            }
            persons.append(person)

        return persons

    def process_video(
        self,
        frame_paths: List[str],
        person_idx: int = 0,
        progress: bool = True,
    ) -> Dict[str, Any]:
        """
        Process all video frames.

        Args:
            frame_paths: List of paths to frame images
            person_idx: Which detected person to track (0 = first/largest)
            progress: Show progress bar

        Returns:
            Dictionary with:
                - keypoints_3d: (T, 133, 3) array
                - scores: (T, 133) array
                - bboxes: (T, 4) array
                - valid_frames: (T,) boolean array
        """
        num_frames = len(frame_paths)
        keypoints_3d = np.zeros((num_frames, self.NUM_KEYPOINTS, 3), dtype=np.float32)
        scores = np.zeros((num_frames, self.NUM_KEYPOINTS), dtype=np.float32)
        bboxes = np.zeros((num_frames, 4), dtype=np.float32)
        valid_frames = np.zeros(num_frames, dtype=bool)

        iterator = tqdm(frame_paths, desc="RTMPose3D inference") if progress else frame_paths

        for idx, frame_path in enumerate(iterator):
            image = cv2.imread(frame_path)
            if image is None:
                continue
            # cv2.imread returns BGR, which is what RTMPose3D expects

            persons = self.process_frame(image)

            if len(persons) > person_idx:
                person = persons[person_idx]
                keypoints_3d[idx] = person["keypoints_3d"]
                scores[idx] = person["scores"]
                bboxes[idx] = person["bbox"]
                valid_frames[idx] = True

        return {
            "keypoints_3d": keypoints_3d,
            "scores": scores,
            "bboxes": bboxes,
            "valid_frames": valid_frames,
        }
