"""
Coordinate system transformation to OpenSim.

Supports two source conventions:

RTMPose3D (after -kpts[..., [0,2,1]] in package):
    - X: Right (lateral)
    - Y: Forward (depth)
    - Z: Up (vertical)

MotionBERT / H36M Camera:
    - X: Right (lateral)
    - Y: Down (vertical, gravity)
    - Z: Forward (depth)

OpenSim (Biomechanical world):
    - X: Forward (anterior)
    - Y: Up (superior)
    - Z: Right (lateral)
"""

from typing import Optional

import numpy as np


class CoordinateTransformer:
    """Transforms coordinates from RTMPose3D to OpenSim."""

    # Rotation: RTMPose3D -> OpenSim axes
    # OpenSim X = RTMPose3D Y (forward)
    # OpenSim Y = RTMPose3D Z (up)
    # OpenSim Z = RTMPose3D X (right)
    RTM_TO_OPENSIM = np.array(
        [
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ],
        dtype=np.float32,
    )

    # Rotation: MotionBERT (H36M camera) -> OpenSim axes
    # Person faces camera, so person's axes are:
    #   person forward = -camera Z, person up = -camera Y, person right = -camera X
    # OpenSim X = person forward = -MB Z
    # OpenSim Y = person up = -MB Y
    # OpenSim Z = person right = -MB X
    MB_TO_OPENSIM = np.array(
        [
            [0, 0, -1],
            [0, -1, 0],
            [-1, 0, 0],
        ],
        dtype=np.float32,
    )

    # COCO-WholeBody keypoint indices
    NOSE_IDX = 0
    LEFT_SHOULDER_IDX = 5
    RIGHT_SHOULDER_IDX = 6
    LEFT_HIP_IDX = 11
    RIGHT_HIP_IDX = 12
    LEFT_ANKLE_IDX = 15
    RIGHT_ANKLE_IDX = 16
    LEFT_HEEL_IDX = 19
    RIGHT_HEEL_IDX = 22
    LEFT_BIG_TOE_IDX = 17
    RIGHT_BIG_TOE_IDX = 20

    def __init__(self, subject_height: float = 1.75, units: str = "m"):
        """
        Args:
            subject_height: Subject height in meters
            units: Output units ('m' or 'mm')
        """
        self.subject_height = subject_height
        self.units = units
        self.scale_factor = 1000.0 if units == "mm" else 1.0

    def transform(
        self,
        keypoints_3d: np.ndarray,
        center_pelvis: bool = True,
        align_to_ground: bool = True,
        correct_lean: bool = True,
        depth_factor: float = 0.3,
    ) -> np.ndarray:
        """
        Transform keypoints from RTMPose3D to OpenSim coordinates.

        Pipeline:
        1. Rotate axes (RTMPose3D -> OpenSim)
        2. Scale to subject height (global average)
        3. Correct forward lean (rotate spine to vertical)
        4. Center at pelvis (XZ plane)
        5. Compress depth axis (reduce RTMPose3D depth noise)
        6. Align feet to ground (Y=0)

        Args:
            keypoints_3d: (N, K, 3) or (K, 3) array
            center_pelvis: Center at pelvis in XZ plane
            align_to_ground: Align feet to Y=0
            correct_lean: Correct systematic forward/backward lean
            depth_factor: Scale factor for X (depth) deviations from pelvis.
                RTMPose3D depth is ~3x noisier than lateral. 0.3 compresses
                depth to match lateral confidence. 1.0 = no compression.

        Returns:
            Transformed keypoints in OpenSim coordinates
        """
        single_frame = keypoints_3d.ndim == 2
        if single_frame:
            keypoints_3d = keypoints_3d[np.newaxis, ...]

        transformed = keypoints_3d.copy()

        # Apply rotation
        for i in range(transformed.shape[0]):
            transformed[i] = transformed[i] @ self.RTM_TO_OPENSIM.T

        # Scale to subject height (global average)
        transformed = self._scale_to_subject(transformed)

        # Correct forward/backward lean
        if correct_lean:
            transformed = self._correct_forward_lean(transformed)

        # Center at pelvis
        if center_pelvis:
            transformed = self._center_at_pelvis(transformed)

        # Compress depth axis (X in OpenSim = forward/backward)
        if depth_factor < 1.0:
            transformed = self._compress_depth(transformed, depth_factor)

        # Align to ground
        if align_to_ground:
            transformed = self._align_to_ground(transformed)

        # Convert units
        transformed = transformed * self.scale_factor

        if single_frame:
            transformed = transformed[0]

        return transformed

    def transform_motionbert(
        self,
        keypoints_3d: np.ndarray,
        center_pelvis: bool = True,
        align_to_ground: bool = True,
        correct_lean: bool = True,
    ) -> np.ndarray:
        """
        Transform keypoints from MotionBERT H36M camera coords to OpenSim.

        Pipeline:
        1. Rotate axes (H36M camera -> OpenSim)
        2. Scale to subject height
        3. Correct forward lean (camera angle creates systematic tilt)
        4. Center at pelvis (XZ plane)
        5. Align feet to ground (Y=0)

        Args:
            keypoints_3d: (N, K, 3) in H36M camera coords
            center_pelvis: Center at pelvis in XZ plane
            align_to_ground: Align feet to Y=0
            correct_lean: Correct systematic forward/backward lean

        Returns:
            Transformed keypoints in OpenSim coordinates
        """
        single_frame = keypoints_3d.ndim == 2
        if single_frame:
            keypoints_3d = keypoints_3d[np.newaxis, ...]

        transformed = keypoints_3d.copy()

        # Apply rotation (H36M camera -> OpenSim)
        for i in range(transformed.shape[0]):
            transformed[i] = transformed[i] @ self.MB_TO_OPENSIM.T

        # Scale to subject height
        transformed = self._scale_to_subject(transformed)

        # Correct forward lean from camera angle
        if correct_lean:
            transformed = self._correct_forward_lean(transformed)

        # Center at pelvis
        if center_pelvis:
            transformed = self._center_at_pelvis(transformed)

        # Align to ground
        if align_to_ground:
            transformed = self._align_to_ground(transformed)

        # Convert units
        transformed = transformed * self.scale_factor

        if single_frame:
            transformed = transformed[0]

        return transformed

    def _scale_to_subject(self, keypoints: np.ndarray) -> np.ndarray:
        """Scale keypoints to match subject height using global average."""
        heights = []
        for i in range(keypoints.shape[0]):
            head = keypoints[i, self.NOSE_IDX]
            ankle_mid = (keypoints[i, self.LEFT_ANKLE_IDX] +
                         keypoints[i, self.RIGHT_ANKLE_IDX]) / 2
            height = np.linalg.norm(head - ankle_mid)
            if height > 0.1:
                heights.append(height)

        if heights:
            avg_height = np.mean(heights)
            estimated_full = avg_height * 1.1
            scale = self.subject_height / estimated_full
            keypoints = keypoints * scale

        return keypoints

    def _correct_forward_lean(
        self, keypoints: np.ndarray, angle: Optional[float] = None
    ) -> np.ndarray:
        """
        Correct systematic forward/backward lean from depth estimation bias.

        RTMPose3D uses constant-depth estimation which can cause the entire
        skeleton to lean forward/backward. This measures the median spine
        angle (pelvis→shoulders) and rotates around Z to make it vertical.

        In OpenSim coords: X=forward, Y=up, Z=right.
        Forward lean = spine tilts toward +X. Correction rotates around Z.

        Args:
            keypoints: (N, K, 3) in OpenSim coordinates
            angle: Override lean angle in degrees (None=auto-detect)

        Returns:
            Corrected keypoints
        """
        if angle is None:
            angle = self._estimate_lean_angle(keypoints)

        if abs(angle) < 1.0:
            return keypoints

        print(f"  Forward lean correction: {angle:.1f}°")

        # Rotation around Z axis (lateral) to correct lean in XY plane
        rad = np.radians(-angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        rotation = np.array([
            [cos_a, -sin_a, 0],
            [sin_a,  cos_a, 0],
            [0,      0,     1],
        ])

        corrected = keypoints.copy()
        for i in range(corrected.shape[0]):
            pelvis = (corrected[i, self.LEFT_HIP_IDX] +
                      corrected[i, self.RIGHT_HIP_IDX]) / 2
            corrected[i] = corrected[i] - pelvis
            corrected[i] = corrected[i] @ rotation.T
            corrected[i] = corrected[i] + pelvis

        return corrected

    def _estimate_lean_angle(self, keypoints: np.ndarray) -> float:
        """
        Estimate forward lean angle from spine orientation.

        Uses pelvis→shoulder midpoint vector projected onto XY (sagittal) plane.
        Returns median angle across all frames for robustness.
        """
        angles = []
        for i in range(keypoints.shape[0]):
            pelvis = (keypoints[i, self.LEFT_HIP_IDX] +
                      keypoints[i, self.RIGHT_HIP_IDX]) / 2
            shoulders = (keypoints[i, self.LEFT_SHOULDER_IDX] +
                         keypoints[i, self.RIGHT_SHOULDER_IDX]) / 2

            spine_vec = shoulders - pelvis
            # Project onto XY plane (forward-up in OpenSim)
            spine_xy = np.array([spine_vec[0], spine_vec[1]])

            if np.linalg.norm(spine_xy) > 0.01:
                vertical = np.array([0, 1])
                cos_angle = np.dot(spine_xy, vertical) / np.linalg.norm(spine_xy)
                lean = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
                # Sign: positive if forward lean (X > 0)
                if spine_vec[0] > 0:
                    angles.append(lean)
                else:
                    angles.append(-lean)

        if angles:
            return float(np.median(angles))
        return 0.0

    def _compress_depth(
        self, keypoints: np.ndarray, factor: float
    ) -> np.ndarray:
        """
        Stabilize depth (X axis) using median-per-keypoint template.

        RTMPose3D's depth estimation varies wildly frame-to-frame but the
        median across frames captures reasonable anatomical depth structure.
        Replace per-frame depth with: median_offset * factor, providing
        a stable facing direction signal without frame-to-frame noise.

        When factor < 1.0, the depth spread is compressed to account for
        RTMPose3D overestimating depth differences.

        Args:
            keypoints: (N, K, 3) pelvis-centered keypoints
            factor: Scale applied to median depth (1.0 = full median, 0.3 = compressed)
        """
        # Compute per-keypoint median X offset across all frames
        median_x = np.median(keypoints[:, :, 0], axis=0)  # (K,)

        # Replace per-frame X with scaled median template
        for i in range(keypoints.shape[0]):
            keypoints[i, :, 0] = median_x * factor

        return keypoints

    def _center_at_pelvis(self, keypoints: np.ndarray) -> np.ndarray:
        """Center keypoints at pelvis midpoint in XZ plane."""
        for i in range(keypoints.shape[0]):
            pelvis = (keypoints[i, self.LEFT_HIP_IDX] +
                      keypoints[i, self.RIGHT_HIP_IDX]) / 2
            keypoints[i, :, 0] -= pelvis[0]
            keypoints[i, :, 2] -= pelvis[2]
        return keypoints

    def _align_to_ground(self, keypoints: np.ndarray) -> np.ndarray:
        """Align feet to Y=0 using smoothed ground reference.

        Uses a rolling minimum of foot heights (smoothed over ~0.5s window)
        instead of per-frame minimum to reduce vertical jitter.
        """
        N = keypoints.shape[0]

        # Collect per-frame minimum foot height
        min_foot_y = np.zeros(N)
        for i in range(N):
            foot_heights = [
                keypoints[i, self.LEFT_HEEL_IDX, 1],
                keypoints[i, self.RIGHT_HEEL_IDX, 1],
                keypoints[i, self.LEFT_BIG_TOE_IDX, 1],
                keypoints[i, self.RIGHT_BIG_TOE_IDX, 1],
            ]
            min_foot_y[i] = min(foot_heights)

        # Smooth the ground reference (15-frame window ≈ 0.5s at 30fps)
        if N > 15:
            from scipy.ndimage import uniform_filter1d
            ground_ref = uniform_filter1d(min_foot_y, size=15)
        else:
            ground_ref = min_foot_y

        for i in range(N):
            keypoints[i, :, 1] -= ground_ref[i]

        return keypoints
