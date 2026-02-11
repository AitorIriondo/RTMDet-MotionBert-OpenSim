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

from typing import Dict, List, Optional

import numpy as np
from scipy.ndimage import uniform_filter1d


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
        single_level: bool = False,
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
            single_level: If True, per-frame strict grounding (lowest foot = 0
                every frame). If False (default), smoothed ground reference.

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
            transformed = self._align_to_ground(transformed, single_level=single_level)

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
        correct_lean: bool = False,
        fps: float = 30.0,
        foot_indices: Optional[Dict[str, List[int]]] = None,
        single_level: bool = False,
    ) -> np.ndarray:
        """
        Transform keypoints from MotionBERT H36M camera coords to OpenSim.

        Pipeline:
        1. Rotate axes (H36M camera -> OpenSim)
        2. Scale to subject height
        3. Correct forward lean via ground-plane estimation (optional)
        4. Center at pelvis (XZ plane)
        5. Align feet to ground (Y=0)

        Args:
            keypoints_3d: (N, K, 3) in H36M camera coords
            center_pelvis: Center at pelvis in XZ plane
            align_to_ground: Align feet to Y=0
            correct_lean: Correct lean using ground-plane from foot contacts
            fps: Video frame rate (needed for foot velocity computation)
            foot_indices: Dict with 'left' and 'right' keypoint index lists.
                Defaults to COCO-133 feet (ankles + toes + heels).
            single_level: If True, per-frame strict grounding (lowest foot = 0
                every frame). If False (default), smoothed ground reference.

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

        # Correct lean via ground-plane estimation from foot contacts
        if correct_lean:
            transformed = self._correct_ground_plane_lean(
                transformed, fps=fps, foot_indices=foot_indices,
            )

        # Center at pelvis
        if center_pelvis:
            transformed = self._center_at_pelvis(transformed)

        # Align to ground
        if align_to_ground:
            transformed = self._align_to_ground(transformed, single_level=single_level)

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

    # ---- Ground-plane lean correction (foot-contact based) ----

    # Default foot keypoint indices (COCO-WholeBody 133)
    _DEFAULT_FOOT_INDICES = {
        "left": [15, 17, 18, 19],   # LAnkle, LBigToe, LSmallToe, LHeel
        "right": [16, 20, 21, 22],  # RAnkle, RBigToe, RSmallToe, RHeel
    }

    def _correct_ground_plane_lean(
        self,
        keypoints: np.ndarray,
        fps: float = 30.0,
        foot_indices: Optional[Dict[str, List[int]]] = None,
        velocity_threshold: float = 0.3,
        max_correction_deg: float = 15.0,
    ) -> np.ndarray:
        """Correct systematic lean by fitting a ground plane to foot contacts.

        Detects stance phases via foot velocity thresholding, fits a plane
        to the 3D foot positions during stance, and rotates the entire
        skeleton so the ground plane becomes horizontal (Y-normal).

        Args:
            keypoints: (T, K, 3) in OpenSim coordinates (X=fwd, Y=up, Z=right)
            fps: Video frame rate for velocity computation
            foot_indices: Dict with 'left'/'right' keypoint index lists.
                Defaults to COCO-133 feet.
            velocity_threshold: Max foot speed (m/s) to count as stance
            max_correction_deg: Safety clamp for correction angle

        Returns:
            Corrected keypoints
        """
        T = keypoints.shape[0]
        if T < 30:
            print("  Ground-plane correction: skipped (< 30 frames)")
            return keypoints

        if foot_indices is None:
            foot_indices = self._DEFAULT_FOOT_INDICES

        # Step 1: Detect ground contacts
        contact_points = self._detect_ground_contacts(
            keypoints, foot_indices, fps, velocity_threshold,
        )

        if len(contact_points) < 10:
            print(f"  Ground-plane correction: skipped ({len(contact_points)} contact points, need ≥10)")
            return keypoints

        # Step 2: Fit ground plane
        result = self._fit_ground_plane(contact_points)
        if result is None:
            print("  Ground-plane correction: plane fit failed")
            return keypoints
        normal, centroid = result

        # Decompose into sagittal and frontal angles for diagnostics
        sagittal_deg = np.degrees(np.arctan2(normal[0], normal[1]))
        frontal_deg = np.degrees(np.arctan2(normal[2], normal[1]))

        # Step 3: Compute rotation
        R = self._compute_ground_rotation(normal, max_correction_deg)

        total_angle = np.degrees(np.arccos(np.clip(
            np.dot(normal, np.array([0, 1, 0])), -1, 1
        )))

        if total_angle < 0.5:
            print(f"  Ground-plane correction: {total_angle:.1f}° (below threshold, skipping)")
            return keypoints

        print(f"  Ground-plane correction: {total_angle:.1f}° "
              f"(sagittal={sagittal_deg:.1f}°, frontal={frontal_deg:.1f}°, "
              f"{len(contact_points)} contact points)")

        # Step 4: Apply rotation around global pelvis center
        pelvis_all = (keypoints[:, self.LEFT_HIP_IDX] +
                      keypoints[:, self.RIGHT_HIP_IDX]) / 2  # (T, 3)
        pelvis_center = np.mean(pelvis_all, axis=0)  # (3,)

        corrected = keypoints.copy()
        corrected -= pelvis_center
        # Apply rotation: (T, K, 3) @ R.T -> each point rotated
        corrected = corrected @ R.T
        corrected += pelvis_center

        return corrected

    def _detect_ground_contacts(
        self,
        keypoints: np.ndarray,
        foot_indices: Dict[str, List[int]],
        fps: float,
        velocity_threshold: float,
    ) -> np.ndarray:
        """Detect stance frames and collect foot contact positions.

        For each foot, computes the centroid velocity across frames.
        Frames below the velocity threshold (after smoothing) are stance.
        Returns all foot keypoint positions during stance.

        Returns:
            (M, 3) array of ground contact positions
        """
        T = keypoints.shape[0]
        contact_points = []

        for side, indices in foot_indices.items():
            # Foot centroid per frame
            foot_pos = keypoints[:, indices, :].mean(axis=1)  # (T, 3)

            # Velocity in m/s (frame-to-frame displacement * fps)
            velocity = np.zeros(T)
            velocity[1:] = np.linalg.norm(np.diff(foot_pos, axis=0), axis=1) * fps

            # Smooth velocity (5-frame window)
            velocity = uniform_filter1d(velocity, size=5)

            # Stance mask: low velocity
            stance = velocity < velocity_threshold

            # Require ≥3 consecutive stance frames (simple erosion+dilation)
            # Erosion: must have neighbors also in stance
            eroded = np.zeros_like(stance)
            for i in range(1, T - 1):
                eroded[i] = stance[i - 1] and stance[i] and stance[i + 1]
            # Dilation: expand back by 1
            dilated = np.zeros_like(eroded)
            for i in range(T):
                if eroded[i]:
                    dilated[i] = True
                    if i > 0:
                        dilated[i - 1] = True
                    if i < T - 1:
                        dilated[i + 1] = True
            stance = dilated

            n_stance = np.sum(stance)

            # Collect foot positions during stance
            for frame_idx in np.where(stance)[0]:
                for kpt_idx in indices:
                    contact_points.append(keypoints[frame_idx, kpt_idx])

        if len(contact_points) == 0:
            return np.empty((0, 3))

        return np.array(contact_points)

    @staticmethod
    def _fit_ground_plane(contact_points: np.ndarray):
        """Fit a plane to 3D contact points using SVD.

        Returns (normal, centroid) where normal points upward (Y > 0),
        or None if the fit fails.
        """
        if len(contact_points) < 10:
            return None

        centroid = contact_points.mean(axis=0)
        centered = contact_points - centroid

        try:
            _, S, Vt = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            return None

        # Normal = direction of smallest variance
        normal = Vt[2]

        # Orient upward (Y > 0 in OpenSim)
        if normal[1] < 0:
            normal = -normal

        return normal, centroid

    @staticmethod
    def _compute_ground_rotation(
        normal: np.ndarray, max_angle_deg: float = 15.0
    ) -> np.ndarray:
        """Compute rotation matrix to align ground normal with Y-up.

        Uses Rodrigues' formula. Clamps to max_angle_deg for safety.

        Returns:
            (3, 3) rotation matrix
        """
        target = np.array([0.0, 1.0, 0.0])
        axis = np.cross(normal, target)
        sin_angle = np.linalg.norm(axis)
        cos_angle = np.dot(normal, target)

        if sin_angle < 1e-6:
            return np.eye(3)

        axis = axis / sin_angle
        angle = np.arctan2(sin_angle, cos_angle)

        # Safety clamp
        if np.degrees(angle) > max_angle_deg:
            print(f"  WARNING: ground tilt {np.degrees(angle):.1f}° exceeds "
                  f"max {max_angle_deg}°, clamping")
            angle = np.radians(max_angle_deg)

        # Rodrigues formula: R = I + sin(a)*K + (1-cos(a))*K^2
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        return R

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
        """Center keypoints at first-frame pelvis midpoint in XZ plane.

        Only subtracts the first frame's pelvis position so that global
        translation (walking, moving) is preserved across subsequent frames.
        """
        first_pelvis = (keypoints[0, self.LEFT_HIP_IDX] +
                        keypoints[0, self.RIGHT_HIP_IDX]) / 2
        keypoints[:, :, 0] -= first_pelvis[0]
        keypoints[:, :, 2] -= first_pelvis[2]
        return keypoints

    def _align_to_ground(self, keypoints: np.ndarray, single_level: bool = False) -> np.ndarray:
        """Align feet to Y=0 using ground reference.

        Args:
            keypoints: (N, K, 3) array in OpenSim coords (Y = up).
            single_level: If True, use per-frame minimum (lowest foot = 0
                every frame, removes all vertical translation). If False
                (default), use a rolling minimum smoothed over ~0.5s to
                preserve natural vertical motion while reducing jitter.
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

        if single_level:
            # Per-frame strict grounding: lowest foot = Y=0 every frame
            ground_ref = min_foot_y
        else:
            # Smooth the ground reference (15-frame window ≈ 0.5s at 30fps)
            if N > 15:
                ground_ref = uniform_filter1d(min_foot_y, size=15)
            else:
                ground_ref = min_foot_y

        for i in range(N):
            keypoints[i, :, 1] -= ground_ref[i]

        return keypoints
