"""
Post-processing utilities for keypoint sequences.

This module provides optional smoothing and bone normalization
for COCO-WholeBody 133-keypoint sequences from RTMPose3D.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np


class PostProcessor:
    """
    Post-processes 3D keypoint sequences.

    Provides:
    - Temporal smoothing (Butterworth filter)
    - Bone length normalization
    - Outlier detection and interpolation
    """

    # Bone connections for COCO-WholeBody skeleton
    # Format: (parent_idx, child_idx)
    BONE_CONNECTIONS = [
        # Pelvis reference
        (11, 12),   # left_hip to right_hip (pelvis width)
        # Left leg
        (11, 13),   # left_hip to left_knee
        (13, 15),   # left_knee to left_ankle
        # Right leg
        (12, 14),   # right_hip to right_knee
        (14, 16),   # right_knee to right_ankle
        # Shoulder width
        (5, 6),     # left_shoulder to right_shoulder
        # Left arm
        (5, 7),     # left_shoulder to left_elbow
        (7, 9),     # left_elbow to left_wrist
        # Right arm
        (6, 8),     # right_shoulder to right_elbow
        (8, 10),    # right_elbow to right_wrist
        # Feet
        (15, 19),   # left_ankle to left_heel
        (15, 17),   # left_ankle to left_big_toe
        (16, 22),   # right_ankle to right_heel
        (16, 20),   # right_ankle to right_big_toe
    ]

    # Bone chains in proximal-to-distal order (for cascading normalization)
    BONE_CHAIN_ORDER = [
        (11, 13), (13, 15),   # left leg: hip→knee→ankle
        (12, 14), (14, 16),   # right leg: hip→knee→ankle
        (5, 7),   (7, 9),     # left arm: shoulder→elbow→wrist
        (6, 8),   (8, 10),    # right arm: shoulder→elbow→wrist
    ]

    # When a joint moves, these chain keypoints must move with it.
    # Only includes direct chain joints (ankle/wrist), NOT foot/hand markers,
    # to avoid pushing extremity markers into ambiguous positions.
    DOWNSTREAM_KEYPOINTS = {
        13: [15],          # L knee → L ankle (not foot markers)
        14: [16],          # R knee → R ankle (not foot markers)
        7:  [9],           # L elbow → L wrist (not hand keypoints)
        8:  [10],          # R elbow → R wrist (not hand keypoints)
    }

    # RTMPose3D depth axis index (Y=forward/depth in RTMPose3D coordinates)
    DEPTH_AXIS = 1

    def __init__(
        self,
        smooth_filter: bool = False,
        filter_cutoff: float = 6.0,
        depth_cutoff: float = 0.5,
        filter_order: int = 4,
        normalize_bones: bool = True,
    ):
        """
        Initialize post-processor.

        Args:
            smooth_filter: Whether to apply Butterworth filter
            filter_cutoff: Filter cutoff frequency in Hz (lateral/vertical)
            depth_cutoff: Filter cutoff for depth axis in Hz (much lower
                because RTMPose3D depth is ~3x noisier than lateral)
            filter_order: Filter order
            normalize_bones: Whether to normalize bone lengths
        """
        self.smooth_filter = smooth_filter
        self.filter_cutoff = filter_cutoff
        self.depth_cutoff = depth_cutoff
        self.filter_order = filter_order
        self.normalize_bones = normalize_bones

    # Left/right pairs for body keypoints
    # Format: (left_idx, right_idx)
    LR_PAIRS = [
        (5, 6),    # shoulders
        (7, 8),    # elbows
        (9, 10),   # wrists
        (11, 12),  # hips
        (13, 14),  # knees
        (15, 16),  # ankles
        (17, 20),  # big toes
        (18, 21),  # small toes
        (19, 22),  # heels
    ]

    # Hand L/R pairs (left hand: 91-111, right hand: 112-132)
    HAND_LR_PAIRS = [(91 + i, 112 + i) for i in range(21)]

    def process(
        self,
        keypoints: np.ndarray,
        fps: float = 30.0,
    ) -> np.ndarray:
        """
        Apply post-processing pipeline.

        Steps:
        1. Interpolate missing frames
        2. Fix L/R flips (cross-product facing detection)
        3. Normalize bone lengths to median
        4. Apply Butterworth smoothing

        Args:
            keypoints: (T, K, 3) keypoint sequence
            fps: Frame rate in Hz

        Returns:
            Processed keypoint sequence
        """
        processed = keypoints.copy()

        # Interpolate missing frames
        processed = self._interpolate_missing(processed)

        # Detect and fix L/R flips using cross-product method
        # (adapted from MotionBERT-OpenSim's fix_left_right_consistency)
        processed = self._fix_lr_flips(processed)

        # Reduce L/R lateral asymmetry (partial, not full forcing)
        processed = self._reduce_lr_asymmetry(processed, strength=0.5)

        # Normalize bone lengths to median (before smoothing)
        if self.normalize_bones:
            processed = self._normalize_bones(processed)

        # Apply smoothing filter
        if self.smooth_filter:
            processed = self._apply_butterworth(processed, fps)

        return processed

    def _reduce_lr_asymmetry(
        self, keypoints: np.ndarray, strength: float = 0.5
    ) -> np.ndarray:
        """
        Partially reduce L/R lateral asymmetry.

        RTMPose3D has systematic L/R asymmetry (one side detected further
        from midline than the other). Instead of forcing both sides to the
        same distance (which shifts errors), this partially moves each side
        toward the symmetric average.

        strength=0: no correction
        strength=1: force both sides to identical distance (old approach)
        strength=0.5: move each side 50% toward the average (recommended)

        Operates on the lateral axis (dim 0 = RTMPose3D X = right/left).
        """
        result = keypoints.copy()
        T = result.shape[0]

        for l_idx, r_idx in self.LR_PAIRS:
            if l_idx >= result.shape[1] or r_idx >= result.shape[1]:
                continue

            # Compute per-frame midline and distances
            midline = (result[:, 11, 0] + result[:, 12, 0]) / 2
            l_dist = result[:, l_idx, 0] - midline  # signed distance
            r_dist = result[:, r_idx, 0] - midline

            # Target: average of absolute distances, with original signs
            avg_abs = (np.abs(l_dist) + np.abs(r_dist)) / 2
            l_target = np.sign(l_dist) * avg_abs
            r_target = np.sign(r_dist) * avg_abs

            # Partial correction
            result[:, l_idx, 0] = midline + l_dist + strength * (l_target - l_dist)
            result[:, r_idx, 0] = midline + r_dist + strength * (r_target - r_dist)

        return result

    def _fix_lr_flips(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Detect and fix left/right flips using cross product of shoulder and
        spine vectors.

        Adapted from MotionBERT-OpenSim's fix_left_right_consistency().

        In RTMPose3D coordinates (X=right, Y=forward, Z=up):
        - shoulder_vec = RShoulder - LShoulder (should point +X)
        - spine_vec = shoulder_center - pelvis_center (should point +Z)
        - cross(shoulder, spine) Y component indicates facing direction

        When L/R are swapped, the cross product Y component flips sign.
        We use the median of the first 30 frames as reference and swap
        all L/R pairs for frames where the sign disagrees.
        """
        result = keypoints.copy()
        T, K, _ = result.shape

        L_SHOULDER, R_SHOULDER = 5, 6
        L_HIP, R_HIP = 11, 12

        # Compute facing indicator (cross product Y component) per frame
        indicators = np.zeros(T)
        for t in range(T):
            shoulder_vec = result[t, R_SHOULDER] - result[t, L_SHOULDER]
            pelvis = (result[t, L_HIP] + result[t, R_HIP]) / 2
            shoulder_center = (result[t, L_SHOULDER] + result[t, R_SHOULDER]) / 2
            spine_vec = shoulder_center - pelvis
            cross = np.cross(shoulder_vec, spine_vec)
            indicators[t] = cross[1]  # Y component

        # Reference sign from first 30 frames
        ref_sign = np.sign(np.median(indicators[:min(30, T)]))

        # All pairs to swap on flip (body + feet + hands)
        all_lr_pairs = list(self.LR_PAIRS) + list(self.HAND_LR_PAIRS)

        n_flipped = 0
        for t in range(T):
            curr_sign = np.sign(indicators[t])
            if curr_sign != 0 and curr_sign != ref_sign:
                for l_idx, r_idx in all_lr_pairs:
                    if l_idx < K and r_idx < K:
                        result[t, [l_idx, r_idx]] = result[t, [r_idx, l_idx]]
                n_flipped += 1

        print(f"  L/R flip detection: flipped {n_flipped}/{T} frames")
        return result

    def _enforce_lr_symmetry(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Enforce left/right lateral symmetry.

        RTMPose3D from monocular video has systematic L/R asymmetry
        (one side up to 1.8x further from midline than the other).
        Uses the median absolute distance across all frames as the
        target for both sides — robust to per-frame noise.

        Operates on the lateral axis (dim 0 = RTMPose3D X = right/left).
        """
        result = keypoints.copy()
        T = result.shape[0]

        # Compute median absolute distance for each pair across all frames
        pair_targets = {}
        for l_idx, r_idx in self.LR_PAIRS:
            if l_idx >= result.shape[1] or r_idx >= result.shape[1]:
                continue
            # Lateral distances from hip midline per frame
            midline = (result[:, 11, 0] + result[:, 12, 0]) / 2
            l_dists = np.abs(result[:, l_idx, 0] - midline)
            r_dists = np.abs(result[:, r_idx, 0] - midline)
            # Use minimum of the two medians (less affected by error)
            target = min(np.median(l_dists), np.median(r_dists))
            pair_targets[(l_idx, r_idx)] = target

        # Apply symmetric distances
        for t in range(T):
            midline_x = (result[t, 11, 0] + result[t, 12, 0]) / 2
            for l_idx, r_idx in self.LR_PAIRS:
                if (l_idx, r_idx) not in pair_targets:
                    continue
                target = pair_targets[(l_idx, r_idx)]
                l_sign = np.sign(result[t, l_idx, 0] - midline_x)
                r_sign = np.sign(result[t, r_idx, 0] - midline_x)
                if l_sign == 0:
                    l_sign = -1  # left defaults to negative
                if r_sign == 0:
                    r_sign = 1   # right defaults to positive
                result[t, l_idx, 0] = midline_x + target * l_sign
                result[t, r_idx, 0] = midline_x + target * r_sign

        return result

    def apply_bone_normalization(
        self,
        keypoints: np.ndarray,
        subject_height: float = 1.75,
    ) -> np.ndarray:
        """Legacy API — bone normalization now runs inside process()."""
        if not self.normalize_bones:
            return keypoints
        return self._normalize_bones(keypoints)

    def _interpolate_missing(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Interpolate frames with missing/invalid keypoints.

        Args:
            keypoints: (T, K, 3) keypoint sequence

        Returns:
            Interpolated sequence
        """
        result = keypoints.copy()
        T, K, _ = result.shape

        for k in range(K):
            for dim in range(3):
                values = result[:, k, dim]

                # Find valid (non-zero, non-NaN) values
                valid_mask = ~np.isnan(values) & (values != 0)

                if np.sum(valid_mask) < 2:
                    continue

                # Interpolate missing values
                valid_indices = np.where(valid_mask)[0]
                invalid_indices = np.where(~valid_mask)[0]

                if len(invalid_indices) > 0:
                    result[invalid_indices, k, dim] = np.interp(
                        invalid_indices, valid_indices, values[valid_indices]
                    )

        return result

    def _normalize_bones(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Normalize bone lengths to per-sequence median for consistency.

        Adjusts child joint positions along each bone to match the median
        length across all frames. Processes bones proximal→distal so each
        chain (hip→knee→ankle) is internally consistent.

        Cascades adjustments to direct chain joints (ankle, wrist) but
        NOT to extremity markers (feet, hands) to avoid IK ambiguity.

        Args:
            keypoints: (T, K, 3) keypoint sequence

        Returns:
            Normalized sequence
        """
        result = keypoints.copy()
        T = result.shape[0]

        # Compute median bone lengths from the sequence
        medians = {}
        for pair in self.BONE_CHAIN_ORDER:
            p, c = pair
            lengths = np.linalg.norm(result[:, c] - result[:, p], axis=1)
            valid = lengths > 0.001
            if np.any(valid):
                medians[pair] = np.median(lengths[valid])
            else:
                medians[pair] = 0.0

        # Normalize each frame (proximal → distal, no foot/hand cascade)
        for t in range(T):
            for pair in self.BONE_CHAIN_ORDER:
                parent_idx, child_idx = pair
                target_len = medians[pair]
                if target_len <= 0:
                    continue

                parent = result[t, parent_idx]
                child = result[t, child_idx]
                vec = child - parent
                current_len = np.linalg.norm(vec)

                if current_len > 0.001:
                    new_child = parent + vec * (target_len / current_len)
                    delta = new_child - child
                    result[t, child_idx] = new_child
                    # Cascade to direct chain joints only
                    for di in self.DOWNSTREAM_KEYPOINTS.get(child_idx, []):
                        if di < result.shape[1]:
                            result[t, di] += delta

        return result

    def _apply_butterworth(
        self, keypoints: np.ndarray, fps: float
    ) -> np.ndarray:
        """
        Apply Butterworth low-pass filter with axis-specific cutoffs.

        The depth axis (RTMPose3D Y) uses a much lower cutoff than
        lateral/vertical because RTMPose3D's depth is ~3x noisier.
        This stabilizes the forward/backward position without
        over-smoothing the more accurate lateral and vertical axes.

        Args:
            keypoints: (T, K, 3) keypoint sequence
            fps: Frame rate in Hz

        Returns:
            Smoothed sequence
        """
        from scipy.signal import butter, filtfilt

        nyquist = fps / 2
        result = keypoints.copy()
        T, K, _ = result.shape

        min_samples = 3 * self.filter_order + 1
        if T < min_samples:
            return keypoints

        # Per-axis cutoff: depth gets heavier smoothing
        cutoffs = [self.filter_cutoff] * 3
        cutoffs[self.DEPTH_AXIS] = self.depth_cutoff

        for dim in range(3):
            cutoff = cutoffs[dim]
            if cutoff >= nyquist or cutoff <= 0:
                continue
            normalized = cutoff / nyquist
            b, a = butter(self.filter_order, normalized, btype="low")
            for k in range(K):
                try:
                    result[:, k, dim] = filtfilt(b, a, result[:, k, dim])
                except Exception:
                    pass

        return result

    def detect_outliers(
        self,
        keypoints: np.ndarray,
        threshold: float = 3.0,
    ) -> np.ndarray:
        """
        Detect outlier frames based on velocity.

        Args:
            keypoints: (T, K, 3) keypoint sequence
            threshold: Z-score threshold for outlier detection

        Returns:
            (T,) boolean mask where True indicates outlier
        """
        T = keypoints.shape[0]
        if T < 3:
            return np.zeros(T, dtype=bool)

        velocities = np.diff(keypoints, axis=0)
        velocity_magnitudes = np.linalg.norm(velocities, axis=2)
        avg_velocity = np.mean(velocity_magnitudes, axis=1)

        mean_vel = np.mean(avg_velocity)
        std_vel = np.std(avg_velocity)

        if std_vel < 1e-6:
            return np.zeros(T, dtype=bool)

        z_scores = np.abs(avg_velocity - mean_vel) / std_vel

        outliers = np.zeros(T, dtype=bool)
        outlier_indices = np.where(z_scores > threshold)[0]
        for idx in outlier_indices:
            outliers[idx] = True
            if idx + 1 < T:
                outliers[idx + 1] = True

        return outliers

    def fix_left_right_swaps(
        self, keypoints: np.ndarray
    ) -> np.ndarray:
        """
        Detect and fix left/right body part swapping.

        Args:
            keypoints: (T, K, 3) keypoint sequence

        Returns:
            Corrected sequence
        """
        result = keypoints.copy()
        T = result.shape[0]

        # Left/right pairs using COCO-WholeBody indices
        lr_pairs = [
            (11, 12),  # hips
            (13, 14),  # knees
            (15, 16),  # ankles
            (5, 6),    # shoulders
            (7, 8),    # elbows
            (9, 10),   # wrists
        ]

        for t in range(1, T):
            swap_detected = False

            for left_idx, right_idx in lr_pairs:
                prev_left = result[t - 1, left_idx]
                prev_right = result[t - 1, right_idx]
                curr_left = result[t, left_idx]
                curr_right = result[t, right_idx]

                vel_normal = (
                    np.linalg.norm(curr_left - prev_left)
                    + np.linalg.norm(curr_right - prev_right)
                )
                vel_swapped = (
                    np.linalg.norm(curr_right - prev_left)
                    + np.linalg.norm(curr_left - prev_right)
                )

                if vel_swapped < vel_normal * 0.5:
                    swap_detected = True
                    break

            if swap_detected:
                for left_idx, right_idx in lr_pairs:
                    result[t, [left_idx, right_idx]] = result[t, [right_idx, left_idx]]

        return result
