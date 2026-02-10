"""
MotionBERT 2D-to-3D lifting inference wrapper.

Takes 2D keypoints in H36M-17 format, outputs 3D keypoints with proper depth.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from easydict import EasyDict
from torch import nn

# Add MotionBERT to path
MOTIONBERT_DIR = Path(__file__).parent.parent / "models" / "MotionBERT"
sys.path.insert(0, str(MOTIONBERT_DIR))

from lib.model.DSTformer import DSTformer


class MotionBERTLifter:
    """Lifts 2D keypoints to 3D using MotionBERT DSTformer."""

    DEFAULT_CONFIG = MOTIONBERT_DIR / "configs" / "pose3d" / "MB_ft_h36m_global_lite.yaml"
    DEFAULT_CHECKPOINT = (
        MOTIONBERT_DIR / "checkpoint" / "pose3d"
        / "FT_MB_lite_MB_ft_h36m_global_lite" / "best_epoch.bin"
    )

    def __init__(
        self,
        config_path: str = None,
        checkpoint_path: str = None,
        device: str = "cuda:0",
    ):
        config_path = Path(config_path or self.DEFAULT_CONFIG)
        checkpoint_path = Path(checkpoint_path or self.DEFAULT_CHECKPOINT)

        with open(config_path) as f:
            self.args = EasyDict(yaml.safe_load(f))

        self.device = torch.device(device)
        self.maxlen = self.args.maxlen  # 243

        # Build model
        self.model = DSTformer(
            dim_in=3,
            dim_out=3,
            dim_feat=self.args.dim_feat,
            dim_rep=self.args.dim_rep,
            depth=self.args.depth,
            num_heads=self.args.num_heads,
            mlp_ratio=self.args.mlp_ratio,
            norm_layer=nn.LayerNorm,
            maxlen=self.args.maxlen,
            num_joints=self.args.num_joints,
            att_fuse=self.args.att_fuse,
        )

        # Load checkpoint (strip DataParallel 'module.' prefix)
        ckpt = torch.load(str(checkpoint_path), map_location="cpu")
        state_dict = {k.replace("module.", ""): v for k, v in ckpt["model_pos"].items()}
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        self.model.to(self.device)

        print(f"  MotionBERT loaded ({sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M params)")

    # H36M training camera reference values
    # H36M uses 4 cameras with focal lengths ~1145-1150px on ~1000x1000 images
    H36M_FOCAL = 1145.0
    H36M_SCALE = 500.0  # min(1000, 1000) / 2

    def lift(
        self,
        keypoints_2d: np.ndarray,
        scores: np.ndarray,
        image_width: int,
        image_height: int,
        flip_augment: bool = True,
        focal_length: float = None,
    ) -> np.ndarray:
        """
        Lift 2D keypoints to 3D.

        Args:
            keypoints_2d: (T, 17, 2) H36M 2D keypoints in pixel coordinates
            scores: (T, 17) confidence scores
            image_width: video width in pixels
            image_height: video height in pixels
            flip_augment: use horizontal flip test-time augmentation
            focal_length: camera focal length in pixels (None = no correction).
                Corrects for FOV mismatch between the recording camera and
                H36M training cameras (~1145px on 1000x1000).

        Returns:
            keypoints_3d: (T, 17, 3) 3D keypoints in meters (camera coordinate system)
        """
        T = keypoints_2d.shape[0]

        # Normalize: center at image center, divide by min(w,h)/2
        # This preserves aspect ratio — critical for correct 3D estimation.
        # For 1920x1080: X range [-1.78, 1.78], Y range [-1, 1]
        kpts_norm = keypoints_2d.copy().astype(np.float32)
        scale = min(image_width, image_height) / 2.0
        kpts_norm[..., 0] = (kpts_norm[..., 0] - image_width / 2.0) / scale
        kpts_norm[..., 1] = (kpts_norm[..., 1] - image_height / 2.0) / scale

        # Focal length correction: rescale normalized coords to match H36M
        # training distribution. A wider FOV camera (lower focal length)
        # produces normalized coords that are too spread out, causing
        # MotionBERT to misinterpret depth and produce forward lean.
        if focal_length is not None:
            correction = (focal_length / scale) / (self.H36M_FOCAL / self.H36M_SCALE)
            kpts_norm[..., :2] *= correction
            print(f"  Focal correction: f={focal_length:.0f}px, factor={correction:.3f}")

        # Stack with confidence: (T, 17, 3) = [x_norm, y_norm, confidence]
        kpts_input = np.concatenate([kpts_norm, scores[..., np.newaxis]], axis=-1)

        # Process in sliding windows of maxlen
        stride = self.args.get("data_stride", 81)
        output_3d = np.zeros((T, 17, 3), dtype=np.float32)
        counts = np.zeros(T, dtype=np.float32)

        for start in range(0, max(1, T - self.maxlen + 1), stride):
            end = min(start + self.maxlen, T)
            clip = kpts_input[start:end]

            # Pad if needed
            actual_len = clip.shape[0]
            if actual_len < self.maxlen:
                pad = np.zeros((self.maxlen - actual_len, 17, 3), dtype=np.float32)
                clip = np.concatenate([clip, pad], axis=0)

            pred = self._infer_clip(clip, flip_augment)
            output_3d[start:start + actual_len] += pred[:actual_len]
            counts[start:start + actual_len] += 1

        # Handle uncovered tail frames with a window ending at T
        uncovered = counts == 0
        if np.any(uncovered):
            tail_start = max(0, T - self.maxlen)
            clip = kpts_input[tail_start:T]
            actual_len = clip.shape[0]
            if actual_len < self.maxlen:
                pad = np.zeros((self.maxlen - actual_len, 17, 3), dtype=np.float32)
                clip = np.concatenate([clip, pad], axis=0)
            pred = self._infer_clip(clip, flip_augment)
            # Only fill uncovered frames
            for i in range(actual_len):
                frame_idx = tail_start + i
                if uncovered[frame_idx]:
                    output_3d[frame_idx] = pred[i]
                    counts[frame_idx] = 1

        # Average overlapping predictions
        output_3d /= np.maximum(counts[:, np.newaxis, np.newaxis], 1)

        return output_3d

    def _infer_clip(self, clip: np.ndarray, flip_augment: bool) -> np.ndarray:
        """Run inference on a single clip of maxlen frames."""
        batch = torch.from_numpy(clip).float().unsqueeze(0).to(self.device)  # (1, T, 17, 3)

        with torch.no_grad():
            pred = self.model(batch)  # (1, T, 17, 3)

            if flip_augment:
                # Horizontal flip augmentation
                batch_flip = batch.clone()
                batch_flip[..., 0] *= -1  # flip X
                # Swap left/right joints
                lr_pairs = [(1, 4), (2, 5), (3, 6), (11, 14), (12, 15), (13, 16)]
                for l, r in lr_pairs:
                    batch_flip[:, :, [l, r]] = batch_flip[:, :, [r, l]]

                pred_flip = self.model(batch_flip)
                pred_flip[..., 0] *= -1  # flip back
                for l, r in lr_pairs:
                    pred_flip[:, :, [l, r]] = pred_flip[:, :, [r, l]]

                pred = (pred + pred_flip) / 2

        return pred.squeeze(0).cpu().numpy()  # (T, 17, 3)
