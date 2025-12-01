import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F


class RGBVideoTransform:
    """Applies spatial transforms to RGB video clips for 3D CNNs, e.g., R3D."""

    def __init__(self, mode="train", crop_size=112, resize_size=128):
        self.mode = mode
        self.crop_size = crop_size
        self.resize_size = resize_size

        # Kinetics normalization stats
        self.mean = [0.43216, 0.394666, 0.37645]
        self.std = [0.22803, 0.22145, 0.216989]

    def __call__(self, frames):
        """Transform video clip from (T,H,W,C) numpy array to tensor (C,T,H,W)."""
        # Frames is numpy array of shape (T, H, W, C)
        clip = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0

        if self.mode == "train":
            clip = self._train_transform(clip)
        else:
            clip = self._val_transform(clip)

        # Normalize
        clip = self._normalize(clip)

        return clip

    def _train_transform(self, clip):
        """Apply training augmentations."""
        C, T, H, W = clip.shape
        first_frame = clip[:, 0, :, :]  # (C, H, W)

        # RandomResizedCrop: apply same crop to all frames
        i, j, h, w = T.RandomResizedCrop.get_params(
            first_frame,
            scale=(0.08, 1.0),
            ratio=(0.9, 1.1),  # More aggressive for grainy images
        )

        # Apply crop and resize to entire clip
        clip = F.resized_crop(clip, i, j, h, w, [self.crop_size, self.crop_size])

        # RandomHorizontalFlip: flip entire clip
        if torch.rand(1) < 0.5:
            clip = F.hflip(clip)

        # ColorJitter: apply same jitter to all frames
        if torch.rand(1) < 0.8:
            brightness = 0.3
            contrast = 0.3
            saturation = 0.2
            jitter = T.ColorJitter(
                brightness=brightness, contrast=contrast, saturation=saturation
            )
            # Apply frame-by-frame (ColorJitter doesn't support 4D)
            clip = torch.stack([jitter(clip[:, i]) for i in range(T)], dim=1)

        return clip

    def _val_transform(self, clip):
        """Apply validation/test transforms."""
        # This already upsamples: 64×64 → 128×128 → center crop to 112×112
        clip = F.resize(clip, [self.resize_size, self.resize_size])
        clip = F.center_crop(clip, [self.crop_size, self.crop_size])
        return clip

    def _normalize(self, clip):
        """Normalize with Kinetics mean/std."""
        for c in range(3):
            clip[c] = (clip[c] - self.mean[c]) / self.std[c]
        return clip


class SpatialStreamTransform:
    """Transforms for Two-Stream spatial pathway (RGB frames)."""

    def __init__(self, mode="train", crop_size=224):
        """
        TODO: Implement for Two-Stream network.
        - Similar to RGBVideoTransform but may use single frame or sparse sampling
        - Standard 2D image augmentations
        """
        self.mode = mode
        self.crop_size = crop_size
        raise NotImplementedError("SpatialStreamTransform not yet implemented")

    def __call__(self, frames):
        raise NotImplementedError


class TemporalStreamTransform:
    """Transforms for Two-Stream temporal pathway (optical flow)."""

    def __init__(self, mode="train", crop_size=224, flow_frames=10):
        """
        TODO: Implement for Two-Stream network.
        - Input: Optical flow computed from consecutive RGB frames
        - Output: (2*flow_frames, H, W) stacked flow (x and y components)
        - No ColorJitter (flow isn't RGB)
        - Custom normalization for flow values (typically [-20, 20])
        """
        self.mode = mode
        self.crop_size = crop_size
        self.flow_frames = flow_frames
        raise NotImplementedError("TemporalStreamTransform not yet implemented")

    def __call__(self, flow_frames):
        raise NotImplementedError
