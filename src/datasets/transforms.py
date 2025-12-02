import torch
import torchvision.transforms as transforms
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
        i, j, h, w = transforms.RandomResizedCrop.get_params(
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
            jitter = transforms.ColorJitter(
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


class SobelMotionTransform:
    """
    Computes Temporal and Spatial derivatives of motion on-the-fly.

    Logic:
    1. Grayscale & Resize
    2. Temporal Derivative (dt)
    3. Spatial Derivatives of dt (dx, dy)
    4. Stack [dx, dy, dt] -> (3, T, H, W)
    """

    def __init__(self, mode="train", crop_size=112, resize_size=128):
        self.mode = mode
        self.crop_size = crop_size
        self.resize_size = resize_size

        # Sobel Kernels (3x3)
        self.sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        ).view(1, 1, 3, 3)

    def __call__(self, frames):
        # 1. Convert to Tensor (T, H, W, C) -> (T, C, H, W)
        clip = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0

        # 2. Convert to Grayscale (Weighted method)
        gray = 0.299 * clip[:, 0:1] + 0.587 * clip[:, 1:2] + 0.114 * clip[:, 2:3]

        # 3. Spatial Resizing & Cropping (Internal)
        # We do this BEFORE Sobel to save computation
        if self.mode == "train":
            # Random Crop
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                gray[0], scale=(0.6, 1.0), ratio=(0.9, 1.1)
            )
            gray = transforms.resized_crop(gray, i, j, h, w, [self.crop_size, self.crop_size])
            # Random Flip
            if torch.rand(1) < 0.5:
                gray = transforms.hflip(gray)
        else:
            # Deterministic Crop
            gray = F.interpolate(
                gray,
                size=(self.resize_size, self.resize_size),
                mode="bilinear",
                align_corners=False,
            )
            gray = transforms.center_crop(gray, [self.crop_size, self.crop_size])

        # 4. Temporal Derivative (dt)
        # Center zero-motion at 0.0 (Differences range -1 to 1)
        # We perform [t] - [t-1]
        dt = torch.zeros_like(gray)
        dt[1:] = gray[1:] - gray[:-1]

        # 5. Spatial Derivatives (dx, dy)
        # We apply Sobel to the *Difference* frame
        # This highlights the EDGES of the moving object
        dx = F.conv2d(dt, self.sobel_x, padding=1)
        dy = F.conv2d(dt, self.sobel_y, padding=1)

        # 6. Stack Channels: [dx, dy, dt]
        # We take absolute value because R3D filters usually expect positive activation features
        # and "moving left" vs "moving right" (negative vs positive) are both "motion".
        motion_clip = torch.cat([torch.abs(dx), torch.abs(dy), torch.abs(dt)], dim=1)

        # Output: (T, 3, H, W) -> Permute to (C, T, H, W) for R3D
        return motion_clip.permute(1, 0, 2, 3)
