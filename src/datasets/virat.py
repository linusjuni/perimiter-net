"""VIRAT surveillance dataset for activity recognition."""

import os
import yaml
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class VIRATDataset(Dataset):
    """VIRAT video dataset for activity classification."""
    
    ACTIVITIES = [
        'activity_walking',
        'activity_running',
        'activity_standing',
        'activity_carrying',
        'activity_gesturing',
        'Entering',
        'Exiting'
    ]
    
    def __init__(self, root_dir, split='train', clip_len=16, transform=None):
        """Initialize VIRAT dataset."""
        self.root_dir = root_dir
        self.split = split
        self.clip_len = clip_len
        self.transform = transform
        
        self.video_dir = os.path.join(root_dir, 'videos')
        self.annot_dir = os.path.join(root_dir, 'annotations')
        
        self.samples = self._load_annotations()
        self.label_to_idx = {label: idx for idx, label in enumerate(self.ACTIVITIES)}
        
    def _load_annotations(self):
        """Parse activities.yml files and return list of (video_path, activity, start_frame, end_frame)."""
        samples = []
        split_file = os.path.join(self.root_dir, 'splits', f'{self.split}.txt')
        
        with open(split_file, 'r') as f:
            video_names = [line.strip() for line in f.readlines()]
        
        for video_name in video_names:
            annot_path = os.path.join(self.annot_dir, f'{video_name}.activities.yml')
            video_path = os.path.join(self.video_dir, f'{video_name}.mp4')
            
            if not os.path.exists(annot_path):
                continue
            
            with open(annot_path, 'r') as f:
                data = yaml.safe_load_all(f)
                
                for item in data:
                    if 'act' not in item:
                        continue
                    
                    act_data = item['act']
                    
                    if 'actors' in act_data and len(act_data['actors']) > 1:
                        continue
                    
                    activity_dict = act_data.get('act2', {})
                    activity = list(activity_dict.keys())[0] if activity_dict else None
                    
                    if activity not in self.ACTIVITIES:
                        continue
                    
                    timespan = act_data.get('timespan', [{}])[0]
                    tsr = timespan.get('tsr0', [])
                    
                    if len(tsr) != 2:
                        continue
                    
                    start_frame, end_frame = tsr[0], tsr[1]
                    
                    if end_frame - start_frame >= self.clip_len:
                        samples.append((video_path, activity, start_frame, end_frame))
        
        return samples
    
    def _load_video_clip(self, video_path, start_frame, end_frame):
        """Load video clip with random temporal sampling."""
        cap = cv2.VideoCapture(video_path)
        
        max_start = end_frame - self.clip_len
        if self.split == 'train':
            clip_start = np.random.randint(start_frame, max(start_frame + 1, max_start))
        else:
            clip_start = start_frame
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start)
        
        frames = []
        for _ in range(self.clip_len):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        
        while len(frames) < self.clip_len:
            frames.append(frames[-1])
        
        return np.array(frames)
    
    def __len__(self):
        """Return dataset size."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Return (video_clip, label) tuple."""
        video_path, activity, start_frame, end_frame = self.samples[idx]
        
        frames = self._load_video_clip(video_path, start_frame, end_frame)
        label = self.label_to_idx[activity]
        
        if self.transform:
            frames = self.transform(frames)
        else:
            frames = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0
        
        return frames, label