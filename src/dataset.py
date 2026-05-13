import os
import json
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

class NeRFDataset(Dataset):
    def __init__(self, data_dir, split='train', H=800, W=800, device='cuda'):
        self.data_dir = data_dir
        self.split = split
        self.H, self.W = H, W
        self.device = device
        
        json_path = os.path.join(data_dir, f'transforms_{split}.json')
        with open(json_path, 'r') as f:
            self.meta = json.load(f)
        
        if 'camera_angle_x' in self.meta:
            self.fl_x = 0.5 * W / np.tan(0.5 * self.meta['camera_angle_x'])
            self.fl_y = self.fl_x
        else:
            self.fl_x = self.meta.get('fl_x', 1111.11)
            self.fl_y = self.meta.get('fl_y', 1111.11)
        
        self.frames = self.meta['frames']
        self.n_frames = len(self.frames)
        
        self.all_rays = []
        self.all_rgbs = []
        self._preload()
        
    def _preload(self):
        print(f"📂 Preprocessing {self.split} rays...")
        
        c2w_correction = np.array([
            [1,  0,  0, 0],
            [0, -1,  0, 0],
            [0,  0, -1, 0],
            [0,  0,  0, 1]
        ])
        
        for frame_data in self.frames:
            img_path = os.path.join(self.data_dir, frame_data['file_path'])
            if not img_path.endswith('.png'):
                img_path += '.png'
            img = Image.open(img_path).resize((self.W, self.H))
            img = np.array(img).astype(np.float32) / 255.0
            
            c2w = np.array(frame_data['transform_matrix'])
            c2w = c2w @ c2w_correction
            c2w[:3, 1:3] *= -1
            
            cam_origin = torch.from_numpy(c2w[:3, 3]).float().to(self.device)
            cam_rot = torch.from_numpy(c2w[:3, :3]).float().to(self.device)
            
            i, j = torch.meshgrid(
                torch.linspace(0, self.W-1, self.W),
                torch.linspace(0, self.H-1, self.H), 
                indexing='xy'
            )
            i, j = i.to(self.device), j.to(self.device)
            
            dirs = torch.stack([
                (i - self.W * 0.5) / self.fl_x,
                -(j - self.H * 0.5) / self.fl_y,
                -torch.ones_like(i)
            ], dim=-1)
            
            dirs_world = (cam_rot @ dirs[..., None]).squeeze(-1)
            dirs_world = dirs_world / torch.norm(dirs_world, dim=-1, keepdim=True)
            origins = cam_origin.expand(dirs_world.shape)
            
            origins_flat = origins.view(-1, 3)
            dirs_flat = dirs_world.view(-1, 3)
            rgbs_flat = torch.from_numpy(img).to(self.device).view(-1, 3)
            
            self.all_rays.append(torch.cat([origins_flat, dirs_flat, rgbs_flat], dim=-1))
            self.all_rgbs.append(rgbs_flat)
        
        self.all_rays = torch.cat(self.all_rays, dim=0)
        print(f"✅ Total rays: {self.all_rays.shape[0]:,} | Shape: {self.all_rays.shape}")
        
    def __len__(self):
        return self.all_rays.shape[0]
    
    def __getitem__(self, idx):
        ray = self.all_rays[idx]
        return ray[:3], ray[3:6], ray[6:9] 