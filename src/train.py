import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import sys

sys.path.append('src')
from dataset import NeRFDataset
from model import create_nerf_model
from renderer import render_batch

def compute_psnr(img1, img2):
    """Hitung PSNR antara dua tensor RGB [N, 3] dalam range [0, 1]"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return torch.tensor(100.0, device=img1.device)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def train_nerf(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Device: {device}")
    print(f"️  Config: {config}")

    print("📂 Loading training dataset...")
    train_dataset = NeRFDataset(
        config['data_dir'], split='train',
        H=config['H'], W=config['W'], device=device
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], 
        shuffle=True, pin_memory=torch.cuda.is_available()
    )
    train_iter = iter(train_loader)

    print("🧠 Initializing NeRF model...")
    model = create_nerf_model(device=device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model parameters: {total_params:,} | Trainable: {total_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config['lr_decay'])

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    print("🔄 Starting training...")
    start_time = time.time()
    global_step = 0
    
    pbar = tqdm(range(config['iterations']), desc="Training", unit="iter")
    for iteration in pbar:
        model.train()
        
        try:
            rays_o, rays_d, target_rgb = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            rays_o, rays_d, target_rgb = next(train_iter)
            
        rays_o, rays_d, target_rgb = (
            rays_o.to(device, non_blocking=True),
            rays_d.to(device, non_blocking=True),
            target_rgb.to(device, non_blocking=True)
        )
        
        # Forward Pass: Render
        ret = render_batch(
            rays_o, rays_d, model,
            chunk=config['chunk'],
            N_samples=config['N_samples'],
            N_samples_hier=config['N_samples_hier'],
            near=config['near'], far=config['far'],
            perturb=True 
        )
        
        rgb_map = ret['rgb_map']
        rgb_map_coarse = ret['rgb_map_coarse']
        
        # Loss: MSE (Fine + weighted Coarse)
        loss_fine = torch.mean((rgb_map - target_rgb) ** 2)
        loss_coarse = torch.mean((rgb_map_coarse - target_rgb) ** 2)
        total_loss = loss_fine + 0.1 * loss_coarse
        
        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Logging
        if iteration % config['log_interval'] == 0:
            with torch.no_grad():
                psnr = compute_psnr(rgb_map, target_rgb)
            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'psnr': f"{psnr.item():.2f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })
        
        # Checkpoint Saving
        if iteration % config['save_interval'] == 0 and iteration > 0:
            torch.save({
                'iteration': iteration,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss.item(),
            }, f'checkpoints/nerf_iter_{iteration}.pt')
            
        global_step += 1

    torch.save(model.state_dict(), 'checkpoints/nerf_final.pt')
    elapsed = time.time() - start_time
    print(f"\n🏁 Training selesai! Waktu: {elapsed/60:.2f} menit")
    print(" Model final tersimpan di: checkpoints/nerf_final.pt")

if __name__ == '__main__':
    config = {
        'data_dir': 'data/blender/chair',
        'H': 100, 'W': 100,           #Resolution
        'batch_size': 2048,          
        'iterations': 300,           
        'lr': 5e-4,                    
        'lr_decay': 0.999,             
        'chunk': 2048,                 
        'N_samples': 64,            
        'N_samples_hier': 0,           #Hierarchical
        'near': 2.0, 'far': 6.0,
        'log_interval': 30,            
        'save_interval': 100           
    }
    
    train_nerf(config)