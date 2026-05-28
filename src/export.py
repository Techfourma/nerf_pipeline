# src/export.py
import os
import torch
import numpy as np
import json
import time
from PIL import Image
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import lpips
import sys

sys.path.append('src')
from dataset import NeRFDataset
from model import create_nerf_model
from renderer import render_batch

def export_nerf():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" Device: {device}")

    print("📦 Loading trained model...")
    model = create_nerf_model(device=device)
    ckpt_path = 'checkpoints/nerf_final.pt'
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"❌ Checkpoint tidak ditemukan: {ckpt_path}\nJalankan training terlebih dahulu.")
        
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # 2. Konfigurasi
    config = {
        'data_dir': 'data/blender/chair',
        'H': 100, 'W': 100,
        'chunk': 2048,
        'N_samples': 64,
        'near': 2.0, 'far': 6.0
    }

    # 3. Load Test Dataset
    print(" Loading test dataset...")
    test_dataset = NeRFDataset(config['data_dir'], split='test', H=config['H'], W=config['W'], device=device)
    rays_per_img = config['H'] * config['W']
    n_images = test_dataset.n_frames

    print("⚙️ Initializing LPIPS network (mungkin butuh waktu di CPU)...")
    lpips_fn = lpips.LPIPS(net='vgg').to(device).eval()

    os.makedirs('results/frames', exist_ok=True)
    metrics_list = []

    print(f"🎨 Rendering {n_images} test views & menghitung metrik...")
    start_time = time.time()

    for i in tqdm(range(n_images), desc="Exporting"):
        idx_start = i * rays_per_img
        idx_end = idx_start + rays_per_img

        # Ambil rays untuk 1 gambar
        rays_o = test_dataset.all_rays[idx_start:idx_end, 0:3]
        rays_d = test_dataset.all_rays[idx_start:idx_end, 3:6]
        gt_rgb = test_dataset.all_rays[idx_start:idx_end, 6:9]

        # Render
        with torch.no_grad():
            ret = render_batch(rays_o, rays_d, model, chunk=config['chunk'],
                               N_samples=config['N_samples'], N_samples_hier=0,
                               near=config['near'], far=config['far'], perturb=False)
        pred_rgb = ret['rgb_map']

        # Reshape ke [H, W, 3] untuk metrik
        gt_np = gt_rgb.cpu().numpy().reshape(config['H'], config['W'], 3)
        pred_np = pred_rgb.cpu().numpy().reshape(config['H'], config['W'], 3)

        # Hitung Metrik Kuantitatif
        p = psnr(gt_np, pred_np, data_range=1.0)
        s = ssim(gt_np, pred_np, multichannel=True, data_range=1.0)

        # LPIPS (memerlukan format [Batch, Channel, H, W])
        gt_t = torch.from_numpy(gt_np).permute(2, 0, 1).unsqueeze(0).to(device)
        pred_t = torch.from_numpy(pred_np).permute(2, 0, 1).unsqueeze(0).to(device)
        l = lpips_fn(gt_t, pred_t).item()

        metrics_list.append({
            'frame': i,
            'psnr': float(p),
            'ssim': float(s),
            'lpips': float(l)
        })

        Image.fromarray((pred_np * 255).astype(np.uint8)).save(f'results/frames/frame_{i:04d}.png')

    # 6. Simpan Summary & Metrik Rata-rata
    avg_psnr = np.mean([m['psnr'] for m in metrics_list])
    avg_ssim = np.mean([m['ssim'] for m in metrics_list])
    avg_lpips = np.mean([m['lpips'] for m in metrics_list])

    summary = {
        'training_info': {'iterations': 300, 'H': 100, 'W': 100, 'device': str(device)},
        'average_metrics': {
            'psnr': round(avg_psnr, 2),
            'ssim': round(avg_ssim, 4),
            'lpips': round(avg_lpips, 4)
        },
        'per_frame_metrics': metrics_list,
        'export_time_sec': round(time.time() - start_time, 2)
    }

    with open('results/metrics.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ EXPORT SELESAI!")
    print(f"️ Waktu eksekusi: {summary['export_time_sec']} detik")
    print(f" Average PSNR: {avg_psnr:.2f} dB")
    print(f" Average SSIM: {avg_ssim:.4f}")
    print(f"📊 Average LPIPS: {avg_lpips:.4f}")
    print(f"💾 Sequence gambar: results/frames/ ({n_images} files)")
    print(f"📄 File metrik: results/metrics.json")

if __name__ == '__main__':
    export_nerf()