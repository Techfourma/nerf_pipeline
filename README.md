# Implementasi Neural Radiance Fields (NeRF) untuk Novel View Synthesis dengan Visualisasi Interaktif

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=flat&logo=pytorch)](https://pytorch.org/)
[![Processing](https://img.shields.io/badge/Visualization-Processing-006699?style=flat&logo=processing)](https://processing.org/)
[![License](https://img.shields.io/badge/License-Academic-000000?style=flat)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Completed-success?style=flat)]()

> **Penulis:** [Jundulloh rizki ananda](https://github.com/JundiLesmana)  
> **Institusi:** [Universitas Pamulang](https://unpam.ac.id/)  
> **Mata Kuliah/Proyek:** Computer Graphics  
> **Tanggal:** 30 Mei 2026

---

## 📋 Daftar Isi
- [📖 Deskripsi Project](#-deskripsi-project)
- [🎯 Tujuan & Kontribusi](#-tujuan--kontribusi)
- [✨ Fitur Utama](#-fitur-utama)
- [📂 Struktur Project](#-struktur-project)
- [🛠️ Prasyarat Sistem](#-prasyarat-sistem)
- [🚀 Instalasi & Setup](#-instalasi--setup)
- [▶️ Cara Menjalankan](#-cara-menjalankan)
- [📊 Hasil & Evaluasi](#-hasil--evaluasi)
- [🔧 Konfigurasi & Parameter](#-konfigurasi--parameter)
- [🐛 Troubleshooting](#-troubleshooting)
- [📚 Referensi Akademik](#-referensi-akademik)
- [🤝 Kontribusi](#-kontribusi)
- [📜 Lisensi](#-lisensi)
- [✉️ Kontak](#-kontak)

---

## 📖 Deskripsi Project

Repository ini berisi implementasi **Neural Radiance Fields (NeRF)** secara *end-to-end* menggunakan framework **PyTorch** untuk tugas *Novel View Synthesis* (sintesis pandangan baru). Sistem ini merekonstruksi representasi 3D kontinu dari sekumpulan gambar 2D dengan pose kamera yang diketahui, kemudian menghasilkan pandangan baru dari sudut yang tidak ada dalam dataset training.

Selain pipeline training berbasis Python, project ini juga menyertakan **aplikasi visualisasi interaktif** berbasis Processing (Java) yang memungkinkan:
- Navigasi frame hasil rendering secara dinamis
- Rotasi 3D dan zoom untuk inspeksi visual
- Display metrik evaluasi (PSNR & SSIM) secara real-time
- Kontrol playback untuk animasi sequence

Project ini dikembangkan sebagai bagian dari tugas akademik untuk mendemonstrasikan pemahaman mendalam tentang neural rendering, volume rendering, positional encoding, dan evaluasi kualitas rekonstruksi 3D.

---

## 🎯 Tujuan & Kontribusi

### Tujuan Penelitian
1. Mengimplementasikan pipeline NeRF lengkap mulai dari preprocessing dataset hingga export hasil rendering.
2. Mengevaluasi kualitas rekonstruksi menggunakan metrik standar akademik: PSNR dan SSIM.
3. Mengembangkan antarmuka visualisasi interaktif untuk analisis kualitatif hasil render.
4. Menganalisis trade-off antara kualitas output dan beban komputasi dalam konteks implementasi terjangkau.

### Kontribusi Project
- ✅ Implementasi NeRF modular dengan arsitektur MLP + positional encoding
- ✅ Script training yang kompatibel dengan CPU/GPU (Colab/Kaggle ready)
- ✅ Evaluasi metrik otomatis dengan export ke JSON
- ✅ Aplikasi Processing untuk visualisasi interaktif dengan kontrol UI lengkap
- ✅ Dokumentasi lengkap dengan panduan reproduksi hasil

---

## ✨ Fitur Utama

| Fitur | Deskripsi | Status |
|-------|-----------|--------|
| **Ray Generation** | Generasi sinar dari pose kamera dengan koreksi koordinat Blender→OpenCV | ✅ |
| **Positional Encoding** | Encoding frekuensi untuk posisi (L=10) dan arah pandang (L=4) | ✅ |
| **MLP Architecture** | 8-layer MLP dengan skip connection dan ReLU activation | ✅ |
| **Volume Rendering** | Integrasi numerik dengan stratified sampling & alpha compositing | ✅ |
| **Chunked Processing** | Batch processing untuk efisiensi memori GPU/CPU | ✅ |
| **Metric Evaluation** | Perhitungan PSNR & SSIM otomatis per-frame dan rata-rata | ✅ |
| **Interactive Visualization** | Aplikasi Processing dengan navigasi frame, rotasi, zoom, metrics overlay | ✅ |
| **Checkpoint Management** | Save/load model weights untuk resume training | ✅ |

---

## 📂 Struktur Project

```text
NERF_Project/
│
├── 📁 src/                         
│   ├── dataset.py                   # Loading dataset, pose correction, ray generation
│   ├── model.py                     # PositionalEncoding class, NeRFModel MLP architecture
│   ├── renderer.py                  # Volume rendering functions: raw2outputs, render_rays, render_batch
│   ├── train.py                     # Training loop, optimizer, scheduler, logging
│   └── export.py                    # Test rendering, metric calculation, JSON export
│
├── 📁 processing_sketch/            # Aplikasi visualisasi (Processing/Java)
│   └── NeRF_Visualizer.pde          # Main sketch: load frames, display metrics, UI controls
│
├── 📁 data/                         # Dataset (tidak di-commit, download terpisah)
│   └── blender/chair/
│       ├── transforms_train.json    # Training poses & file paths
│       ├── transforms_test.json     # Test poses & file paths
│       ├── train/                   # Training images (PNG)
│       └── test/                    # Test images (PNG)
│
├── 📁 results/                      # Output hasil export (di-generate otomatis)
│   ├── frames/                      # Rendered test images (frame_0000.png - frame_0039.png)
│   └── metrics.json                 # PSNR/SSIM per-frame + average metrics
│
├── 📁 checkpoints/                  # Saved model weights
│   └── nerf_final.pt                # Model state dict setelah training
│
├── 📁 documentation/                # Assets untuk makalah
│   ├── screenshots/                 # Screenshot aplikasi & console
│   ├── summary.txt                  # Ringkasan hasil metrik
│   └── demo_video.mp4               # Rekaman demo aplikasi
│
├── requirements.txt                 # Python dependencies
├── README.md                         
└── LICENSE                          # Lisensi penggunaan
```

---

## 🛠️ Prasyarat Sistem

### Hardware Minimum
| Komponen | Spesifikasi Minimum | Rekomendasi |
|----------|-------------------|-------------|
| **CPU** | 4-core Intel/AMD | 8-core+ |
| **RAM** | 8 GB | 16 GB+ |
| **GPU** | Tidak wajib (CPU-only support) | NVIDIA GTX 1060+ / T4 (Colab) |
| **Storage** | 2 GB free space | 10 GB+ untuk dataset full-res |

### Software Requirements
- **Python** ≥ 3.10 (tested on 3.12)
- **PyTorch** ≥ 2.0 (with CUDA support optional)
- **Processing IDE** ≥ 4.0 (untuk visualisasi)
- **Git** (untuk cloning repository)

### Environment Rekomendasi
Untuk hasil optimal, gunakan salah satu platform berikut:
- 🟢 **Google Colab (Free Tier)**: GPU T4, 12 GB RAM, 12 jam runtime
- 🟢 **Kaggle Notebooks**: GPU P100, 16 GB RAM, 30 jam/minggu
- 🟡 **Local CPU**: Lebih lambat (~25-30 menit untuk 300 iterasi), tapi tidak perlu setup cloud

---

## 🚀 Instalasi & Setup

### Langkah 1: Clone Repository
```bash
git clone https://github.com/[USERNAME]/[REPO_NAME].git
cd [REPO_NAME]
```

### Langkah 2: Setup Virtual Environment (Disarankan)
```bash
# Buat virtual environment
python -m venv nerf_env

# Aktifkan environment
# Windows:
nerf_env\Scripts\activate
# Linux/macOS:
source nerf_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Langkah 3: Persiapan Dataset
Download dataset **Blender Synthetic** dari sumber resmi:
```bash
# Buat folder dataset
mkdir -p data/blender

# Download (pilih salah satu metode)

# Metode 1: Direct download (jika tersedia)
wget -O data/blender/nerf_synthetic.zip "https://drive.google.com/uc?export=download&id=18JxhpWD-4ZmuFKLzKlAw-w5PpzZxXOcG"

# Metode 2: Manual download
# 1. Kunjungi: https://github.com/bmild/nerf#downloading-legacy-nerf-data
# 2. Download "nerf_synthetic.zip"
# 3. Extract ke: data/blender/

# Extract dataset
unzip data/blender/nerf_synthetic.zip -d data/blender/
rm data/blender/nerf_synthetic.zip
```

**Verifikasi struktur dataset:**
```bash
ls data/blender/chair/
# Expected output:
# transforms_train.json  transforms_test.json  train/  test/
```

### Langkah 4: Setup Processing (Untuk Visualisasi)
1. Download Processing IDE: https://processing.org/download
2. Install sesuai panduan OS Anda
3. Buka Processing → Menu: `Sketch → Import Library → Add Library...`
4. Cari **"ControlP5"** → Klik **Install**
5. Buka file: `processing_sketch/NeRF_Visualizer.pde`

---

## ▶️ Cara Menjalankan

### 🔄 Training Model
```bash
# Jalankan training dengan konfigurasi default
python src/train.py
```

**Output yang diharapkan:**
```
🚀 Device: cuda  # atau cpu
📂 Loading training dataset...
📂 Preprocessing train rays...
✅ Total rays: 200,000 | Shape: torch.Size([200000, 9])
🧠 Initializing NeRF model...
✅ Model parameters: 624,204 | Trainable: 624,204
🔄 Starting training...
Training: 100%|██████████| 300/300 [04:32<00:00, 1.10iter/s, loss=0.0069, psnr=22.05, lr=3.81e-04]
🏁 Training selesai! Waktu: 4.53 menit
✅ Checkpoint saved: checkpoints/nerf_final.pt
```

**Konfigurasi Training (opsional):**
Edit `src/train.py` bagian `config` untuk menyesuaikan:
```python
config = {
    'data_dir': 'data/blender/chair',  # Path dataset
    'H': 100, 'W': 100,                 # Resolusi render
    'batch_size': 2048,                 # Rays per iteration
    'iterations': 300,                  # Total training steps
    'lr': 5e-4,                         # Learning rate awal
    'lr_decay': 0.999,                  # Exponential decay gamma
    'chunk': 2048,                      # Rays per chunk (hemat memori)
    'N_samples': 64,                    # Samples per ray
    'near': 2.0, 'far': 6.0,            # Depth bounds
    'log_interval': 50,                 # Logging frequency
    'save_interval': 100                # Checkpoint frequency
}
```

### 📤 Export Hasil & Evaluasi Metrik
```bash
# Setelah training selesai, jalankan export
python src/export.py
```

**Output:**
```
📤 Exporting on cuda...
🎨 Rendering 40 test views...
Exporting: 100%|██████████| 40/40 [02:15<00:00, 3.39s/it]
✅ EXPORT SELESAI!
📊 Average PSNR: 22.01 dB
📊 Average SSIM: 0.8105
💾 Frames: results/frames/ (40 files)
📄 Metrics: results/metrics.json
```

### 🎨 Menjalankan Visualisasi Interaktif
1. Buka Processing IDE
2. File → Open → Pilih `processing_sketch/NeRF_Visualizer.pde`
3. **PENTING**: Edit path absolut di fungsi `loadMetrics()` dan `loadFrames()`:
   ```java
   // Ganti dengan path lokal Anda
   String path = "/home/jundi-lesmana/nerf_pipeline/results/metrics.json";
   String folder = "/home/jundi-lesmana/nerf_pipeline/results/frames/";
   ```
4. Klik **Run** (▶) atau tekan `Ctrl+R`

**Kontrol Aplikasi:**
| Input | Fungsi |
|-------|--------|
| 🖱️ Drag Kiri | Rotasi view 3D |
| 🖱️ Scroll Mouse | Zoom in/out |
| ⌨️ Arrow ←/→ | Navigasi frame sebelumnya/berikutnya |
| ⌨️ Spasi | Play/Pause animasi otomatis |
| ⌨️ + / - | Tingkatkan/turunkan kecepatan playback |
| 🎚️ Slider Frame | Navigasi manual ke frame tertentu |
| 🎚️ Slider Speed | Atur kecepatan animasi (0.1x - 3.0x) |
| 🎚️ Slider Zoom | Kontrol zoom level (0.1x - 3.0x) |

---

## 📊 Hasil & Evaluasi

### Konfigurasi Pengujian
| Parameter | Nilai |
|-----------|-------|
| Dataset | Blender Synthetic - Chair |
| Resolusi Input | 100 × 100 piksel |
| Iterasi Training | 300 |
| Samples per Ray | 64 |
| Positional Encoding | L=10 (posisi), L=4 (arah) |
| Hardware | NVIDIA T4 GPU (Kaggle) |
| Waktu Training | ~4-5 menit |

### Metrik Evaluasi
```json
{
  "average_metrics": {
    "psnr": 22.01,
    "ssim": 0.8105
  },
  "per_frame_metrics": [
    {"frame": 0, "psnr": 21.70, "ssim": 0.8000},
    {"frame": 1, "psnr": 22.52, "ssim": 0.8194},
    ...
    {"frame": 39, "psnr": 23.01, "ssim": 0.8382}
  ]
}
```

### Interpretasi Hasil
| Metrik | Nilai | Threshold Akademik | Status |
|--------|-------|-------------------|--------|
| **PSNR** | 22.01 dB | > 20 dB (valid) | ✅ Baik |
| **SSIM** | 0.8105 | > 0.75 (valid) | ✅ Baik |

**Catatan:** Nilai PSNR bervariasi antar frame (20.24–23.51 dB) tergantung sudut pandang. Frame dengan oklusi parsial atau sudut ekstrem cenderung memiliki metrik lebih rendah, sesuai dengan karakteristik NeRF yang sensitif terhadap kompleksitas geometri [Mildenhall et al., 2020].

---

## 🔧 Konfigurasi & Parameter

### Parameter Training (`src/train.py`)
| Parameter | Default | Deskripsi | Rekomendasi |
|-----------|---------|-----------|-------------|
| `iterations` | 300 | Jumlah langkah training | 50k-200k untuk hasil SOTA |
| `batch_size` | 2048 | Rays per iteration | 1024-4096 (sesuai VRAM) |
| `lr` | 5e-4 | Learning rate awal | 1e-4 – 1e-3 |
| `lr_decay` | 0.999 | Faktor decay eksponensial | 0.999 – 0.9999 |
| `N_samples` | 64 | Samples per ray | 32-128 (trade-off kualitas/waktu) |
| `chunk` | 2048 | Rays per forward pass | Sesuaikan dengan memori GPU |
| `H`, `W` | 100 | Resolusi render | 200-800 untuk kualitas tinggi |

### Parameter Model (`src/model.py`)
| Parameter | Default | Deskripsi |
|-----------|---------|-----------|
| `depth` | 8 | Jumlah layer MLP |
| `width` | 256 | Neuron per hidden layer |
| `num_freqs` | 10 | Level frekuensi positional encoding (posisi) |
| `num_freqs_views` | 4 | Level frekuensi positional encoding (arah) |
| `skip_layer` | 4 | Layer untuk skip connection |

### Parameter Rendering (`src/renderer.py`)
| Parameter | Default | Deskripsi |
|-----------|---------|-----------|
| `near` | 2.0 | Batas dekat sampling depth |
| `far` | 6.0 | Batas jauh sampling depth |
| `raw_noise_std` | 0.0 | Noise untuk regularisasi density (training only) |

---

## 🐛 Troubleshooting

### ❌ Error: `ModuleNotFoundError: No module named 'torch'`
**Penyebab:** Dependencies belum terinstall.  
**Solusi:**
```bash
pip install -r requirements.txt
# Atau install manual:
pip install torch torchvision numpy scikit-image lpips tqdm Pillow
```

### ❌ Error: `CUDA out of memory`
**Penyebab:** Batch size atau chunk terlalu besar untuk VRAM GPU.  
**Solusi:**
- Kurangi `batch_size` di `train.py` (misal: 2048 → 1024)
- Kurangi `chunk` di `renderer.py` (misal: 2048 → 512)
- Gunakan `N_samples` lebih kecil (64 → 32)

### ❌ Error: `FileNotFoundError: metrics.json`
**Penyebab:** Export belum dijalankan atau path salah di Processing.  
**Solusi:**
1. Pastikan `python src/export.py` sudah dijalankan sukses
2. Di Processing, edit path absolut di `loadMetrics()`:
   ```java
   String path = "/path/ke/nerf_pipeline/results/metrics.json";
   ```

### ❌ Processing: `NullPointerException` saat load frames
**Penyebab:** Folder `frames/` tidak ditemukan atau path salah.  
**Solusi:**
- Verifikasi folder `results/frames/` berisi 40 file PNG
- Update path di `loadFrames()` Processing dengan path absolut

### ❌ Training sangat lambat di CPU
**Penyebab:** NeRF secara komputasi berat; CPU tidak memiliki paralelisasi GPU.  
**Solusi:**
- Gunakan Google Colab/Kaggle (GPU gratis)
- Kurangi `iterations` untuk testing (300 → 100)
- Gunakan resolusi lebih kecil (`H=50, W=50`)

---

## 📚 Referensi Akademik

### Paper Utama
```bibtex
@inproceedings{mildenhall2020nerf,
  title={NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis},
  author={Mildenhall, Ben and Srinivasan, Pratul P and Tancik, Matthew 
          and Barron, Jonathan T and Ramamoorthi, Ravi and Ng, Ren},
  booktitle={European Conference on Computer Vision},
  pages={405--421},
  year={2020},
  organization={Springer}
}
```

### Framework & Tools
```bibtex
@article{paszke2019pytorch,
  title={PyTorch: An Imperative Style, High-Performance Deep Learning Library},
  author={Paszke, Adam and Gross, Sam and Massa, Francisco and others},
  journal={Advances in Neural Information Processing Systems},
  volume={32},
  year={2019}
}

@book{reas2014processing,
  title={Processing: A Programming Handbook for Visual Designers and Artists},
  author={Reas, Casey and Fry, Ben},
  edition={2nd},
  publisher={MIT Press},
  year={2014}
}
```

### Metrik Evaluasi
```bibtex
@article{wang2004ssim,
  title={Image Quality Assessment: From Error Visibility to Structural Similarity},
  author={Wang, Zhou and Bovik, Alan C and Sheikh, Hamid R and Simoncelli, Eero P},
  journal={IEEE Transactions on Image Processing},
  volume={13},
  number={4},
  pages={600--612},
  year={2004}
}
```

### Dataset
```bibtex
@misc{blender_synthetic,
  title={Blender Synthetic Dataset for NeRF},
  author={Mildenhall, Ben et al.},
  year={2020},
  howpublished={\url{https://github.com/bmild/nerf}},
  note={Lisensi: CC-BY-NC 2.0}
}
```

---

## 🤝 Kontribusi

Kontribusi sangat diterima untuk pengembangan project ini! Jika Anda ingin berkontribusi:

1. **Fork** repository ini
2. Buat branch fitur baru: `git checkout -b fitur/nama-fitur`
3. Commit perubahan: `git commit -m 'feat: tambah fitur X'`
4. Push ke branch: `git push origin fitur/nama-fitur`
5. Buka **Pull Request** dengan deskripsi perubahan

**Guidelines Kontribusi:**
- Ikuti PEP 8 untuk kode Python
- Tambahkan docstring untuk fungsi baru
- Update dokumentasi jika mengubah API
- Test perubahan di CPU sebelum submit PR

---

## 📜 Lisensi

Project ini dilisensikan di bawah **Lisensi Akademik** untuk keperluan pendidikan dan riset.  
- ✅ Diperbolehkan: Menggunakan, memodifikasi, dan mendistribusikan untuk keperluan akademik dengan atribusi.  
- ❌ Dilarang: Penggunaan komersial tanpa izin tertulis dari penulis.  

Dataset Blender Synthetic dilisensikan di bawah **CC-BY-NC 2.0** oleh Mildenhall et al. (2020).

---

## ✉️ Kontak

**Jundulloh Rizki Ananda**  
📧 [Gmail](jundulloh2109@gmail.com)  
🔗 [GitHub Profile](https://github.com/JundiLesmana)  
🎓 [Universitas Pamulang](https://unpam.ac.id/)

*Dibimbing oleh: DEDE SUPIYAN S.Kom., M.Kom.*  
*Mata Kuliah: Computer Graphics*  
*Semester/Tahun: II - Genap 2025/2026*

---

> 💡 **Tip Reproduksi**: Untuk hasil yang mendekati paper asli NeRF, jalankan training dengan:
> - Iterasi: 200.000+
> - Resolusi: 800×800
> - N_samples: 64 (coarse) + 128 (fine)
> - GPU: V100/A100 dengan VRAM ≥ 16 GB
>
> Project ini difokuskan pada **validasi pipeline dan pembelajaran konsep**, bukan optimasi state-of-the-art.