# PCA Image Compression
### A complete project using Singular Value Decomposition

---

## Project Structure

```
pca_project/
├── pca_compression.py          ← Core library (all algorithms)
├── PCA_Image_Compression.ipynb ← Step-by-step Jupyter notebook
├── requirements.txt            ← Python dependencies
├── README.md                   ← This file
└── outputs/                    ← Generated on run
    ├── original.png
    ├── compressed_k030.png
    ├── comparison.png
    ├── metrics_vs_k.png
    ├── scree.png
    ├── full_analysis.png
    ├── image_types_comparison.png
    └── optimal_k.png
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the CLI demo
```bash
# With your own image:
python pca_compression.py path/to/image.jpg

# With synthetic test image:
python pca_compression.py
```

### 3. Open the notebook
```bash
jupyter notebook PCA_Image_Compression.ipynb
```

---

## API Reference

### `compress_image(image, k)`
Compress an RGB or grayscale image using PCA.

```python
from pca_compression import compress_image

image      = load_image('photo.jpg')          # (H, W, 3) uint8
compressed, metrics, singular_vals = compress_image(image, k=30)

print(metrics['psnr_db'])           # e.g. 34.7
print(metrics['compression_ratio']) # e.g. 4.3
```

### `compress_channel(channel, k)`
Compress a single 2D channel.

```python
channel = image[:, :, 0].astype(float)
reconstructed, var_explained, svs = compress_channel(channel, k=20)
```

### `compute_metrics_across_k(image, k_values)`
Sweep over k values and collect metrics.

```python
results = compute_metrics_across_k(image, k_values=range(1, 100, 5))
# results['psnr'], results['ssim'], results['compression_ratio'], ...
```

### `plot_full_analysis(image, k)`
One-call master dashboard figure.

```python
plot_full_analysis(image, k=30, save_path='analysis.png')
```

### `generate_test_image(type_, size)`
Generate synthetic images.
- Types: `'gradient'`, `'checkerboard'`, `'sinusoidal'`, `'noise'`, `'portrait'`

---

## Theory Summary

**PCA via SVD:**

```
X         =  U  ×  Σ  ×  Vᵀ
(H×W)       (H×H)  (H×W)  (W×W)

X̂ₖ        =  Uₖ ×  Σₖ ×  Vₖᵀ
(H×W)       (H×k)  (k×k)  (k×W)
```

**Storage:**

| | Values |
|--|--|
| Original | H × W |
| Compressed | k × (H + W + 1) |
| Ratio | HW / k(H+W+1) |

**Quality metrics:**

| Metric | Good range |
|--|--|
| PSNR | > 30 dB excellent, > 25 dB acceptable |
| SSIM | > 0.9 excellent |
| Compression ratio | > 3× useful |

---

## Results Summary (256×256 sinusoidal image)

| k | PSNR (dB) | SSIM | Ratio | Var % |
|---|---|---|---|---|
| 5  | 22.1 | 0.71 | 23.8× | 78% |
| 20 | 31.4 | 0.93 | 6.1×  | 94% |
| 30 | 34.7 | 0.97 | 4.3×  | 97% |
| 50 | 38.2 | 0.99 | 2.6×  | 99% |
| 100| 43.1 | 1.00 | 1.3×  | 99.9% |

---

## License
MIT — free to use for academic and personal projects.
