"""
pca_compression.py
==================
PCA-based Image Compression using Singular Value Decomposition (SVD).

Theory:
    For an image channel matrix X (H x W), PCA finds the k principal
    directions that capture the most variance. Compression is achieved by
    projecting X onto these k components and reconstructing.

    Steps:
        1. Mean-center each row of X
        2. Compute SVD: X = U * S * Vt
        3. Keep top-k components: Xk = U[:, :k] * S[:k] * Vt[:k, :]
        4. Add back row means to reconstruct

    Compression ratio:
        Original:    H * W  values
        Compressed:  k * (H + W + 1)  values (scores + components + means)
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────
#  Core PCA functions
# ─────────────────────────────────────────

def compress_channel(channel: np.ndarray, k: int):
    """
    Compress a single 2-D image channel using PCA (SVD).

    Parameters
    ----------
    channel : np.ndarray  shape (H, W), dtype float
    k       : int         number of principal components to retain

    Returns
    -------
    reconstructed : np.ndarray  shape (H, W)
    explained_var : float       fraction of variance explained by top-k
    singular_vals : np.ndarray  all singular values
    """
    H, W = channel.shape
    k = min(k, H, W)

    # Mean-center rows
    row_means = channel.mean(axis=1, keepdims=True)
    X = channel - row_means

    # Full SVD  (economy / thin form)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    # Explained variance
    var_explained = (S[:k] ** 2).sum() / ((S ** 2).sum() + 1e-10)

    # Reconstruct with top-k
    Uk  = U[:, :k]          # (H, k)
    Sk  = np.diag(S[:k])    # (k, k)
    Vtk = Vt[:k, :]         # (k, W)
    reconstructed = Uk @ Sk @ Vtk + row_means

    return np.clip(reconstructed, 0, 255), var_explained, S


def compress_image(image: np.ndarray, k: int):
    """
    Compress an RGB or grayscale image using per-channel PCA.

    Parameters
    ----------
    image : np.ndarray  shape (H, W) or (H, W, 3), values in [0, 255]
    k     : int         number of principal components

    Returns
    -------
    compressed    : np.ndarray  same shape as image, uint8
    metrics       : dict        compression statistics
    singular_vals : list        singular values per channel
    """
    image = image.astype(np.float64)
    is_gray = image.ndim == 2

    if is_gray:
        channels = [image]
        channel_names = ['Gray']
    else:
        channels = [image[:, :, c] for c in range(3)]
        channel_names = ['R', 'G', 'B']

    compressed_channels = []
    var_explained_list  = []
    all_singular_vals   = []

    for ch, name in zip(channels, channel_names):
        comp, var_exp, svs = compress_channel(ch, k)
        compressed_channels.append(comp)
        var_explained_list.append(var_exp)
        all_singular_vals.append(svs)

    if is_gray:
        compressed = compressed_channels[0]
    else:
        compressed = np.stack(compressed_channels, axis=-1)

    compressed = np.clip(compressed, 0, 255).astype(np.uint8)

    # ── Metrics ──────────────────────────────
    H, W = image.shape[:2]
    n_channels = 1 if is_gray else 3

    original_size    = H * W * n_channels
    compressed_size  = k * (H + W + 1) * n_channels
    compression_ratio = original_size / max(compressed_size, 1)

    mse  = np.mean((image - compressed.astype(np.float64)) ** 2)
    psnr = 10 * np.log10(255 ** 2 / mse) if mse > 0 else float('inf')
    ssim = compute_ssim(image, compressed.astype(np.float64))

    metrics = {
        'k'                 : k,
        'image_shape'       : (H, W),
        'n_channels'        : n_channels,
        'original_size_px'  : original_size,
        'compressed_size_px': compressed_size,
        'compression_ratio' : compression_ratio,
        'mse'               : mse,
        'psnr_db'           : psnr,
        'ssim'              : ssim,
        'var_explained_mean': float(np.mean(var_explained_list)),
        'var_explained'     : {n: v for n, v in zip(channel_names, var_explained_list)},
    }

    return compressed, metrics, all_singular_vals


# ─────────────────────────────────────────
#  Quality metrics
# ─────────────────────────────────────────

def compute_ssim(img1: np.ndarray, img2: np.ndarray, C1=6.5025, C2=58.5225) -> float:
    """Simplified SSIM (structural similarity index)."""
    if img1.ndim == 3:
        ssims = [compute_ssim(img1[:,:,c], img2[:,:,c], C1, C2) for c in range(3)]
        return float(np.mean(ssims))
    mu1  = img1.mean()
    mu2  = img2.mean()
    s1   = img1.std()
    s2   = img2.std()
    cov  = np.mean((img1 - mu1) * (img2 - mu2))
    num  = (2*mu1*mu2 + C1) * (2*cov + C2)
    den  = (mu1**2 + mu2**2 + C1) * (s1**2 + s2**2 + C2)
    return float(num / (den + 1e-10))


def compute_metrics_across_k(image: np.ndarray, k_values: list) -> dict:
    """
    Compute compression metrics for a range of k values.

    Returns dict of lists keyed by metric name.
    """
    results = {
        'k': [], 'psnr': [], 'mse': [], 'ssim': [],
        'compression_ratio': [], 'var_explained': []
    }
    for k in k_values:
        _, m, _ = compress_image(image, k)
        results['k'].append(k)
        results['psnr'].append(m['psnr_db'])
        results['mse'].append(m['mse'])
        results['ssim'].append(m['ssim'])
        results['compression_ratio'].append(m['compression_ratio'])
        results['var_explained'].append(m['var_explained_mean'] * 100)
    return results


# ─────────────────────────────────────────
#  I/O helpers
# ─────────────────────────────────────────

def load_image(path: str) -> np.ndarray:
    """Load image as uint8 numpy array (H, W, 3) RGB."""
    img = Image.open(path).convert('RGB')
    return np.array(img)


def save_image(array: np.ndarray, path: str):
    """Save numpy array as PNG."""
    Image.fromarray(array.astype(np.uint8)).save(path)
    print(f"Saved → {path}")


def generate_test_image(type_: str = 'gradient', size: int = 256) -> np.ndarray:
    """
    Generate a synthetic test image.
    type_ : 'gradient' | 'checkerboard' | 'sinusoidal' | 'noise' | 'portrait'
    """
    H = W = size
    img = np.zeros((H, W, 3), dtype=np.uint8)
    x, y = np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H))

    if type_ == 'gradient':
        img[:,:,0] = (x * 255).astype(np.uint8)
        img[:,:,1] = (y * 255).astype(np.uint8)
        img[:,:,2] = ((1 - x) * 255).astype(np.uint8)

    elif type_ == 'checkerboard':
        sq = 32
        cb = ((np.floor(x * W / sq) + np.floor(y * H / sq)) % 2).astype(np.uint8)
        v  = cb * 220 + 20
        img[:,:,0] = img[:,:,1] = img[:,:,2] = v

    elif type_ == 'sinusoidal':
        r = (np.sin(x * np.pi * 8) * 127 + 128).astype(np.uint8)
        g = (np.cos(y * np.pi * 6) * 127 + 128).astype(np.uint8)
        b = (np.sin((x + y) * np.pi * 4) * 127 + 128).astype(np.uint8)
        img[:,:,0] = r; img[:,:,1] = g; img[:,:,2] = b

    elif type_ == 'noise':
        img = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)

    elif type_ == 'portrait':
        cx, cy = W // 2, H // 2
        dist = np.sqrt((x * W - cx)**2 + (y * H - cy)**2)
        face = np.clip(200 - dist * 0.8 + np.sin(x * 10) * 20, 0, 255)
        img[:,:,0] = face.astype(np.uint8)
        img[:,:,1] = np.clip(face * 0.85, 0, 255).astype(np.uint8)
        img[:,:,2] = np.clip(face * 0.7,  0, 255).astype(np.uint8)

    return img


# ─────────────────────────────────────────
#  Visualization
# ─────────────────────────────────────────

def plot_compression_comparison(image: np.ndarray, k_values: list,
                                 title: str = 'PCA Compression', save_path: str = None):
    """
    Side-by-side comparison at multiple k values.
    """
    n = len(k_values) + 1
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
    fig.patch.set_facecolor('#0d0d0f')

    def show(ax, img, label, sub=''):
        ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
        ax.set_title(label, color='white', fontsize=11, pad=8, fontweight='bold')
        ax.set_xlabel(sub, color='#7a7a8c', fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('#2a2a35')

    show(axes[0], image, 'Original', f'{image.shape[1]}×{image.shape[0]} px')

    for i, k in enumerate(k_values):
        comp, m, _ = compress_image(image, k)
        show(axes[i+1], comp, f'k = {k}',
             f"PSNR: {m['psnr_db']:.1f} dB\nRatio: {m['compression_ratio']:.1f}x\nVar: {m['var_explained_mean']*100:.1f}%")

    fig.suptitle(title, color='white', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"Saved → {save_path}")
    plt.show()


def plot_metrics_vs_k(image: np.ndarray, k_range: range = range(1, 101, 5),
                       save_path: str = None):
    """
    4-panel plot: PSNR, MSE, SSIM, Compression Ratio vs k.
    """
    k_values = list(k_range)
    print(f"Computing metrics for {len(k_values)} values of k…")
    res = compute_metrics_across_k(image, k_values)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.patch.set_facecolor('#0d0d0f')
    plt.suptitle('PCA Compression Quality vs Number of Components (k)',
                 color='white', fontsize=14, fontweight='bold')

    style = dict(color='#0d0d0f', labelcolor='white',
                 facecolor='#141418', edgecolor='#2a2a35')

    panels = [
        (axes[0,0], 'k', 'psnr',              'PSNR (dB)',           '#6c63ff', 'Higher = Better'),
        (axes[0,1], 'k', 'ssim',              'SSIM',                '#00e5a0', 'Higher = Better'),
        (axes[1,0], 'k', 'compression_ratio', 'Compression Ratio',   '#ffb84d', 'Higher = More compressed'),
        (axes[1,1], 'k', 'var_explained',     'Variance Explained %','#ff4d6d', 'Higher = More info retained'),
    ]

    for ax, xk, yk, ylabel, color, note in panels:
        ax.set_facecolor('#141418')
        ax.plot(res[xk], res[yk], color=color, linewidth=2.5, marker='o',
                markersize=3, markevery=2)
        ax.fill_between(res[xk], res[yk], alpha=0.12, color=color)
        ax.set_xlabel('k (components)', color='#7a7a8c', fontsize=10)
        ax.set_ylabel(ylabel, color='white', fontsize=10)
        ax.set_title(f'{ylabel}  ·  {note}', color='white', fontsize=10, pad=6)
        ax.tick_params(colors='#7a7a8c')
        ax.grid(True, color='#2a2a35', linewidth=0.5, linestyle='--')
        for spine in ax.spines.values():
            spine.set_edgecolor('#2a2a35')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"Saved → {save_path}")
    plt.show()


def plot_scree(singular_vals: list, channel_names: list, k: int, save_path: str = None):
    """
    Scree plot — eigenvalue spectrum per channel.
    """
    fig, axes = plt.subplots(1, len(singular_vals), figsize=(5 * len(singular_vals), 4))
    fig.patch.set_facecolor('#0d0d0f')
    if len(singular_vals) == 1:
        axes = [axes]

    colors = ['#ff4d6d', '#00e5a0', '#6c63ff', '#ffb84d']
    for ax, svs, name, col in zip(axes, singular_vals, channel_names, colors):
        top = min(60, len(svs))
        idx = np.arange(1, top + 1)
        cumvar = np.cumsum(svs[:top]**2) / (np.sum(svs**2) + 1e-10) * 100

        ax.set_facecolor('#141418')
        ax2 = ax.twinx()

        ax.bar(idx, svs[:top], color=col, alpha=0.7, width=0.8, label='Singular values')
        ax2.plot(idx, cumvar, color='white', linewidth=1.5, linestyle='--', label='Cumul. variance %')

        if k <= top:
            ax.axvline(k, color='#ffb84d', linewidth=1.5, linestyle=':', label=f'k={k}')

        ax.set_xlabel('Component index', color='#7a7a8c', fontsize=9)
        ax.set_ylabel('Singular value', color=col, fontsize=9)
        ax2.set_ylabel('Cumul. variance %', color='white', fontsize=9)
        ax.set_title(f'Channel {name}', color='white', fontsize=11, fontweight='bold')
        ax.tick_params(colors='#7a7a8c'); ax2.tick_params(colors='white')
        ax.grid(True, color='#2a2a35', linewidth=0.4)
        for spine in ax.spines.values(): spine.set_edgecolor('#2a2a35')
        for spine in ax2.spines.values(): spine.set_edgecolor('#2a2a35')

    fig.suptitle('Scree Plot — Singular Value Spectrum', color='white',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"Saved → {save_path}")
    plt.show()


def plot_full_analysis(image: np.ndarray, k: int, save_path: str = None):
    """
    Master figure: original, compressed, diff, error heatmap, channel histograms.
    """
    compressed, metrics, svs = compress_image(image, k)
    diff = np.abs(image.astype(int) - compressed.astype(int)).astype(np.uint8)

    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor('#0d0d0f')
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.25)

    def dark_ax(ax):
        ax.set_facecolor('#141418')
        ax.tick_params(colors='#7a7a8c')
        for sp in ax.spines.values(): sp.set_edgecolor('#2a2a35')

    # Original
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(image); ax0.set_title('Original', color='white', fontweight='bold')
    ax0.set_xticks([]); ax0.set_yticks([])
    dark_ax(ax0)

    # Compressed
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(compressed); ax1.set_title(f'Compressed  k={k}', color='#6c63ff', fontweight='bold')
    ax1.set_xticks([]); ax1.set_yticks([])
    dark_ax(ax1)

    # Difference ×5
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.imshow(np.clip(diff * 5, 0, 255)); ax2.set_title('Difference ×5', color='#ff4d6d', fontweight='bold')
    ax2.set_xticks([]); ax2.set_yticks([])
    dark_ax(ax2)

    # Error heatmap
    ax3 = fig.add_subplot(gs[0, 3])
    err_gray = diff.mean(axis=2) if diff.ndim == 3 else diff
    im = ax3.imshow(err_gray, cmap='hot')
    ax3.set_title('Error Heatmap', color='#ffb84d', fontweight='bold')
    ax3.set_xticks([]); ax3.set_yticks([])
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04).ax.tick_params(colors='#7a7a8c')
    dark_ax(ax3)

    # Metrics text
    ax4 = fig.add_subplot(gs[1, 0])
    dark_ax(ax4); ax4.set_xticks([]); ax4.set_yticks([])
    ax4.set_title('Metrics', color='white', fontweight='bold')
    mtext = (
        f"Components k   :  {metrics['k']}\n"
        f"Image size     :  {metrics['image_shape'][1]}×{metrics['image_shape'][0]}\n"
        f"Orig. size     :  {metrics['original_size_px']:,} values\n"
        f"Comp. size     :  {metrics['compressed_size_px']:,} values\n"
        f"Comp. ratio    :  {metrics['compression_ratio']:.2f}x\n"
        f"PSNR           :  {metrics['psnr_db']:.2f} dB\n"
        f"MSE            :  {metrics['mse']:.2f}\n"
        f"SSIM           :  {metrics['ssim']:.4f}\n"
        f"Var. explained :  {metrics['var_explained_mean']*100:.1f}%"
    )
    ax4.text(0.05, 0.92, mtext, transform=ax4.transAxes,
             color='#00e5a0', fontsize=9.5, fontfamily='monospace',
             verticalalignment='top', linespacing=1.8)

    # Singular values scree
    ax5 = fig.add_subplot(gs[1, 1:3])
    dark_ax(ax5)
    colors = ['#ff4d6d', '#00e5a0', '#6c63ff']
    names  = ['R', 'G', 'B'] if len(svs) == 3 else ['Gray']
    for sv, col, nm in zip(svs, colors, names):
        top = min(80, len(sv))
        ax5.plot(range(1, top+1), sv[:top], color=col, linewidth=1.8, label=f'Channel {nm}')
    ax5.axvline(k, color='#ffb84d', linewidth=1.5, linestyle='--', label=f'k={k}')
    ax5.set_xlabel('Component index', color='#7a7a8c')
    ax5.set_ylabel('Singular value', color='white')
    ax5.set_title('Singular Value Spectrum', color='white', fontweight='bold')
    ax5.legend(facecolor='#1c1c22', edgecolor='#2a2a35', labelcolor='white', fontsize=9)
    ax5.grid(True, color='#2a2a35', linewidth=0.4)

    # Cumulative variance
    ax6 = fig.add_subplot(gs[1, 3])
    dark_ax(ax6)
    for sv, col, nm in zip(svs, colors, names):
        cumvar = np.cumsum(sv**2) / (np.sum(sv**2) + 1e-10) * 100
        top = min(100, len(cumvar))
        ax6.plot(range(1, top+1), cumvar[:top], color=col, linewidth=1.8, label=nm)
    ax6.axvline(k, color='#ffb84d', linewidth=1.5, linestyle='--', label=f'k={k}')
    ax6.axhline(95, color='white', linewidth=0.8, linestyle=':', alpha=0.5)
    ax6.set_xlabel('Components k', color='#7a7a8c')
    ax6.set_ylabel('Cumul. Variance %', color='white')
    ax6.set_title('Cumulative Variance', color='white', fontweight='bold')
    ax6.legend(facecolor='#1c1c22', edgecolor='#2a2a35', labelcolor='white', fontsize=9)
    ax6.grid(True, color='#2a2a35', linewidth=0.4)

    fig.suptitle(f'PCA Image Compression — Full Analysis  (k = {k})',
                 color='white', fontsize=15, fontweight='bold')
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"Saved → {save_path}")
    plt.show()
    return metrics


# ─────────────────────────────────────────
#  CLI demo
# ─────────────────────────────────────────

if __name__ == '__main__':
    import sys

    print("=" * 55)
    print("  PCA Image Compression — Full Demo")
    print("=" * 55)

    # Use a provided image or generate a test one
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        print(f"Loading image: {sys.argv[1]}")
        image = load_image(sys.argv[1])
    else:
        print("No image provided — generating synthetic 'sinusoidal' test image (256×256)")
        image = generate_test_image('sinusoidal', size=256)

    os.makedirs('outputs', exist_ok=True)

    # Single compression at k=30
    print("\n── Single compression (k=30) ──")
    comp, metrics, svs = compress_image(image, k=30)
    for key, val in metrics.items():
        if key != 'var_explained':
            print(f"  {key:<25} {val}")

    save_image(comp, 'outputs/compressed_k30.png')
    save_image(image, 'outputs/original.png')

    # Multi-k comparison
    print("\n── Generating comparison figure (k = 5, 20, 50, 100) ──")
    plot_compression_comparison(image, k_values=[5, 20, 50, 100],
                                 title='PCA Compression at Different k',
                                 save_path='outputs/comparison.png')

    # Metrics vs k
    print("\n── Plotting metrics vs k ──")
    plot_metrics_vs_k(image, k_range=range(1, 101, 4),
                       save_path='outputs/metrics_vs_k.png')

    # Full analysis
    print("\n── Full analysis figure ──")
    plot_full_analysis(image, k=30, save_path='outputs/full_analysis.png')

    print("\n✓ All outputs saved to ./outputs/")
