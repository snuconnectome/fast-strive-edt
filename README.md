# Fast-STRIVE-EDT

**Exact 2D-Tiled Triton/PyTorch CUDA 3D Euclidean Distance Transform and Topological White-Matter Lesion Spatial Partitioning**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-green.svg)](pyproject.toml)

`fast-strive-edt` provides a mathematically exact, anisotropy-aware 3D Euclidean Distance Transform (EDT) and multi-compartment spatial partitioning engine for neuroimaging White Matter Hyperintensity (FLAIR WMH) and T1 White-Matter Hypointensity (`aseg.mgz` label 77) masks, implementing both volumetric distance conventions (DeCarli et al., 2005; Kim et al., 2008; Griffanti et al., 2018) and STRIVE-1/2 (Wardlaw et al., 2013; Duering et al., 2023) 3D connected-component continuity rules.

---

## Key Features

1. **Exact 2D-Tiled Triton CUDA 3D Separable EDT (`triton_edt3d`)**:
   - Implements the 3-pass separable quadratic min-convolution formulation (Meijster et al., 2000; Maurer et al., 2003; Felzenszwalb & Huttenlocher, 2012) with a register-tiled 2D Triton kernel (`_edt1d_tiled_kernel`, $O(N^2)$ per 1D line / $O(DHW(D+H+W))$ total ops entirely in registers/SRAM) that eliminates strided global memory transactions and avoids 16 GB 4D tensor allocations.
   - Includes a line-chunked pure PyTorch CUDA/CPU fallback (`pytorch_edt3d`, `chunk_lines=1024`, `<532 MB` peak VRAM on full $256^3$ volumes) and an anisotropy-aware SciPy CPU backend (`scipy_edt3d`).
2. **Mathematically Lossless Anisotropic Bounding-Box Pruning (`fast_bbox_3d`)**:
   - Computes physical per-axis padding $p_d = \lceil \tau_{\max} / s_d \rceil + 2$ using the NIfTI header voxel spacing $\mathbf{s} = (s_z, s_y, s_x)$ and verifies affine axis orthogonality ($\max_{i \ne j}|\cos \theta_{ij}| < 10^{-3}$).
3. **Comprehensive Anatomical & Topological Partitioning (`partition_lesion_topography`)**:
   - **Ventricle Reference Definitions**: Strict Lateral Ventricles (`labels [4, 43]`) vs Full Ventricular System (`labels [4, 43, 5, 44, 14]`).
   - **Multi-Threshold Distance Cutoffs**: $3.0\text{ mm}$, $10.0\text{ mm}$ (DeCarli et al., 2005; FSL BIANCA), and $13.0\text{ mm}$ (Kim et al., 2008).
   - **Juxtacortical vs Deep Subcortical Sub-Classification**: Partitions non-periventricular lesions ($>10.0\text{ mm}$ from ventricles) into Juxtacortical ($\le 4.0\text{ mm}$ from cerebral cortex `labels [3, 42]`) and Deep Subcortical ($>4.0\text{ mm}$ from cortex).
   - **3D 26-Connected-Component Continuity Rule**: Classifies confluent lesion clusters originating within $3.0\text{ mm}$ of the ventricular ependyma as periventricular regardless of how far they extend into deep white matter.
4. **Concurrent Multi-GPU / Slurm Batch Ledger (`SQLiteWALBatchLedger`)**:
   - Built-in SQLite Write-Ahead Logging (`PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA busy_timeout=30000;`) for lock-free progress tracking across multi-GPU worker arrays.

---

## Empirical Benchmark Summary (8× NVIDIA Tesla V100-SXM2-16GB, $N = 4,176$ Brain Scans)

| Configuration / Backend | Grid / ROI Regime | Latency per Scan (ms) | Speedup vs Uncropped CPU | Speedup vs Cropped CPU | Max Abs Error (mm) | 10mm PV Voxel Match / Dice |
|---|---|---:|---:|---:|---:|---:|
| SciPy CPU (`1-Core`, `sampling=zooms`) | Full Volume ($256^3$) | `2,371.37 ms` | `1.0×` | — | `0.00` (Ref) | `100.0%` (`1.00000000`) |
| Triton 2D-Tiled CUDA (`1× V100`) | Full Volume ($256^3$) | `11.42 ms` | **`207.6×`** | — | `1.51e-5 mm` | `100.0%` (`1.00000000`) |
| SciPy CPU (`1-Core`, `sampling=zooms`) | Padded BBox Crop ($N=500$ GARD) | `183.89 ms` | `12.9×` | `1.0×` | `0.00` (Ref) | `100.0%` (`1.00000000`) |
| SciPy CPU (`32-Core` Parallel) | Padded BBox Crop ($N=500$ GARD) | `20.19 ms` | `117.5×` | `9.1×` | `0.00` (Ref) | `100.0%` (`1.00000000`) |
| PyTorch CUDA Separable (`1× V100` Warm) | Padded BBox Crop ($N=500$ GARD) | `44.86 ms` | `52.9×` | `4.1×` | `3.81e-6 mm` | `500/500` (`1.00000000`) |
| **Triton 2D-Tiled CUDA (`1× V100` Warm)** | **Padded BBox Crop ($N=500$ GARD)** | **`2.44 ms`** | **`970.7×`** | **`75.3×`** | **`3.81e-6 mm`** | **`500/500` (`1.00000000`)** |
| **Multi-GPU End-to-End (`8× V100`)** | **Full Cohort ($N=4,176$, 3× EDT + 26-CC + I/O)** | **`6.52 ms`** (`27.23 s` total) | — | — | **`3.81e-6 mm`** | **`100.0%` (`1.00000000`)** |

---

## Installation & Quick Start

```bash
pip install -e ".[dev]"

# Run synthetic anisotropic 3D phantom verification
fast-strive-edt verify-phantom

# Run 3D EDT speed and numerical accuracy benchmark
fast-strive-edt benchmark --repeats 20

# Run unit & mathematical invariant test suite
python3 -m unittest discover -s tests -p "test_*.py" -v
```

### Python API Example

```python
import nibabel as nib
import numpy as np
from fast_strive_edt import partition_lesion_topography

img = nib.load("sub-01/mri/aseg.mgz")
zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
data = np.asarray(img.dataobj, dtype=np.int16)

metrics = partition_lesion_topography(
    data,
    spacing=zooms,
    affine=img.affine,
    backend="auto",          # Uses Triton CUDA if available, else PyTorch CUDA or SciPy CPU
    vent_mode="strict",      # 'strict' ([4, 43]) or 'full' ([4, 43, 5, 44, 14])
    thresholds_mm=(3.0, 10.0, 13.0),
    juxtacortical_mm=4.0,
    continuity_seed_mm=3.0,
)
print(metrics)
```

---

## Method Paper Preprint

The full short method paper preprint is available in:
- Markdown: [`paper/PREPRINT_FAST_STRIVE_EDT.md`](paper/PREPRINT_FAST_STRIVE_EDT.md)
- De-identified Benchmark Artifact: [`benchmarks/dgx1_v100_gard4176_benchmark.json`](benchmarks/dgx1_v100_gard4176_benchmark.json)
