"""Exact Anisotropic 3D Euclidean Distance Transform (EDT) on GPU (Triton & PyTorch) and CPU.

Implements the separable 3-pass parabolic lower-envelope formulation (Meijster et al., 2000;
Maurer et al., 2003; Felzenszwalb & Huttenlocher, 2012):
    D^2(z, y, x) = min_{z', y', x' in S} [ s_z^2 (z - z')^2 + s_y^2 (y - y')^2 + s_x^2 (x - x')^2 ]
via three sequential 1D parabolic envelope reductions along X, Y, and Z, accelerated on CUDA GPUs
using a 2D register-tiled Triton kernel (`_edt1d_tiled_kernel`) that loads `(BLOCK_R, BLOCK_N)`
tiles into SRAM registers once per row-block and eliminates strided global memory transactions.
"""

from typing import Tuple, Optional, Union
import numpy as np
import scipy.ndimage as ndi

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _edt1d_tiled_kernel(
        in_ptr, out_ptr,
        n_rows, N,
        scale_sq,
        BLOCK_R: tl.constexpr, BLOCK_N: tl.constexpr
    ):
        pid_r = tl.program_id(0)
        row_offs = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        row_mask = row_offs < n_rows
        col_offs = tl.arange(0, BLOCK_N)
        col_mask = col_offs < N

        ptrs = in_ptr + row_offs[:, None] * N + col_offs[None, :]
        f_rj = tl.load(ptrs, mask=row_mask[:, None] & col_mask[None, :], other=1e20)
        j_pos = col_offs.to(tl.float32)[None, :]

        for i in range(N):
            diff = i.to(tl.float32) - j_pos
            cand = f_rj + scale_sq * diff * diff
            min_val = tl.min(cand, axis=1)
            out_ptrs = out_ptr + row_offs * N + i
            tl.store(out_ptrs, min_val, mask=row_mask)

    def _triton_edt1d_last_axis(f: "torch.Tensor", spacing: float) -> "torch.Tensor":
        orig_shape = f.shape
        N = orig_shape[-1]
        f_2d = f.contiguous().view(-1, N)
        n_rows = f_2d.shape[0]
        out_2d = torch.empty_like(f_2d)
        BLOCK_N = max(64, triton.next_power_of_2(N))
        BLOCK_R = 16
        grid = (triton.cdiv(n_rows, BLOCK_R),)
        _edt1d_tiled_kernel[grid](
            f_2d, out_2d,
            n_rows, N,
            float(spacing * spacing),
            BLOCK_R=BLOCK_R, BLOCK_N=BLOCK_N,
        )
        return out_2d.view(orig_shape)


def triton_edt3d(
    binary_fg: "torch.Tensor",
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> "torch.Tensor":
    """Compute exact 3D Euclidean distance (in mm) to nearest foreground voxel (`binary_fg > 0`)
    using the 2D-tiled Triton CUDA kernel.
    """
    if not (HAS_TORCH and HAS_TRITON and binary_fg.is_cuda):
        raise RuntimeError("triton_edt3d requires a CUDA torch.Tensor and Triton installed.")
    INF = 1e12
    f = torch.where(binary_fg > 0, 0.0, INF).to(torch.float32)
    f = _triton_edt1d_last_axis(f, float(spacing[2]))
    f = _triton_edt1d_last_axis(f.permute(0, 2, 1), float(spacing[1])).permute(0, 2, 1)
    f = _triton_edt1d_last_axis(f.permute(1, 2, 0), float(spacing[0])).permute(2, 0, 1)
    return torch.sqrt(torch.clamp(f, min=0.0))


def _pytorch_edt1d_chunked(f: "torch.Tensor", spacing: float, chunk_lines: int = 1024) -> "torch.Tensor":
    """Memory-bounded 1D quadratic min-convolution along the last axis (<270 MB peak VRAM at 256^3)."""
    orig_shape = f.shape
    N = orig_shape[-1]
    f_2d = f.contiguous().view(-1, N)
    n_rows = f_2d.shape[0]
    out_2d = torch.empty_like(f_2d)
    coords = torch.arange(N, device=f.device, dtype=torch.float32) * float(spacing)
    sq_dist = (coords.unsqueeze(1) - coords.unsqueeze(0)).pow_(2)
    for start in range(0, n_rows, chunk_lines):
        end = min(start + chunk_lines, n_rows)
        chunk = f_2d[start:end]
        out_2d[start:end] = torch.min(chunk.unsqueeze(1) + sq_dist.unsqueeze(0), dim=-1).values
    return out_2d.view(orig_shape)


def pytorch_edt3d(
    binary_fg: "torch.Tensor",
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    chunk_lines: int = 1024,
) -> "torch.Tensor":
    """Compute exact 3D Euclidean distance (in mm) to nearest foreground voxel (`binary_fg > 0`)
    using memory-bounded chunked PyTorch tensor reductions (works on CUDA, MPS, or CPU without Triton).
    """
    if not HAS_TORCH:
        raise RuntimeError("pytorch_edt3d requires PyTorch installed.")
    sz, sy, sx = float(spacing[0]), float(spacing[1]), float(spacing[2])
    INF = 1.0e12
    d = torch.where(binary_fg > 0, 0.0, INF).to(torch.float32)
    d = _pytorch_edt1d_chunked(d, sx, chunk_lines=chunk_lines)
    d = _pytorch_edt1d_chunked(d.permute(0, 2, 1), sy, chunk_lines=chunk_lines).permute(0, 2, 1)
    d = _pytorch_edt1d_chunked(d.permute(1, 2, 0), sz, chunk_lines=chunk_lines).permute(2, 0, 1)
    return torch.sqrt(torch.clamp(d, min=0.0))


def scipy_edt3d(
    binary_fg: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Compute exact 3D Euclidean distance (in mm) to nearest foreground voxel (`binary_fg > 0`)
    using anisotropy-aware SciPy CPU (`scipy.ndimage.distance_transform_edt`).
    """
    return ndi.distance_transform_edt(~(binary_fg > 0), sampling=spacing).astype(np.float32)


def edt3d(
    binary_fg: Union[np.ndarray, "torch.Tensor"],
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    backend: str = "auto",
) -> np.ndarray:
    """Unified 3D EDT dispatcher ('auto', 'triton', 'pytorch', 'scipy')."""
    if isinstance(binary_fg, np.ndarray):
        arr_np = binary_fg > 0
    else:
        arr_np = (binary_fg > 0).detach().cpu().numpy()

    if not np.any(arr_np):
        return np.full(arr_np.shape, np.inf, dtype=np.float32)

    if backend == "scipy" or (backend == "auto" and not (HAS_TORCH and torch.cuda.is_available())):
        return scipy_edt3d(arr_np, spacing=spacing)

    device = torch.device("cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu")
    t_fg = torch.from_numpy(np.ascontiguousarray(arr_np)).to(device)
    if backend == "triton" or (backend == "auto" and HAS_TRITON and t_fg.is_cuda):
        return triton_edt3d(t_fg, spacing=spacing).cpu().numpy()
    return pytorch_edt3d(t_fg, spacing=spacing).cpu().numpy()


def compute_affine_shear(affine: np.ndarray) -> float:
    """Compute maximum off-diagonal direction cosine magnitude |cos(theta_ij)| (i != j)
    of the 3x3 linear voxel-to-world transformation matrix.
    A value > 1e-3 indicates oblique/sheared voxel axes where separable axis-aligned EDT
    requires prior orthogonal resampling.
    """
    R = np.asarray(affine[:3, :3], dtype=np.float64)
    c0, c1, c2 = R[:, 0], R[:, 1], R[:, 2]
    n0 = float(np.sqrt(np.sum(c0 * c0)))
    n1 = float(np.sqrt(np.sum(c1 * c1)))
    n2 = float(np.sqrt(np.sum(c2 * c2)))
    c01 = abs(float(np.sum(c0 * c1)) / max(n0 * n1, 1e-12))
    c02 = abs(float(np.sum(c0 * c2)) / max(n0 * n2, 1e-12))
    c12 = abs(float(np.sum(c1 * c2)) / max(n1 * n2, 1e-12))
    return max(c01, c02, c12)


def fast_bbox_3d(mask: np.ndarray) -> Optional[Tuple[int, int, int, int, int, int]]:
    """Compute tight 3D bounding box (z0, z1, y0, y1, x0, x1) via O(N) boolean axis reductions."""
    z_any = np.any(mask, axis=(1, 2))
    if not np.any(z_any):
        return None
    y_any = np.any(mask, axis=(0, 2))
    x_any = np.any(mask, axis=(0, 1))
    z_idx = np.flatnonzero(z_any)
    y_idx = np.flatnonzero(y_any)
    x_idx = np.flatnonzero(x_any)
    return int(z_idx[0]), int(z_idx[-1]) + 1, int(y_idx[0]), int(y_idx[-1]) + 1, int(x_idx[0]), int(x_idx[-1]) + 1
