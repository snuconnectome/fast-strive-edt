"""Command-Line Interface for Fast-STRIVE-EDT."""

import json
import time
import argparse
import numpy as np

from .edt3d import edt3d, scipy_edt3d
from .strive import partition_lesion_topography


def run_verify_phantom() -> int:
    """Run synthetic anisotropic phantom verification test."""
    vol = np.zeros((48, 64, 48), dtype=np.int16)
    vol[24, 24, 24] = 4  # Lateral ventricle seed
    vol[20, 40, 24] = 3  # Cerebral cortex seed
    # Voxel 1: 6 voxels along Z (6 * 2.0 = 12.0 mm > 10.0 mm from V, > 4.0 mm from C -> Deep Subcortical)
    vol[30, 24, 24] = 77
    # Voxel 2: 16 voxels along Y (16 * 0.5 = 8.0 mm <= 10.0 mm from V -> Periventricular)
    vol[24, 40, 24] = 77
    # Voxel 3: near cortex at (20, 40, 26) -> 1.0 mm from cortex, > 10.0 mm from V -> Juxtacortical
    vol[20, 40, 26] = 77

    res = partition_lesion_topography(vol, spacing=(2.0, 0.5, 0.5), backend="auto")
    print(json.dumps(res, indent=2))
    assert res["status"] == "OK"
    assert res["n_lesion_vox"] == 3
    assert res["pv_vox_10mm"] == 1
    assert res["deep_vox_10mm"] == 2
    assert res["juxtacortical_vox"] == 1
    assert res["deep_subcortical_vox"] == 1
    print("[PASS] Synthetic anisotropic phantom verification succeeded.")
    return 0


def run_benchmark(shape=(128, 160, 128), repeats=10) -> int:
    """Benchmark 3D EDT on a synthetic brain volume."""
    np.random.seed(42)
    fg = np.zeros(shape, dtype=bool)
    d, h, w = shape
    fg[d // 3 : 2 * d // 3, h // 3 : 2 * h // 3, w // 3 : 2 * w // 3] = True
    spacing = (1.0, 1.2, 0.9)

    t0 = time.perf_counter()
    dt_ref = scipy_edt3d(fg, spacing=spacing)
    ms_scipy = (time.perf_counter() - t0) * 1000.0

    # Warmup auto backend
    _ = edt3d(fg, spacing=spacing, backend="auto")
    t0 = time.perf_counter()
    for _ in range(repeats):
        dt_auto = edt3d(fg, spacing=spacing, backend="auto")
    ms_auto = (time.perf_counter() - t0) * 1000.0 / repeats

    max_err = float(np.max(np.abs(dt_ref - dt_auto)))
    print(
        json.dumps(
            {
                "shape": list(shape),
                "spacing_mm": list(spacing),
                "scipy_cpu_ms": round(ms_scipy, 3),
                "auto_backend_ms": round(ms_auto, 3),
                "speedup": round(ms_scipy / max(ms_auto, 1e-6), 2),
                "max_abs_err_mm": max_err,
            },
            indent=2,
        )
    )
    return 0


def main():
    parser = argparse.ArgumentParser(prog="fast-strive-edt", description="Fast-STRIVE-EDT CLI")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("verify-phantom", help="Verify exactness on anisotropic 3D phantom")
    b_p = sub.add_parser("benchmark", help="Benchmark 3D EDT speed and accuracy")
    b_p.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()

    if args.cmd == "benchmark":
        raise SystemExit(run_benchmark(repeats=args.repeats))
    else:
        raise SystemExit(run_verify_phantom())


if __name__ == "__main__":
    main()
