"""Unit and mathematical invariant tests for Fast-STRIVE-EDT."""

from pathlib import Path
import os
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fast_strive_edt import (
    scipy_edt3d,
    pytorch_edt3d,
    edt3d,
    compute_affine_shear,
    fast_bbox_3d,
    partition_lesion_topography,
    compute_wmh_burden_tier,
    evaluate_wmh_qc_flag,
    normalize_session_id,
    SQLiteWALBatchLedger,
)


class TestFastStriveEDT(unittest.TestCase):

    def test_01_anisotropic_spacing_and_padding_exactness(self):
        """Anisotropic spacing (2.0, 0.5, 0.5) mm and physical >12-voxel padding are exact."""
        vol = np.zeros((48, 64, 48), dtype=np.int16)
        vol[24, 24, 24] = 4  # Lateral ventricle
        vol[20, 40, 24] = 3  # Cerebral cortex
        vol[30, 24, 24] = 77  # 6 * 2.0 = 12.0 mm -> Deep Subcortical
        vol[24, 40, 24] = 77  # 16 * 0.5 = 8.0 mm -> Periventricular (<= 10.0 mm)
        vol[20, 40, 26] = 77  # 1.0 mm from cortex, > 10.0 mm from V -> Juxtacortical

        res = partition_lesion_topography(vol, spacing=(2.0, 0.5, 0.5), backend="scipy")
        self.assertEqual(res["status"], "OK")
        self.assertEqual(res["n_lesion_vox"], 3)
        self.assertEqual(res["pv_vox_10mm"], 1)
        self.assertEqual(res["deep_vox_10mm"], 2)
        self.assertEqual(res["juxtacortical_vox"], 1)
        self.assertEqual(res["deep_subcortical_vox"], 1)
        self.assertAlmostEqual(res["total_mask_vol_mm3"], 1.5, places=5)

    def test_02_pytorch_vs_scipy_exact_agreement(self):
        """pytorch_edt3d matches scipy_edt3d to single-precision floating point accuracy."""
        import torch
        np.random.seed(7)
        fg = np.zeros((24, 28, 20), dtype=bool)
        fg[8:14, 10:16, 6:12] = True
        spacing = (1.5, 0.8, 1.1)
        dt_ref = scipy_edt3d(fg, spacing=spacing)
        dt_pt = pytorch_edt3d(torch.from_numpy(fg), spacing=spacing).numpy()
        self.assertLess(float(np.max(np.abs(dt_ref - dt_pt))), 1e-4)

    def test_03_26cc_continuity_confluent_extension(self):
        """A connected lesion chain starting at 2mm from ventricle and extending to 14mm is 100% PV under continuity."""
        vol = np.zeros((32, 32, 32), dtype=np.int16)
        vol[16, 16, 5] = 4  # Ventricle at x=5
        # Connected chain from x=7 (2mm from V) to x=20 (15mm from V)
        vol[16, 16, 7:21] = 77
        # Isolated deep lesion at x=25 (20mm from V)
        vol[16, 16, 25] = 77

        res = partition_lesion_topography(vol, spacing=(1.0, 1.0, 1.0), backend="scipy")
        self.assertEqual(res["n_lesion_vox"], 15)
        self.assertEqual(res["n_lesion_clusters"], 2)
        # Under 10mm distance rule: x=7..15 (9 voxels) are <=10mm
        self.assertEqual(res["pv_vox_10mm"], 9)
        # Under 26-CC continuity rule: entire 14-voxel chain touching <=3mm is PV
        self.assertEqual(res["pv_vox_continuity"], 14)

    def test_04_ventricle_mode_strict_vs_full(self):
        """vent_mode='full' includes inferior lateral ventricles (label 5) whereas 'strict' includes only [4, 43]."""
        vol = np.zeros((32, 32, 32), dtype=np.int16)
        vol[10, 10, 10] = 4   # Lateral ventricle
        vol[25, 25, 25] = 5   # Inferior lateral ventricle
        vol[25, 25, 27] = 77  # 2mm from Inf Lat Vent, >20mm from Lat Vent

        res_strict = partition_lesion_topography(vol, vent_mode="strict", backend="scipy")
        res_full = partition_lesion_topography(vol, vent_mode="full", backend="scipy")
        self.assertEqual(res_strict["pv_vox_10mm"], 0)
        self.assertEqual(res_full["pv_vox_10mm"], 1)

    def test_05_zero_lesion_and_missing_ventricle_handling(self):
        """Zero-lesion scan returns status='OK' and PASS QC; missing ventricle returns status='NO_VENT' and FAIL QC."""
        vol_healthy = np.zeros((16, 16, 16), dtype=np.int16)
        vol_healthy[8, 8, 8] = 4
        res_ok = partition_lesion_topography(vol_healthy, backend="scipy")
        self.assertEqual(res_ok["status"], "OK")
        self.assertEqual(res_ok["n_lesion_vox"], 0)
        self.assertEqual(evaluate_wmh_qc_flag(0.0, 0.0, status=res_ok["status"]), "PASS")

        vol_no_vent = np.zeros((16, 16, 16), dtype=np.int16)
        vol_no_vent[8, 8, 8] = 77
        res_nv = partition_lesion_topography(vol_no_vent, backend="scipy")
        self.assertEqual(res_nv["status"], "NO_VENT")
        self.assertEqual(evaluate_wmh_qc_flag(10.0, 0.001, status=res_nv["status"]), "FAIL")

    def test_06_pre_clipping_qc_and_affine_shear_guard(self):
        """QC flag catches raw_etiv_ratio > 0.15 and oblique affine shear > 1e-3."""
        self.assertEqual(evaluate_wmh_qc_flag(200000.0, 0.18, status="OK"), "FAIL")
        self.assertEqual(evaluate_wmh_qc_flag(40000.0, 0.02, status="OK"), "WARN")
        ortho_aff = np.diag([1.0, 1.2, 0.9, 1.0])
        self.assertLess(compute_affine_shear(ortho_aff), 1e-9)
        sheared_aff = np.array([[1.0, 0.2, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        self.assertGreater(compute_affine_shear(sheared_aff), 0.1)
        self.assertEqual(evaluate_wmh_qc_flag(1000.0, 0.001, status="OK", off_diag_shear=compute_affine_shear(sheared_aff)), "FAIL")

    def test_07_normalize_session_id_and_burden_tiers(self):
        """normalize_session_id strips '.0' and compute_wmh_burden_tier maps thresholds accurately."""
        self.assertEqual(normalize_session_id("14051808.0"), "14051808")
        self.assertEqual(normalize_session_id(14051808.0), "14051808")
        self.assertEqual(compute_wmh_burden_tier(0.0), 0)
        self.assertEqual(compute_wmh_burden_tier(499.9), 0)
        self.assertEqual(compute_wmh_burden_tier(500.0), 1)
        self.assertEqual(compute_wmh_burden_tier(3000.0), 2)
        self.assertEqual(compute_wmh_burden_tier(10000.0), 3)

    def test_08_sqlite_wal_batch_ledger(self):
        """SQLiteWALBatchLedger records and queries sessions in WAL mode."""
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "test.sqlite")
            ledger = SQLiteWALBatchLedger(db)
            ledger.record_session("14051808.0", "gpu0", "OK", 500, 475, 25, 2.44)
            done = ledger.get_completed_sessions()
            self.assertEqual(done, {"14051808"})


if __name__ == "__main__":
    unittest.main()
