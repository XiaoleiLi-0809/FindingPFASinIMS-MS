from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from core_calculate_experimental_riMp1_isotope_envelope import (
    calculate_first_isotopic_peak_ratio,
)
from pfas_screening_app.engine import (
    ProcessingConfig,
    fit_mass_calibration,
    fit_cims_calibration,
    match_pfas_database,
    pair_first_isotopic_peaks,
    prepare_pfas_database,
    prioritize_pfas_peaks,
    process_file,
    reduced_mass,
    theoretical_isotope_ratios,
)


class EngineTests(unittest.TestCase):
    def test_theoretical_isotope_ratios(self) -> None:
        ratios, status = theoretical_isotope_ratios("CCl")
        self.assertEqual(status, "calculated")
        self.assertAlmostEqual(
            ratios[1],
            100.0 * 0.0107 / 0.9893,
            places=6,
        )
        self.assertAlmostEqual(
            ratios[2],
            100.0 * 0.2424 / 0.7576,
            places=6,
        )

    def test_priority_filter_and_database_matching(self) -> None:
        peaks = pd.DataFrame(
            {
                "PFAS_level": [2, 1, 3, 0],
                "PFAS_level_name": ["L2", "L1", "L3", "L0"],
                "PFAS_isotope_role": ["M", "M", "M+1", "M"],
                "PFAS_M_intensity": [5000.0, 2000.0, 9000.0, 8000.0],
                "PFAS_prob_mean": [0.9, 0.8, 0.95, 0.1],
                "PFAS_prob_p05": [0.8, 0.7, 0.9, 0.05],
                "PFAS_mz_used": [101.007276466621, 151.0, 201.0, 301.0],
                "PFAS_rMp1": [1.1, 2.0, 3.0, 4.0],
                "PFAS_rMp2": [32.0, 1.0, 2.0, 3.0],
                "PFAS_rMp3": [0.3, 0.2, 0.1, 0.0],
                "PFAS_rMp4": [10.0, 0.0, 0.0, 0.0],
            }
        )
        prioritized = prioritize_pfas_peaks(peaks, minimum_intensity=1000, top_n=1)
        self.assertEqual(len(prioritized), 1)
        self.assertEqual(prioritized.index[0], 0)

        database = prepare_pfas_database(
            pd.DataFrame(
                {
                    "PREFERRED NAME": ["Test PFAS"],
                    "MOLECULAR FORMULA": ["CCl"],
                    "MONOISOTOPIC MASS": [100.0],
                    "SMILES": ["CCl"],
                    "DTXSID": ["DTXSIDTEST"],
                    "CASRN": ["1-00-0"],
                    "INCHIKEY": ["TEST"],
                }
            )
        )
        matches = match_pfas_database(
            prioritized,
            database,
            mode="POS",
            tolerance_ppm=5.0,
        )
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches.loc[0, "PFAS_ion_hypothesis"], "[M+H]+")
        self.assertAlmostEqual(matches.loc[0, "PFAS_mass_error_ppm"], 0.0)
        self.assertEqual(matches.loc[0, "molecular_formula"], "CCl")

        negative_peaks = prioritized.copy()
        negative_peaks["PFAS_mz_used"] = 100.0 - 1.007276466621
        negative_matches = match_pfas_database(
            negative_peaks,
            database,
            mode="NEG",
            tolerance_ppm=5.0,
        )
        self.assertEqual(len(negative_matches), 1)
        self.assertEqual(
            negative_matches.loc[0, "PFAS_ion_hypothesis"],
            "[M-H]-",
        )
        self.assertAlmostEqual(
            negative_matches.loc[0, "PFAS_mass_error_ppm"],
            0.0,
        )

    def test_lock_mass_calibration(self) -> None:
        standards = pd.DataFrame(
            {
                "observed": [100.001, 200.002],
                "exact": [100.0, 200.0],
            }
        )
        apply, formula, r2, table, _, _ = fit_mass_calibration(
            standards, "observed", "exact", "lock"
        )
        corrected = apply(pd.Series([300.003])).iloc[0]
        self.assertAlmostEqual(corrected, 300.0, places=5)
        self.assertIn("lock mass", formula)
        self.assertIsNone(r2)
        self.assertIn("PFAS_mass_residual_ppm", table.columns)

    def test_linear_mass_calibration(self) -> None:
        exact = np.array([100.0, 300.0, 500.0])
        ppm_error = np.array([2.0, 4.0, 6.0])
        observed = exact * (1.0 + ppm_error / 1e6)
        standards = pd.DataFrame({"observed": observed, "exact": exact})
        apply, formula, r2, _, _, _ = fit_mass_calibration(
            standards, "observed", "exact", "linear"
        )
        sample_exact = 400.0
        sample_ppm = 5.0
        sample_observed = sample_exact * (1.0 + sample_ppm / 1e6)
        corrected = apply(pd.Series([sample_observed])).iloc[0]
        self.assertAlmostEqual(corrected, sample_exact, places=4)
        self.assertGreater(r2, 0.99)
        self.assertIn("linear ppm", formula)

    def test_cims_reduced_ccs_calibration(self) -> None:
        tc = np.array([1.0, 2.0, 3.0, 4.0])
        mz = np.array([100.0, 200.0, 400.0, 800.0])
        charge = np.ones(4)
        mu = reduced_mass(mz, charge)
        reduced_ccs = 5.0 * tc + 40.0
        ccs = reduced_ccs / np.sqrt(mu)
        standards = pd.DataFrame(
            {
                "tc": tc,
                "mz": mz,
                "CCS": ccs,
            }
        )
        slope, intercept, r2, table = fit_cims_calibration(
            standards, "tc", "mz", "CCS"
        )
        self.assertAlmostEqual(slope, 5.0)
        self.assertAlmostEqual(intercept, 40.0)
        self.assertAlmostEqual(r2, 1.0)
        np.testing.assert_allclose(table["PFAS_CCS_residual"], 0.0, atol=1e-10)

    def test_end_to_end_cims_calibration(self) -> None:
        standards_tc = np.array([1.0, 2.0, 3.0, 4.0])
        standards_mz = np.array([100.0, 200.0, 400.0, 800.0])
        standards_mu = reduced_mass(standards_mz, np.ones(4))
        standards_ccs = (4.0 * standards_tc + 50.0) / np.sqrt(standards_mu)
        standards = pd.DataFrame(
            {
                "tc": standards_tc,
                "mz": standards_mz,
                "CCS": standards_ccs,
            }
        )
        sample = pd.DataFrame(
            {
                "tc": [2.5],
                "mz": [300.0],
                "riMp1": [12.0],
            }
        )
        expected_ccs = (4.0 * 2.5 + 50.0) / np.sqrt(
            reduced_mass(np.array([300.0]), np.array([1.0]))[0]
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            standards_path = root / "standards.xlsx"
            input_path = root / "input.xlsx"
            output_path = root / "output.xlsx"
            standards.to_excel(standards_path, index=False)
            sample.to_excel(input_path, index=False)
            summary = process_file(
                ProcessingConfig(
                    input_path=input_path,
                    output_path=output_path,
                    ion_mode="POS",
                    first_calibration_method="cims_reduced_ccs",
                    second_calibration_method="none",
                    rimp1_method="existing",
                    standards_path=standards_path,
                    standards_tc_column="tc",
                    standards_mz_column="mz",
                    standards_ccs_column="CCS",
                    tc_column="tc",
                    mz_column="mz",
                    rimp1_column="riMp1",
                    mc_iterations=1,
                )
            )
            self.assertEqual(summary.predictable_rows, 1)
            output = pd.read_excel(output_path, sheet_name="Labeled data")
            self.assertAlmostEqual(
                output.loc[0, "PFAS_CCS_stage1"], expected_ccs, places=8
            )
            self.assertFalse(bool(output.loc[0, "PFAS_calibration_extrapolated"]))
            self.assertFalse(summary.second_calibration_applied)

    def test_end_to_end_mass_calibration(self) -> None:
        source = pd.DataFrame(
            {
                "mz": [250.0025, 500.005],
                "CCS": [155.0, 220.0],
                "riMp1": [12.0, 25.0],
            }
        )
        mass_standards = pd.DataFrame(
            {
                "observed": [100.001, 600.006],
                "exact": [100.0, 600.0],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.xlsx"
            mass_path = root / "mass.xlsx"
            output_path = root / "output.xlsx"
            source.to_excel(input_path, index=False)
            mass_standards.to_excel(mass_path, index=False)
            summary = process_file(
                ProcessingConfig(
                    input_path=input_path,
                    output_path=output_path,
                    ion_mode="POS",
                    mass_calibration_method="linear",
                    mass_standards_path=mass_path,
                    mass_observed_mz_column="observed",
                    mass_exact_mz_column="exact",
                    first_calibration_method="existing_ccs",
                    second_calibration_method="none",
                    rimp1_method="existing",
                    mc_iterations=1,
                )
            )
            self.assertEqual(summary.predictable_rows, 2)
            output = pd.read_excel(output_path, sheet_name="Labeled data")
            self.assertIn("PFAS_mz_raw", output.columns)
            self.assertIn("PFAS_mz_calibrated", output.columns)
            self.assertAlmostEqual(output.loc[0, "PFAS_mz_calibrated"], 250.0, places=4)
            with pd.ExcelFile(output_path) as workbook:
                self.assertIn("Mass calibration", workbook.sheet_names)

    def test_first_isotopic_peak_pairing_and_ratio(self) -> None:
        source = pd.DataFrame(
            {
                "mz": [
                    100.0,
                    101.0020,
                    101.0033,
                    101.9970,
                    102.0067,
                    102.0190,
                    102.9990,
                    103.9940,
                    104.0130,
                    150.0,
                ],
                "rt": [
                    5.0,
                    5.001,
                    5.009,
                    5.002,
                    4.997,
                    5.030,
                    5.004,
                    4.996,
                    5.006,
                    9.0,
                ],
                "3D_TC": [
                    20.0,
                    20.010,
                    20.020,
                    20.010,
                    20.020,
                    20.030,
                    20.030,
                    20.040,
                    20.050,
                    30.0,
                ],
                "intensity": [
                    1000.0,
                    250.0,
                    125.0,
                    40.0,
                    60.0,
                    500.0,
                    30.0,
                    10.0,
                    20.0,
                    500.0,
                ],
            }
        )
        result = pair_first_isotopic_peaks(
            source,
            "mz",
            "rt",
            "3D_TC",
            "intensity",
        )
        self.assertAlmostEqual(
            result.loc[0, "first_isotopic_peak_ratio"],
            37.5,
        )
        self.assertEqual(
            result.loc[0, "PFAS_M_plus_1_candidate_count"],
            2,
        )
        self.assertTrue(result.loc[0, "PFAS_M_plus_1_ambiguous"])
        self.assertAlmostEqual(
            result.loc[0, "PFAS_first_13C_peak_mz"],
            101.0033,
        )
        self.assertAlmostEqual(result.loc[0, "PFAS_first_13C_peak_ratio"], 12.5)
        self.assertAlmostEqual(result.loc[0, "PFAS_rMp1"], 37.5)
        self.assertAlmostEqual(result.loc[0, "PFAS_rMp2"], 10.0)
        self.assertAlmostEqual(result.loc[0, "PFAS_rMp3"], 3.0)
        self.assertAlmostEqual(result.loc[0, "PFAS_rMp4"], 3.0)
        self.assertEqual(result.loc[0, "PFAS_M_plus_2_candidate_count"], 2)
        self.assertTrue(result.loc[0, "PFAS_M_plus_2_ambiguous"])
        self.assertAlmostEqual(
            result.loc[0, "PFAS_M_plus_2_intensity_sum"],
            100.0,
        )
        self.assertEqual(result.loc[1, "PFAS_isotope_role"], "M+1")
        self.assertTrue(pd.isna(result.loc[1, "PFAS_rMp1"]))
        self.assertTrue(pd.isna(result.loc[9, "first_isotopic_peak_ratio"]))

    def test_same_mz_rt_different_tc_remain_independent_m_features(self) -> None:
        source = pd.DataFrame(
            {
                "mz": [100.0, 100.0, 101.0033, 101.0033],
                "rt": [5.0, 5.0, 5.001, 5.001],
                "3D_TC": [20.0, 30.0, 20.010, 30.010],
                "intensity": [1000.0, 2000.0, 100.0, 300.0],
            }
        )
        result = pair_first_isotopic_peaks(
            source,
            "mz",
            "rt",
            "3D_TC",
            "intensity",
        )

        self.assertEqual(result.loc[0, "PFAS_isotope_role"], "M")
        self.assertEqual(result.loc[1, "PFAS_isotope_role"], "M")
        self.assertEqual(result.loc[2, "PFAS_isotope_role"], "M+1")
        self.assertEqual(result.loc[3, "PFAS_isotope_role"], "M+1")
        self.assertEqual(result.loc[2, "PFAS_isotope_envelope_base_sorted_index"], 0)
        self.assertEqual(result.loc[3, "PFAS_isotope_envelope_base_sorted_index"], 1)
        self.assertAlmostEqual(result.loc[0, "PFAS_rMp1"], 10.0)
        self.assertAlmostEqual(result.loc[1, "PFAS_rMp1"], 15.0)

    def test_standalone_ratio_script_keeps_tc_separated_m_features(self) -> None:
        source = pd.DataFrame(
            {
                "mz": [100.0, 100.0, 101.0033, 101.0033],
                "rt": [5.0, 5.0, 5.001, 5.001],
                "tc": [20.0, 30.0, 20.010, 30.010],
                "intensity": [1000.0, 2000.0, 100.0, 300.0],
            }
        )
        result = calculate_first_isotopic_peak_ratio(source)

        self.assertEqual(result.loc[0, "isotope_role"], "M")
        self.assertEqual(result.loc[1, "isotope_role"], "M")
        self.assertEqual(result.loc[2, "isotope_role"], "M+1")
        self.assertEqual(result.loc[3, "isotope_role"], "M+1")
        self.assertEqual(result.loc[2, "isotope_envelope_base_sorted_index"], 0)
        self.assertEqual(result.loc[3, "isotope_envelope_base_sorted_index"], 1)
        self.assertAlmostEqual(result.loc[0, "rMp1"], 10.0)
        self.assertAlmostEqual(result.loc[1, "rMp1"], 15.0)

    def test_end_to_end_existing_ccs(self) -> None:
        source = pd.DataFrame(
            {
                "mz": [250.0, 500.0],
                "CCS": [155.0, 220.0],
                "riMp1": [12.0, 25.0],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.xlsx"
            output_path = root / "output.xlsx"
            source.to_excel(input_path, index=False)
            summary = process_file(
                ProcessingConfig(
                    input_path=input_path,
                    output_path=output_path,
                    ion_mode="POS",
                    first_calibration_method="existing_ccs",
                    second_calibration_method="none",
                    rimp1_method="existing",
                    mc_iterations=2,
                )
            )
            self.assertEqual(summary.predictable_rows, 2)
            self.assertTrue(output_path.exists())
            output = pd.read_excel(output_path, sheet_name="Labeled data")
            self.assertIn("PFAS_level_name", output.columns)
            self.assertIn("PFAS_CCS_model_input", output.columns)


if __name__ == "__main__":
    unittest.main()
