from __future__ import annotations

import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import joblib
from openpyxl.chart import Reference, ScatterChart, Series


FIRST_ISOTOPE_SPACING_DA = 1.00335483507
MASS_TOLERANCE_DA = 0.010
HIGHER_ISOTOPE_WINDOW_DA = 0.020
RT_TOLERANCE_MIN = 0.020
TC_TOLERANCE = 0.060
ISOTOPE_ORDERS = (1, 2, 3, 4)
NITROGEN_MASS_DA = 28.0134
PROTON_MASS_DA = 1.007276466621
DEFAULT_DATABASE_FILENAME = "Chemical List PFAS.xlsx"

# Nominal isotope shifts and natural abundances relative to the light
# monoisotopic isotope. Elements absent from this map are reported as
# unsupported rather than assigned a misleading theoretical distribution.
ISOTOPE_ABUNDANCES = {
    "H": {0: 0.999885, 1: 0.000115},
    "C": {0: 0.9893, 1: 0.0107},
    "N": {0: 0.99636, 1: 0.00364},
    "O": {0: 0.99757, 1: 0.00038, 2: 0.00205},
    "Si": {0: 0.92223, 1: 0.04685, 2: 0.03092},
    "S": {0: 0.9499, 1: 0.0075, 2: 0.0425, 4: 0.0001},
    "Cl": {0: 0.7576, 2: 0.2424},
    "Br": {0: 0.5069, 2: 0.4931},
    "F": {0: 1.0},
    "Na": {0: 1.0},
    "Al": {0: 1.0},
    "P": {0: 1.0},
    "I": {0: 1.0},
    "Au": {0: 1.0},
}

CALIBRATION = {
    "POS": (0.8289, 38.258),
    "NEG": (0.9585, 9.3196),
}

FEATURE_COLS = [
    "mz_percentile",
    "mz",
    "CCS",
    "riMp1",
    "CCS_over_m",
    "CCS_over_Mp1",
    "M_over_Mp1",
    "CCS_over_sqrtm",
    "CCS_over_m23",
]

COLUMN_ALIASES = {
    "mz": [
        "mz",
        "m/z",
        "PrecursorMz",
        "3D_m_z",
        "3D_m_z_uncal",
        "mz_M+H",
        "mz_M-H",
    ],
    "observed_mz": [
        "observed_mz",
        "observed m/z",
        "measured_mz",
        "measured m/z",
        "mz_obs",
        "m/z observed",
        "m/z",
        "mz",
    ],
    "exact_mz": [
        "exact_mz",
        "exact m/z",
        "reference_mz",
        "reference m/z",
        "known_mz",
        "known m/z",
        "theoretical_mz",
        "theoretical m/z",
    ],
    "ccs": [
        "CCS",
        "CCS(Progensis)",
        "Reference CCS",
        "Standard CCS",
        "PrecursorCCS",
        "CCS_correct_linear",
        "CCS_M+H_ALLCCS2",
        "CCS_M-H_ALLCCS2",
        "CCS_M+H",
        "CCS_M-H",
    ],
    "tc": [
        "tc",
        "t_c",
        "corrected drift time",
        "dt",
        "drift time",
        "3D_omegaD",
        "3D_omegaD_corr_3",
    ],
    "charge": ["charge", "z", "PrecursorCharge", "3D_z"],
    "rt": ["rt", "RT", "retention time", "Retention time (min)", "PrecursorRT", "3D_RetTime"],
    "intensity": ["intensity", "Intensity", "intb", "M", "3D_Intensity", "3D_Mobility_Mass"],
    "rimp1": [
        "first_isotopic_peak_ratio",
        "riMp1",
        "rimp1",
        "Mp1int",
        "Ap1intNorm",
    ],
    "m_intensity": ["M", "intensity", "Intensity", "intb", "3D_Intensity", "3D_Mobility_Mass"],
    "mp1_intensity": ["M+1", "intensity_mp1", "M+1_intensity"],
}


class ProcessingError(RuntimeError):
    """Raised when an input table cannot be processed safely."""


@dataclass(frozen=True)
class ProcessingConfig:
    input_path: Path
    output_path: Path
    sheet_name: str | int = 0
    ion_mode: str = "POS"
    rimp1_method: str = "auto"
    mass_calibration_method: str = "none"
    first_calibration_method: str = "cims_reduced_ccs"
    second_calibration_method: str = "none"
    mz_column: str | None = None
    ccs_column: str | None = None
    tc_column: str | None = None
    charge_column: str | None = None
    default_charge: int = 1
    rt_column: str | None = None
    intensity_column: str | None = None
    rimp1_column: str | None = None
    m_intensity_column: str | None = None
    mp1_intensity_column: str | None = None
    standards_path: Path | None = None
    standards_sheet_name: str | int = 0
    standards_tc_column: str | None = None
    standards_mz_column: str | None = None
    standards_ccs_column: str | None = None
    standards_charge_column: str | None = None
    mass_standards_path: Path | None = None
    mass_standards_sheet_name: str | int = 0
    mass_observed_mz_column: str | None = None
    mass_exact_mz_column: str | None = None
    minimum_m_intensity: float = 0.0
    priority_top_n: int = 200
    database_path: Path | None = None
    database_sheet_name: str | int = 0
    database_mass_tolerance_ppm: float = 5.0
    mc_iterations: int = 200


@dataclass(frozen=True)
class ProcessingSummary:
    input_rows: int
    predictable_rows: int
    level_counts: dict[str, int]
    rimp1_method: str
    first_calibration_formula: str
    first_calibration_r2: float | None
    second_calibration_applied: bool
    output_path: Path


def _resource_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS"))
    source_path = Path(__file__).resolve()
    if source_path.parents[1].name == "code_for_submission":
        return source_path.parents[2]
    return source_path.parents[1]


def model_path(mode: str) -> Path:
    joblib_filename = f"rf_sklearn_bundle_{mode}.joblib"
    pickle_filename = f"rf_sklearn_bundle_{mode}.pkl"
    candidates = [
        _resource_root() / "models" / joblib_filename,
        _resource_root() / "models" / pickle_filename,
        _resource_root()
        / "outputs_20260604_reanalysis_8"
        / "models"
        / pickle_filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise ProcessingError(
        f"Model file was not found: {joblib_filename} or {pickle_filename}"
    )


def default_database_path() -> Path:
    """Return the bundled PFAS database path."""
    return _resource_root() / DEFAULT_DATABASE_FILENAME


def load_bundle(mode: str) -> dict:
    mode = mode.upper()
    if mode not in CALIBRATION:
        raise ProcessingError(f"Unsupported ion mode: {mode}")
    path = model_path(mode)
    if path.suffix.lower() == ".joblib":
        return joblib.load(path)
    with path.open("rb") as handle:
        return pickle.load(handle)


def list_excel_sheets(path: Path) -> list[str]:
    try:
        return pd.ExcelFile(path).sheet_names
    except Exception as exc:
        raise ProcessingError(f"Cannot read Excel workbook: {exc}") from exc


def read_input_table(path: Path, sheet_name: str | int = 0) -> pd.DataFrame:
    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        return pd.read_excel(path, sheet_name=sheet_name)
    except Exception as exc:
        raise ProcessingError(f"Cannot read input table: {exc}") from exc


def _normalized(value: object) -> str:
    return "".join(str(value).strip().lower().split())


def detect_column(columns: list[object], field: str, mode: str | None = None) -> str | None:
    names = [str(column) for column in columns]
    normalized = {_normalized(column): column for column in names}
    aliases = list(COLUMN_ALIASES[field])

    if field == "mz" and mode:
        aliases = (["mz_M+H"] if mode == "POS" else ["mz_M-H"]) + aliases
    if field == "ccs" and mode:
        aliases = (
            ["CCS_M+H_ALLCCS2", "CCS_M+H"] if mode == "POS"
            else ["CCS_M-H_ALLCCS2", "CCS_M-H"]
        ) + aliases

    for alias in aliases:
        match = normalized.get(_normalized(alias))
        if match is not None:
            return match
    return None


def suggested_columns(df: pd.DataFrame, mode: str) -> dict[str, str | None]:
    columns = list(df.columns)
    return {field: detect_column(columns, field, mode) for field in COLUMN_ALIASES}


def _selected_or_detected(
    df: pd.DataFrame,
    selected: str | None,
    field: str,
    mode: str,
    required: bool = True,
) -> str | None:
    if selected and selected in df.columns:
        return selected
    detected = detect_column(list(df.columns), field, mode)
    if required and detected is None:
        raise ProcessingError(
            f"Could not identify the {field} column. Select it in Advanced column mapping."
        )
    return detected


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def parse_molecular_formula(formula: object) -> dict[str, int]:
    """Parse a simple molecular formula such as C8H5ClF16O3S."""
    text = str(formula).strip()
    if not text or text.lower() == "nan":
        raise ValueError("Molecular formula is blank.")
    tokens = re.findall(r"([A-Z][a-z]?)(\d*)", text)
    if not tokens or "".join(element + count for element, count in tokens) != text:
        raise ValueError(f"Unsupported molecular formula format: {text}")
    composition: dict[str, int] = {}
    for element, count_text in tokens:
        count = int(count_text) if count_text else 1
        composition[element] = composition.get(element, 0) + count
    return composition


def theoretical_isotope_ratios(
    formula: object,
    hydrogen_adjustment: int = 0,
    max_order: int = 4,
) -> tuple[dict[int, float], str]:
    """Calculate nominal M+n/M ratios from natural isotope abundances."""
    try:
        composition = parse_molecular_formula(formula)
    except ValueError as exc:
        return {}, str(exc)

    if hydrogen_adjustment:
        composition["H"] = composition.get("H", 0) + hydrogen_adjustment
        if composition["H"] < 0:
            return {}, "Formula does not contain enough hydrogen for [M-H]-."
        if composition["H"] == 0:
            composition.pop("H")

    unsupported = sorted(set(composition) - set(ISOTOPE_ABUNDANCES))
    if unsupported:
        return {}, "Unsupported isotope elements: " + ", ".join(unsupported)

    distribution = np.zeros(max_order + 1, dtype=float)
    distribution[0] = 1.0
    for element, count in composition.items():
        atom_profile = ISOTOPE_ABUNDANCES[element]
        for _ in range(count):
            updated = np.zeros(max_order + 1, dtype=float)
            for current_order, current_probability in enumerate(distribution):
                if current_probability == 0:
                    continue
                for shift, abundance in atom_profile.items():
                    if current_order + shift <= max_order:
                        updated[current_order + shift] += (
                            current_probability * abundance
                        )
            distribution = updated

    if distribution[0] <= 0:
        return {}, "Monoisotopic abundance is zero under the configured profile."
    ratios = {
        order: float(100.0 * distribution[order] / distribution[0])
        for order in range(1, max_order + 1)
    }
    return ratios, "calculated"


def _database_column(
    data: pd.DataFrame,
    aliases: list[str],
    required: bool = True,
) -> str | None:
    normalized = {_normalized(column): str(column) for column in data.columns}
    for alias in aliases:
        match = normalized.get(_normalized(alias))
        if match is not None:
            return match
    if required:
        raise ProcessingError(
            "PFAS database is missing a required column. Expected one of: "
            + ", ".join(aliases)
        )
    return None


def prepare_pfas_database(data: pd.DataFrame) -> pd.DataFrame:
    """Standardize the Chemical List PFAS database schema."""
    mass_column = _database_column(
        data,
        ["MONOISOTOPIC MASS", "ExactMonoisotopicMass", "MonoisotopicMass"],
    )
    formula_column = _database_column(
        data,
        ["MOLECULAR FORMULA", "Molecula_formula2", "Molecula_formula"],
    )
    name_column = _database_column(
        data,
        ["PREFERRED NAME", "Chemical Name", "name"],
        required=False,
    )
    smiles_column = _database_column(
        data,
        ["SMILES", "SMILES_conical", "SMILES_canon", "QSAR Ready SMILES"],
        required=False,
    )
    dtxsid_column = _database_column(data, ["DTXSID"], required=False)
    cas_column = _database_column(data, ["CASRN", "CAS.."], required=False)
    inchikey_column = _database_column(data, ["INCHIKEY", "InChIKey"], required=False)

    database = pd.DataFrame(
        {
            "database_row": np.arange(2, len(data) + 2),
            "neutral_monoisotopic_mass": _numeric(data[mass_column]),
            "formula": data[formula_column].astype("string"),
            "name": (
                data[name_column].astype("string")
                if name_column
                else pd.Series(pd.NA, index=data.index, dtype="string")
            ),
            "smiles": (
                data[smiles_column].astype("string")
                if smiles_column
                else pd.Series(pd.NA, index=data.index, dtype="string")
            ),
            "dtxsid": (
                data[dtxsid_column].astype("string")
                if dtxsid_column
                else pd.Series(pd.NA, index=data.index, dtype="string")
            ),
            "casrn": (
                data[cas_column].astype("string")
                if cas_column
                else pd.Series(pd.NA, index=data.index, dtype="string")
            ),
            "inchikey": (
                data[inchikey_column].astype("string")
                if inchikey_column
                else pd.Series(pd.NA, index=data.index, dtype="string")
            ),
        }
    )
    database = database.dropna(subset=["neutral_monoisotopic_mass"]).copy()
    database = database[database["neutral_monoisotopic_mass"] > 0].copy()

    theory_rows: list[dict[str, object]] = []
    for formula in database["formula"]:
        ratios, status = theoretical_isotope_ratios(formula)
        theory_rows.append(
            {
                "theoretical_rMp1": ratios.get(1, np.nan),
                "theoretical_rMp2": ratios.get(2, np.nan),
                "theoretical_rMp3": ratios.get(3, np.nan),
                "theoretical_rMp4": ratios.get(4, np.nan),
                "theoretical_isotope_status": status,
            }
        )
    database = pd.concat(
        [database.reset_index(drop=True), pd.DataFrame(theory_rows)],
        axis=1,
    )
    return database.sort_values(
        "neutral_monoisotopic_mass", kind="mergesort"
    ).reset_index(drop=True)


def prioritize_pfas_peaks(
    data: pd.DataFrame,
    minimum_intensity: float,
    top_n: int,
) -> pd.DataFrame:
    """Select the most intense Level >=1 monoisotopic PFAS candidates."""
    level = _numeric(data["PFAS_level"])
    intensity = _numeric(data["PFAS_M_intensity"])
    isotope_role = (
        data["PFAS_isotope_role"].astype(str)
        if "PFAS_isotope_role" in data
        else pd.Series("M", index=data.index)
    )
    eligible = (
        (level >= 1)
        & (isotope_role == "M")
        & intensity.notna()
        & (intensity >= minimum_intensity)
    )
    prioritized = data.loc[eligible].copy()
    prioritized = prioritized.sort_values(
        ["PFAS_M_intensity", "PFAS_prob_mean"],
        ascending=[False, False],
        kind="mergesort",
    )
    if top_n > 0:
        prioritized = prioritized.head(top_n)
    prioritized.insert(
        0,
        "PFAS_intensity_rank",
        np.arange(1, len(prioritized) + 1),
    )
    return prioritized


def match_pfas_database(
    peaks: pd.DataFrame,
    database: pd.DataFrame,
    mode: str,
    tolerance_ppm: float,
) -> pd.DataFrame:
    """Match prioritized peaks to neutral PFAS masses under two ion hypotheses."""
    if peaks.empty or database.empty:
        return pd.DataFrame()

    mode = mode.upper()
    if mode == "POS":
        hypotheses = [
            ("[M+H]+", -PROTON_MASS_DA, 0),
            ("M+", 0.0, 0),
        ]
    elif mode == "NEG":
        hypotheses = [
            ("[M-H]-", PROTON_MASS_DA, -1),
            ("M-", 0.0, 0),
        ]
    else:
        raise ProcessingError(f"Unsupported ion mode for database matching: {mode}")

    db_mass = database["neutral_monoisotopic_mass"].to_numpy(float)
    matches: list[dict[str, object]] = []
    for peak_index, peak in peaks.iterrows():
        observed_mz = float(peak["PFAS_mz_used"])
        for ion_hypothesis, neutral_adjustment, hydrogen_adjustment in hypotheses:
            target_neutral_mass = observed_mz + neutral_adjustment
            mass_window = target_neutral_mass * tolerance_ppm / 1e6
            start = int(
                np.searchsorted(
                    db_mass,
                    target_neutral_mass - mass_window,
                    side="left",
                )
            )
            stop = int(
                np.searchsorted(
                    db_mass,
                    target_neutral_mass + mass_window,
                    side="right",
                )
            )
            for database_index in range(start, stop):
                candidate = database.iloc[database_index]
                neutral_mass = float(candidate["neutral_monoisotopic_mass"])
                theoretical_mz = (
                    neutral_mass + PROTON_MASS_DA
                    if ion_hypothesis == "[M+H]+"
                    else neutral_mass - PROTON_MASS_DA
                    if ion_hypothesis == "[M-H]-"
                    else neutral_mass
                )
                ppm_error = (
                    (observed_mz - theoretical_mz) / theoretical_mz * 1e6
                )

                ratios, theory_status = theoretical_isotope_ratios(
                    candidate["formula"],
                    hydrogen_adjustment=hydrogen_adjustment,
                )
                if not ratios:
                    ratios = {
                        order: candidate.get(f"theoretical_rMp{order}", np.nan)
                        for order in ISOTOPE_ORDERS
                    }
                    theory_status = str(
                        candidate.get("theoretical_isotope_status", theory_status)
                    )

                row = {
                    "PFAS_peak_output_index": int(peak_index),
                    "PFAS_intensity_rank": peak.get("PFAS_intensity_rank", np.nan),
                    "PFAS_ion_mode": mode,
                    "PFAS_ion_hypothesis": ion_hypothesis,
                    "PFAS_observed_mz": observed_mz,
                    "PFAS_target_neutral_mass": target_neutral_mass,
                    "PFAS_theoretical_mz": theoretical_mz,
                    "PFAS_mass_error_Da": observed_mz - theoretical_mz,
                    "PFAS_mass_error_ppm": ppm_error,
                    "PFAS_M_intensity": peak.get("PFAS_M_intensity", np.nan),
                    "PFAS_level": peak.get("PFAS_level", np.nan),
                    "PFAS_level_name": peak.get("PFAS_level_name", ""),
                    "PFAS_prob_mean": peak.get("PFAS_prob_mean", np.nan),
                    "PFAS_prob_p05": peak.get("PFAS_prob_p05", np.nan),
                    "experimental_rMp1": peak.get("PFAS_rMp1", np.nan),
                    "experimental_rMp2": peak.get("PFAS_rMp2", np.nan),
                    "experimental_rMp3": peak.get("PFAS_rMp3", np.nan),
                    "experimental_rMp4": peak.get("PFAS_rMp4", np.nan),
                    "database_row": candidate["database_row"],
                    "DTXSID": candidate["dtxsid"],
                    "CASRN": candidate["casrn"],
                    "preferred_name": candidate["name"],
                    "molecular_formula": candidate["formula"],
                    "neutral_monoisotopic_mass": neutral_mass,
                    "SMILES": candidate["smiles"],
                    "InChIKey": candidate["inchikey"],
                    "theoretical_isotope_status": theory_status,
                }
                for order in ISOTOPE_ORDERS:
                    theory_value = ratios.get(order, np.nan)
                    experimental_value = peak.get(f"PFAS_rMp{order}", np.nan)
                    row[f"theoretical_rMp{order}"] = theory_value
                    row[f"delta_rMp{order}"] = (
                        float(experimental_value) - float(theory_value)
                        if pd.notna(experimental_value) and pd.notna(theory_value)
                        else np.nan
                    )
                matches.append(row)

    if not matches:
        return pd.DataFrame()
    result = pd.DataFrame(matches)
    result["PFAS_absolute_mass_error_ppm"] = result[
        "PFAS_mass_error_ppm"
    ].abs()
    isotope_delta_columns = [
        f"delta_rMp{order}" for order in ISOTOPE_ORDERS
    ]
    result["PFAS_mean_absolute_isotope_delta"] = result[
        isotope_delta_columns
    ].abs().mean(axis=1)
    return result.sort_values(
        [
            "PFAS_intensity_rank",
            "PFAS_absolute_mass_error_ppm",
            "PFAS_mean_absolute_isotope_delta",
        ],
        kind="mergesort",
    ).reset_index(drop=True)


def pair_first_isotopic_peaks(
    data: pd.DataFrame,
    mz_column: str,
    rt_column: str,
    tc_column: str,
    intensity_column: str,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Build 3D_TC-resolved M+1 to M+4 isotope envelopes."""
    result = data.copy()
    result["_PFAS_input_row"] = np.arange(1, len(result) + 1)
    result["_PFAS_mz"] = _numeric(result[mz_column])
    result["_PFAS_rt"] = _numeric(result[rt_column])
    result["_PFAS_tc"] = _numeric(result[tc_column])
    result["_PFAS_intensity"] = _numeric(result[intensity_column])
    result = result.dropna(
        subset=["_PFAS_mz", "_PFAS_rt", "_PFAS_tc", "_PFAS_intensity"]
    ).copy()
    result = result.sort_values("_PFAS_mz", kind="mergesort").reset_index(drop=True)

    mz_values = result["_PFAS_mz"].to_numpy(float)
    rt_values = result["_PFAS_rt"].to_numpy(float)
    tc_values = result["_PFAS_tc"].to_numpy(float)
    intensity_values = result["_PFAS_intensity"].to_numpy(float)
    row_count = len(result)

    if progress:
        progress("Building mutually exclusive M+1 to M+4 isotope envelopes...")

    envelope_base = np.full(row_count, -1, dtype=int)
    isotope_order = np.zeros(row_count, dtype=int)
    mass_offset = np.full(row_count, np.nan)
    rt_error = np.full(row_count, np.nan)
    tc_error = np.full(row_count, np.nan)

    # A feature assigned as M+n cannot seed another envelope in the same
    # 3D_TC-resolved trace. Equal-mass, coeluting peaks at distinct 3D_TC
    # values remain independent M features.
    for candidate_index in range(row_count):
        candidate_mz = mz_values[candidate_index]
        assignments: list[
            tuple[float, float, float, int, int, float, float, float]
        ] = []
        for order in ISOTOPE_ORDERS:
            expected_base_mz = candidate_mz - float(order)
            start = int(
                np.searchsorted(
                    mz_values,
                    expected_base_mz - HIGHER_ISOTOPE_WINDOW_DA,
                    side="left",
                )
            )
            stop = int(
                np.searchsorted(
                    mz_values,
                    expected_base_mz + HIGHER_ISOTOPE_WINDOW_DA,
                    side="right",
                )
            )
            for base_index in range(start, min(stop, candidate_index)):
                if isotope_order[base_index] != 0:
                    continue
                observed_mass_offset = candidate_mz - mz_values[base_index]
                observed_mass_error = observed_mass_offset - float(order)
                observed_rt_error = rt_values[candidate_index] - rt_values[base_index]
                observed_tc_error = tc_values[candidate_index] - tc_values[base_index]
                if (
                    abs(observed_mass_error) <= HIGHER_ISOTOPE_WINDOW_DA
                    and abs(observed_rt_error) <= RT_TOLERANCE_MIN
                    and abs(observed_tc_error) <= TC_TOLERANCE
                ):
                    assignments.append(
                        (
                            abs(observed_rt_error),
                            abs(observed_tc_error),
                            abs(observed_mass_error),
                            base_index,
                            order,
                            observed_mass_offset,
                            observed_rt_error,
                            observed_tc_error,
                        )
                    )
        if not assignments:
            continue
        (
            _,
            _,
            _,
            selected_base,
            selected_order,
            selected_mass_offset,
            selected_rt_error,
            selected_tc_error,
        ) = min(assignments)
        envelope_base[candidate_index] = selected_base
        isotope_order[candidate_index] = selected_order
        mass_offset[candidate_index] = selected_mass_offset
        rt_error[candidate_index] = selected_rt_error
        tc_error[candidate_index] = selected_tc_error

    result["PFAS_isotope_envelope_base_sorted_index"] = envelope_base
    result["PFAS_isotope_order"] = isotope_order
    result["PFAS_isotope_mass_offset_Da"] = mass_offset
    result["PFAS_isotope_rt_error_min"] = rt_error
    result["PFAS_isotope_tc_error"] = tc_error
    result["PFAS_isotope_role"] = np.where(
        isotope_order == 0,
        "M",
        np.char.add("M+", isotope_order.astype(str)),
    )

    denominator = pd.Series(intensity_values).replace(0, np.nan).to_numpy(float)
    for order in ISOTOPE_ORDERS:
        intensity_sum = np.full(row_count, np.nan)
        candidate_count = np.zeros(row_count, dtype=int)
        mz_min = np.full(row_count, np.nan)
        mz_max = np.full(row_count, np.nan)
        offset_min = np.full(row_count, np.nan)
        offset_max = np.full(row_count, np.nan)
        rt_min = np.full(row_count, np.nan)
        rt_max = np.full(row_count, np.nan)
        tc_min = np.full(row_count, np.nan)
        tc_max = np.full(row_count, np.nan)

        assigned = np.where(isotope_order == order)[0]
        for base_index in np.unique(envelope_base[assigned]):
            indices = assigned[envelope_base[assigned] == base_index]
            intensity_sum[base_index] = np.sum(intensity_values[indices])
            candidate_count[base_index] = len(indices)
            mz_min[base_index] = np.min(mz_values[indices])
            mz_max[base_index] = np.max(mz_values[indices])
            offsets = mz_values[indices] - mz_values[base_index]
            rt_errors = rt_values[indices] - rt_values[base_index]
            tc_errors = tc_values[indices] - tc_values[base_index]
            offset_min[base_index] = np.min(offsets)
            offset_max[base_index] = np.max(offsets)
            rt_min[base_index] = np.min(rt_errors)
            rt_max[base_index] = np.max(rt_errors)
            tc_min[base_index] = np.min(tc_errors)
            tc_max[base_index] = np.max(tc_errors)

        prefix = f"PFAS_M_plus_{order}"
        result[f"{prefix}_intensity_sum"] = intensity_sum
        result[f"{prefix}_candidate_count"] = candidate_count
        result[f"{prefix}_mz_min"] = mz_min
        result[f"{prefix}_mz_max"] = mz_max
        result[f"{prefix}_mass_offset_min_Da"] = offset_min
        result[f"{prefix}_mass_offset_max_Da"] = offset_max
        result[f"{prefix}_rt_error_min"] = rt_min
        result[f"{prefix}_rt_error_max"] = rt_max
        result[f"{prefix}_tc_error_min"] = tc_min
        result[f"{prefix}_tc_error_max"] = tc_max
        result[f"{prefix}_ambiguous"] = candidate_count > 1
        result[f"PFAS_rMp{order}"] = 100.0 * intensity_sum / denominator

    carbon_mz = np.full(row_count, np.nan)
    carbon_intensity = np.full(row_count, np.nan)
    carbon_mass_error = np.full(row_count, np.nan)
    carbon_rt_error = np.full(row_count, np.nan)
    carbon_tc_error = np.full(row_count, np.nan)
    carbon_candidate_count = np.zeros(row_count, dtype=int)
    assigned_m1 = np.where(isotope_order == 1)[0]
    for base_index in np.unique(envelope_base[assigned_m1]):
        indices = assigned_m1[envelope_base[assigned_m1] == base_index]
        errors = (
            mz_values[indices]
            - mz_values[base_index]
            - FIRST_ISOTOPE_SPACING_DA
        )
        eligible = indices[np.abs(errors) <= MASS_TOLERANCE_DA]
        carbon_candidate_count[base_index] = len(eligible)
        if len(eligible) == 0:
            continue
        selected = min(
            eligible,
            key=lambda index: (
                abs(
                    mz_values[index]
                    - mz_values[base_index]
                    - FIRST_ISOTOPE_SPACING_DA
                ),
                abs(rt_values[index] - rt_values[base_index]),
                abs(tc_values[index] - tc_values[base_index]),
            ),
        )
        carbon_mz[base_index] = mz_values[selected]
        carbon_intensity[base_index] = intensity_values[selected]
        carbon_mass_error[base_index] = (
            mz_values[selected] - mz_values[base_index] - FIRST_ISOTOPE_SPACING_DA
        )
        carbon_rt_error[base_index] = rt_values[selected] - rt_values[base_index]
        carbon_tc_error[base_index] = tc_values[selected] - tc_values[base_index]

    result["PFAS_first_13C_peak_mz"] = carbon_mz
    result["PFAS_first_13C_peak_intensity"] = carbon_intensity
    result["PFAS_first_13C_peak_mass_error_Da"] = carbon_mass_error
    result["PFAS_first_13C_peak_rt_error_min"] = carbon_rt_error
    result["PFAS_first_13C_peak_tc_error"] = carbon_tc_error
    result["PFAS_first_13C_peak_candidate_count"] = carbon_candidate_count
    result["PFAS_first_13C_peak_ratio"] = 100.0 * carbon_intensity / denominator

    # The deployed model expects the complete nominal M+1 cluster ratio.
    result["first_isotopic_peak_ratio"] = result["PFAS_rMp1"]
    return result


def _calculate_first_isotopic_peak_ratio(
    df: pd.DataFrame,
    config: ProcessingConfig,
    mz_column: str,
    progress: Callable[[str], None] | None,
) -> tuple[pd.DataFrame, str]:
    mode = config.ion_mode.upper()
    method = config.rimp1_method.lower()
    rimp1_column = _selected_or_detected(
        df, config.rimp1_column, "rimp1", mode, required=False
    )
    m_column = _selected_or_detected(
        df, config.m_intensity_column, "m_intensity", mode, required=False
    )
    mp1_column = _selected_or_detected(
        df, config.mp1_intensity_column, "mp1_intensity", mode, required=False
    )

    if method == "auto":
        if rimp1_column and _numeric(df[rimp1_column]).notna().any():
            method = "existing"
        elif m_column and mp1_column:
            method = "abundance"
        else:
            method = "pairing"

    result = df.copy()
    if method == "existing":
        if not rimp1_column:
            raise ProcessingError(
                "Existing first isotopic peak ratio was selected, but no compatible "
                "column was found."
            )
        result["first_isotopic_peak_ratio"] = _numeric(result[rimp1_column])
    elif method == "abundance":
        if not m_column or not mp1_column:
            raise ProcessingError(
                "Calculating the first isotopic peak ratio requires both intensity columns."
            )
        denominator = _numeric(result[m_column]).replace(0, np.nan)
        result["first_isotopic_peak_ratio"] = (
            100.0 * _numeric(result[mp1_column]) / denominator
        )
    elif method == "pairing":
        rt_column = _selected_or_detected(df, config.rt_column, "rt", mode)
        tc_column = _selected_or_detected(df, config.tc_column, "tc", mode)
        intensity_column = _selected_or_detected(
            df, config.intensity_column, "intensity", mode
        )
        result = pair_first_isotopic_peaks(
            result,
            mz_column,
            rt_column,
            tc_column,
            intensity_column,
            progress,
        )
    else:
        raise ProcessingError(
            f"Unsupported first isotopic peak ratio method: {config.rimp1_method}"
        )
    return result, method


def _should_calibrate(method: str, ccs_column: str) -> bool:
    method = method.lower()
    if method == "apply":
        return True
    if method == "none":
        return False
    if method == "auto":
        name = _normalized(ccs_column)
        return "allccs" in name or "pred" in name or "uncal" in name
    raise ProcessingError(f"Unsupported calibration method: {method}")


def fit_mass_calibration(
    standards: pd.DataFrame,
    observed_mz_column: str,
    exact_mz_column: str,
    method: str,
) -> tuple[Callable[[pd.Series], pd.Series], str, float | None, pd.DataFrame, float, float]:
    calibration = standards.copy()
    calibration["PFAS_mass_observed_mz"] = _numeric(
        calibration[observed_mz_column]
    )
    calibration["PFAS_mass_exact_mz"] = _numeric(calibration[exact_mz_column])
    calibration = calibration.dropna(
        subset=["PFAS_mass_observed_mz", "PFAS_mass_exact_mz"]
    ).copy()
    calibration = calibration[
        (calibration["PFAS_mass_observed_mz"] > 0)
        & (calibration["PFAS_mass_exact_mz"] > 0)
    ].copy()
    if calibration.empty:
        raise ProcessingError(
            "Mass calibration requires standards with observed and exact m/z values."
        )
    calibration["PFAS_mass_error_ppm_observed"] = (
        (calibration["PFAS_mass_observed_mz"] - calibration["PFAS_mass_exact_mz"])
        / calibration["PFAS_mass_exact_mz"]
        * 1e6
    )
    method = method.lower()
    if method == "lock":
        ppm_offset = float(calibration["PFAS_mass_error_ppm_observed"].mean())

        def apply(observed: pd.Series) -> pd.Series:
            return _numeric(observed) / (1.0 + ppm_offset / 1e6)

        calibration["PFAS_mass_fitted_error_ppm"] = ppm_offset
        formula = f"lock mass: corrected_mz = observed_mz / (1 + {ppm_offset:.10g}/1e6)"
        r2 = None
    elif method == "linear":
        if len(calibration) < 2:
            raise ProcessingError(
                "Linear mass calibration requires at least two standards."
            )
        if calibration["PFAS_mass_observed_mz"].nunique() < 2:
            raise ProcessingError(
                "Linear mass calibration requires at least two distinct observed m/z values."
            )
        slope, intercept = np.polyfit(
            calibration["PFAS_mass_observed_mz"].to_numpy(float),
            calibration["PFAS_mass_error_ppm_observed"].to_numpy(float),
            1,
        )

        def apply(observed: pd.Series) -> pd.Series:
            observed_numeric = _numeric(observed)
            fitted_ppm = slope * observed_numeric + intercept
            return observed_numeric / (1.0 + fitted_ppm / 1e6)

        calibration["PFAS_mass_fitted_error_ppm"] = (
            slope * calibration["PFAS_mass_observed_mz"] + intercept
        )
        total = (
            calibration["PFAS_mass_error_ppm_observed"]
            - calibration["PFAS_mass_error_ppm_observed"].mean()
        )
        residual_ppm = (
            calibration["PFAS_mass_error_ppm_observed"]
            - calibration["PFAS_mass_fitted_error_ppm"]
        )
        denominator = float(np.square(total).sum())
        r2 = (
            1.0 - float(np.square(residual_ppm).sum()) / denominator
            if denominator
            else 1.0
        )
        formula = (
            f"linear ppm: ppm_error = {slope:.10g} * observed_mz + "
            f"{intercept:.10g}; corrected_mz = observed_mz / (1 + ppm_error/1e6)"
        )
    else:
        raise ProcessingError(f"Unsupported mass calibration method: {method}")

    calibration["PFAS_mass_corrected_mz"] = apply(
        calibration["PFAS_mass_observed_mz"]
    )
    calibration["PFAS_mass_residual_ppm"] = (
        (calibration["PFAS_mass_corrected_mz"] - calibration["PFAS_mass_exact_mz"])
        / calibration["PFAS_mass_exact_mz"]
        * 1e6
    )
    calibration = calibration.sort_values("PFAS_mass_observed_mz").reset_index(
        drop=True
    )
    mz_min = float(calibration["PFAS_mass_observed_mz"].min())
    mz_max = float(calibration["PFAS_mass_observed_mz"].max())
    return apply, formula, r2, calibration, mz_min, mz_max


def apply_mass_calibration(
    source: pd.DataFrame,
    config: ProcessingConfig,
    mode: str,
    mz_column: str,
) -> tuple[pd.DataFrame, str, float | None, pd.DataFrame | None]:
    result = source.copy()
    method = config.mass_calibration_method.lower()
    result["PFAS_mz_raw"] = _numeric(result[mz_column])
    if method == "none":
        result["PFAS_mz_calibrated"] = result["PFAS_mz_raw"]
        result["PFAS_mass_error_ppm_estimated"] = np.nan
        result["PFAS_mass_calibration_extrapolated"] = False
        result["PFAS_mass_calibration_formula"] = "None"
        return result, "None", None, None
    if config.mass_standards_path is None or not config.mass_standards_path.exists():
        raise ProcessingError("Choose a mass calibration standards file.")
    standards = read_input_table(
        config.mass_standards_path, config.mass_standards_sheet_name
    )
    observed_column = _selected_or_detected(
        standards, config.mass_observed_mz_column, "observed_mz", mode
    )
    exact_column = _selected_or_detected(
        standards, config.mass_exact_mz_column, "exact_mz", mode
    )
    apply, formula, r2, calibration_table, mz_min, mz_max = fit_mass_calibration(
        standards, observed_column, exact_column, method
    )
    result["PFAS_mz_calibrated"] = apply(result["PFAS_mz_raw"])
    result["PFAS_mass_error_ppm_estimated"] = (
        (result["PFAS_mz_raw"] - result["PFAS_mz_calibrated"])
        / result["PFAS_mz_calibrated"]
        * 1e6
    )
    result["PFAS_mass_calibration_extrapolated"] = (
        (result["PFAS_mz_raw"] < mz_min) | (result["PFAS_mz_raw"] > mz_max)
    )
    result["PFAS_mass_calibration_formula"] = formula
    return result, formula, r2, calibration_table


def _validated_charge(
    values: pd.Series,
    field_name: str,
) -> pd.Series:
    charge = _numeric(values).abs()
    invalid = charge.notna() & (charge <= 0)
    if invalid.any():
        raise ProcessingError(f"{field_name} must contain non-zero charge values.")
    return charge


def reduced_mass(
    mz: pd.Series | np.ndarray,
    charge: pd.Series | np.ndarray,
) -> np.ndarray:
    mz_values = np.asarray(mz, dtype=float)
    charge_values = np.abs(np.asarray(charge, dtype=float))
    ion_mass = mz_values * charge_values
    return ion_mass * NITROGEN_MASS_DA / (ion_mass + NITROGEN_MASS_DA)


def fit_cims_calibration(
    standards: pd.DataFrame,
    tc_column: str,
    mz_column: str,
    ccs_column: str,
    charge_column: str | None = None,
    default_charge: int = 1,
) -> tuple[float, float, float, pd.DataFrame]:
    calibration = standards.copy()
    calibration["PFAS_standard_tc"] = _numeric(calibration[tc_column])
    calibration["PFAS_standard_mz"] = _numeric(calibration[mz_column])
    calibration["PFAS_standard_CCS"] = _numeric(calibration[ccs_column])
    if charge_column:
        calibration["PFAS_standard_charge"] = _validated_charge(
            calibration[charge_column], "Standard charge"
        )
    else:
        if default_charge == 0:
            raise ProcessingError("Default charge must be non-zero.")
        calibration["PFAS_standard_charge"] = abs(default_charge)
    calibration = calibration.dropna(
        subset=[
            "PFAS_standard_tc",
            "PFAS_standard_mz",
            "PFAS_standard_CCS",
            "PFAS_standard_charge",
        ]
    ).copy()
    calibration = calibration[
        (calibration["PFAS_standard_mz"] > 0)
        & (calibration["PFAS_standard_CCS"] > 0)
        & (calibration["PFAS_standard_charge"] > 0)
    ].copy()
    if len(calibration) < 2:
        raise ProcessingError(
            "At least two standards with numeric tc, m/z, CCS, and charge are required."
        )
    if calibration["PFAS_standard_tc"].nunique() < 2:
        raise ProcessingError("Standard tc values must contain at least two values.")
    calibration["PFAS_reduced_mass_N2"] = reduced_mass(
        calibration["PFAS_standard_mz"],
        calibration["PFAS_standard_charge"],
    )
    calibration["PFAS_standard_reduced_CCS"] = (
        calibration["PFAS_standard_CCS"]
        * np.sqrt(calibration["PFAS_reduced_mass_N2"])
        / calibration["PFAS_standard_charge"]
    )
    slope, intercept = np.polyfit(
        calibration["PFAS_standard_tc"].to_numpy(float),
        calibration["PFAS_standard_reduced_CCS"].to_numpy(float),
        1,
    )
    predicted_reduced = slope * calibration["PFAS_standard_tc"] + intercept
    residual_reduced = (
        calibration["PFAS_standard_reduced_CCS"] - predicted_reduced
    )
    total = (
        calibration["PFAS_standard_reduced_CCS"]
        - calibration["PFAS_standard_reduced_CCS"].mean()
    )
    denominator = float(np.square(total).sum())
    r2 = (
        1.0 - float(np.square(residual_reduced).sum()) / denominator
        if denominator
        else 1.0
    )
    calibration["PFAS_fitted_reduced_CCS"] = predicted_reduced
    calibration["PFAS_back_calculated_CCS"] = (
        predicted_reduced
        * calibration["PFAS_standard_charge"]
        / np.sqrt(calibration["PFAS_reduced_mass_N2"])
    )
    calibration["PFAS_CCS_residual"] = (
        calibration["PFAS_standard_CCS"]
        - calibration["PFAS_back_calculated_CCS"]
    )
    calibration["PFAS_CCS_relative_error_percent"] = (
        100.0
        * calibration["PFAS_CCS_residual"]
        / calibration["PFAS_standard_CCS"].replace(0, np.nan)
    )
    calibration = calibration.sort_values("PFAS_standard_tc").reset_index(drop=True)
    return float(slope), float(intercept), float(r2), calibration


def add_calibration_chart(
    worksheet,
    calibration_table: pd.DataFrame,
    r2: float,
) -> None:
    row_count = len(calibration_table)
    if row_count < 2:
        return
    tc_column = calibration_table.columns.get_loc("PFAS_standard_tc") + 1
    ccs_column = (
        calibration_table.columns.get_loc("PFAS_standard_reduced_CCS") + 1
    )
    fitted_column = (
        calibration_table.columns.get_loc("PFAS_fitted_reduced_CCS") + 1
    )
    x_values = Reference(
        worksheet, min_col=tc_column, min_row=2, max_row=row_count + 1
    )

    chart = ScatterChart()
    chart.title = f"cIMS reduced-CCS calibration (R2 = {r2:.5f})"
    chart.x_axis.title = "Corrected arrival time, tc"
    chart.y_axis.title = "Reduced CCS (CCS * sqrt(mu) / |z|)"
    chart.height = 9
    chart.width = 16

    measured_values = Reference(
        worksheet, min_col=ccs_column, min_row=2, max_row=row_count + 1
    )
    measured = Series(measured_values, x_values, title="Standards")
    measured.marker.symbol = "circle"
    measured.marker.size = 7
    measured.graphicalProperties.line.noFill = True
    chart.series.append(measured)

    fitted_values = Reference(
        worksheet, min_col=fitted_column, min_row=2, max_row=row_count + 1
    )
    fitted = Series(fitted_values, x_values, title="Linear fit")
    fitted.marker.symbol = "none"
    fitted.graphicalProperties.line.solidFill = "D62728"
    fitted.graphicalProperties.line.width = 22000
    chart.series.append(fitted)
    worksheet.add_chart(chart, f"A{row_count + 4}")


def apply_first_calibration(
    source: pd.DataFrame,
    config: ProcessingConfig,
    mode: str,
    mz_column: str,
) -> tuple[pd.DataFrame, str, float | None, pd.DataFrame | None, str]:
    result = source.copy()
    method = config.first_calibration_method.lower()
    if method == "cims_reduced_ccs":
        if config.standards_path is None or not config.standards_path.exists():
            raise ProcessingError(
                "Choose a standards file for the first-stage cIMS calibration."
            )
        sample_tc_column = _selected_or_detected(
            source, config.tc_column, "tc", mode
        )
        standards = read_input_table(
            config.standards_path, config.standards_sheet_name
        )
        standards_tc_column = _selected_or_detected(
            standards, config.standards_tc_column, "tc", mode
        )
        standards_mz_column = _selected_or_detected(
            standards, config.standards_mz_column, "mz", mode
        )
        standards_ccs_column = _selected_or_detected(
            standards, config.standards_ccs_column, "ccs", mode
        )
        standards_charge_column = _selected_or_detected(
            standards,
            config.standards_charge_column,
            "charge",
            mode,
            required=False,
        )
        slope, intercept, r2, calibration_table = fit_cims_calibration(
            standards,
            standards_tc_column,
            standards_mz_column,
            standards_ccs_column,
            standards_charge_column,
            config.default_charge,
        )
        sample_charge_column = _selected_or_detected(
            source, config.charge_column, "charge", mode, required=False
        )
        result["PFAS_tc_input"] = _numeric(result[sample_tc_column])
        if sample_charge_column:
            result["PFAS_charge_used"] = _validated_charge(
                result[sample_charge_column], "Sample charge"
            ).fillna(abs(config.default_charge))
        else:
            if config.default_charge == 0:
                raise ProcessingError("Default charge must be non-zero.")
            result["PFAS_charge_used"] = abs(config.default_charge)
        result["PFAS_reduced_mass_N2"] = reduced_mass(
            _numeric(result[mz_column]),
            result["PFAS_charge_used"],
        )
        result["PFAS_reduced_CCS_fitted"] = (
            slope * result["PFAS_tc_input"] + intercept
        )
        result["PFAS_CCS_stage1"] = (
            result["PFAS_reduced_CCS_fitted"]
            * result["PFAS_charge_used"]
            / np.sqrt(result["PFAS_reduced_mass_N2"])
        )
        tc_min = float(calibration_table["PFAS_standard_tc"].min())
        tc_max = float(calibration_table["PFAS_standard_tc"].max())
        mz_min = float(calibration_table["PFAS_standard_mz"].min())
        mz_max = float(calibration_table["PFAS_standard_mz"].max())
        sample_mz = _numeric(result[mz_column])
        result["PFAS_calibration_tc_extrapolated"] = (
            (result["PFAS_tc_input"] < tc_min)
            | (result["PFAS_tc_input"] > tc_max)
        )
        result["PFAS_calibration_mz_extrapolated"] = (
            (sample_mz < mz_min) | (sample_mz > mz_max)
        )
        result["PFAS_calibration_extrapolated"] = (
            result["PFAS_calibration_tc_extrapolated"]
            | result["PFAS_calibration_mz_extrapolated"]
        )
        formula = (
            f"reduced_CCS = {slope:.10g} * tc + {intercept:.10g}; "
            "CCS = reduced_CCS * |z| / sqrt(mu_N2)"
        )
        result["PFAS_CCS_stage1_formula"] = formula
        return result, formula, r2, calibration_table, "PFAS_CCS_stage1"
    if method == "existing_ccs":
        ccs_column = _selected_or_detected(source, config.ccs_column, "ccs", mode)
        result["PFAS_CCS_stage1"] = _numeric(result[ccs_column])
        formula = f"Existing CCS column: {ccs_column}"
        result["PFAS_CCS_stage1_formula"] = formula
        return result, formula, None, None, ccs_column
    raise ProcessingError(
        f"Unsupported first-stage calibration method: {config.first_calibration_method}"
    )


def make_features(
    raw: pd.DataFrame,
    medians: pd.Series,
    train_mz_sorted: np.ndarray,
) -> pd.DataFrame:
    d = raw[["mz", "CCS", "riMp1"]].copy()
    eps = 1e-12
    for column in ["mz", "CCS", "riMp1"]:
        d[column] = _numeric(d[column])
    denominator = max(len(train_mz_sorted) - 1, 1)
    d["mz_percentile"] = (
        np.searchsorted(train_mz_sorted, d["mz"].to_numpy(float), side="right")
        / denominator
    )
    d["CCS_over_m"] = d["CCS"] / (d["mz"] + eps)
    d["CCS_over_Mp1"] = d["CCS"] / (d["riMp1"] + eps)
    d["M_over_Mp1"] = d["mz"] / (d["riMp1"] + eps)
    d["CCS_over_sqrtm"] = d["CCS"] / (np.sqrt(d["mz"]) + eps)
    d["CCS_over_m23"] = d["CCS"] / ((d["mz"] ** (2 / 3)) + eps)
    return d[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(medians)


def _predict_probability(bundle: dict, raw: pd.DataFrame) -> np.ndarray:
    features = make_features(raw, bundle["medians"], bundle["train_mz_sorted"])
    return bundle["model"].predict_proba(features)[:, 1]


def mc_predict(
    bundle: dict,
    raw: pd.DataFrame,
    iterations: int,
    progress: Callable[[str], None] | None = None,
    chunk_size: int = 8000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if iterations < 1:
        raise ProcessingError("Monte Carlo iterations must be at least 1.")
    rng = np.random.default_rng(42)
    n = len(raw)
    pmean = np.full(n, np.nan)
    pstd = np.full(n, np.nan)
    p05 = np.full(n, np.nan)
    rimp1_noise = float(bundle.get("riMp1_rel_noise", 0.20))
    ccs_noise = float(bundle.get("ccs_rel_noise", 0.03))

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        base = raw.iloc[start:end].reset_index(drop=True)
        probabilities = np.zeros((iterations, len(base)), dtype=np.float32)
        for index in range(iterations):
            perturbed = base.copy()
            perturbed["riMp1"] = np.clip(
                perturbed["riMp1"].to_numpy(float)
                * (1 + rng.uniform(-rimp1_noise, rimp1_noise, len(base))),
                1e-12,
                None,
            )
            perturbed["CCS"] = np.clip(
                perturbed["CCS"].to_numpy(float)
                * (1 + rng.uniform(-ccs_noise, ccs_noise, len(base))),
                1e-12,
                None,
            )
            probabilities[index] = _predict_probability(bundle, perturbed)
            if progress and (index + 1) % max(iterations // 5, 1) == 0:
                progress(
                    f"Predicting rows {start + 1}-{end}: "
                    f"Monte Carlo {index + 1}/{iterations}"
                )
        pmean[start:end] = probabilities.mean(axis=0)
        pstd[start:end] = probabilities.std(axis=0)
        p05[start:end] = np.quantile(probabilities, 0.05, axis=0)
    return pmean, pstd, p05


def assign_tier(
    pmean: np.ndarray,
    p05: np.ndarray,
    t_low: float,
    t_high: float,
) -> tuple[np.ndarray, np.ndarray]:
    level = np.zeros(len(pmean), dtype=int)
    level[(pmean >= t_high) & (pmean < t_low)] = 1
    level[(pmean >= t_low) & (p05 < t_low)] = 2
    level[p05 >= t_low] = 3
    names = np.array(
        [
            "Level 0 Not PFAS",
            "Level 1 Candidate PFAS",
            "Level 2 Medium confidence",
            "Level 3 High-confidence PFAS",
        ],
        dtype=object,
    )
    return level, names[level]


def process_file(
    config: ProcessingConfig,
    progress: Callable[[str], None] | None = None,
) -> ProcessingSummary:
    mode = config.ion_mode.upper()
    notify = progress or (lambda _message: None)
    notify("Reading input table...")
    source = read_input_table(config.input_path, config.sheet_name)
    if source.empty:
        raise ProcessingError("The selected sheet contains no rows.")

    original_mz_column = _selected_or_detected(source, config.mz_column, "mz", mode)
    notify("Applying optional mass calibration...")
    (
        mass_source,
        mass_formula,
        mass_r2,
        mass_calibration_table,
    ) = apply_mass_calibration(source, config, mode, original_mz_column)
    mz_column = "PFAS_mz_calibrated"
    notify("Applying first-stage cIMS CCS calibration...")
    (
        stage1_source,
        first_formula,
        first_r2,
        calibration_table,
        original_ccs_source,
    ) = apply_first_calibration(mass_source, config, mode, mz_column)
    prepared, used_rimp1_method = _calculate_first_isotopic_peak_ratio(
        stage1_source,
        config,
        mz_column,
        notify,
    )

    if "_PFAS_mz" not in prepared:
        prepared["_PFAS_input_row"] = np.arange(1, len(prepared) + 1)
        prepared["_PFAS_mz"] = _numeric(prepared[mz_column])
    if "_PFAS_CCS_input" not in prepared:
        prepared["_PFAS_CCS_input"] = _numeric(prepared["PFAS_CCS_stage1"])

    second_calibration_applied = _should_calibrate(
        config.second_calibration_method, original_ccs_source
    )
    prepared["PFAS_ion_mode"] = mode
    prepared["PFAS_mz_used"] = prepared["_PFAS_mz"]
    prepared["PFAS_CCS_stage1"] = prepared["_PFAS_CCS_input"]
    if second_calibration_applied:
        slope, intercept = CALIBRATION[mode]
        prepared["PFAS_CCS_model_input"] = (
            slope * prepared["PFAS_CCS_stage1"] + intercept
        )
        prepared["PFAS_CCS_stage2_formula"] = (
            f"{mode}: {slope} * input CCS + {intercept}"
        )
    else:
        prepared["PFAS_CCS_model_input"] = prepared["PFAS_CCS_stage1"]
        prepared["PFAS_CCS_stage2_formula"] = (
            "None (stage-1 CCS used directly as model input)"
        )
    prepared["PFAS_first_isotopic_peak_ratio_used"] = _numeric(
        prepared["first_isotopic_peak_ratio"]
    )
    prepared["PFAS_first_isotopic_peak_ratio_method"] = used_rimp1_method
    if "PFAS_rMp1" not in prepared:
        prepared["PFAS_rMp1"] = prepared[
            "PFAS_first_isotopic_peak_ratio_used"
        ]
    for order in (2, 3, 4):
        target = f"PFAS_rMp{order}"
        if target in prepared:
            continue
        source_column = next(
            (
                column
                for column in [f"rMp{order}", f"riMp{order}"]
                if column in prepared
            ),
            None,
        )
        prepared[target] = (
            _numeric(prepared[source_column])
            if source_column
            else np.nan
        )
    if "PFAS_isotope_role" not in prepared:
        prepared["PFAS_isotope_role"] = (
            prepared["isotope_role"].astype(str)
            if "isotope_role" in prepared
            else "M"
        )

    # The trained RF pipeline retains the historical internal feature name riMp1.
    raw = pd.DataFrame(
        {
            "mz": prepared["PFAS_mz_used"],
            "CCS": prepared["PFAS_CCS_model_input"],
            "riMp1": prepared["PFAS_first_isotopic_peak_ratio_used"],
        }
    )
    valid = (
        raw.notna().all(axis=1)
        & (raw["mz"] > 0)
        & (raw["CCS"] > 0)
        & (raw["riMp1"] > 0)
        & (prepared["PFAS_isotope_role"] == "M")
    )
    prepared["PFAS_predictable"] = valid
    prepared["PFAS_processing_note"] = np.where(
        prepared["PFAS_isotope_role"] != "M",
        "Assigned as an isotope peak; only monoisotopic M features are screened",
        np.where(
            valid,
            "",
            "Missing or non-positive m/z, CCS, or first isotopic peak ratio",
        ),
    )
    if "PFAS_calibration_extrapolated" in prepared:
        extrapolated = (
            prepared["PFAS_calibration_extrapolated"].fillna(False).astype(bool)
        )
        prepared.loc[
            valid & extrapolated, "PFAS_processing_note"
        ] = "Prediction used CCS calibration outside the standards' tc or m/z range"
    if "PFAS_mass_calibration_extrapolated" in prepared:
        mass_extrapolated = (
            prepared["PFAS_mass_calibration_extrapolated"].fillna(False).astype(bool)
        )
        current_note = prepared["PFAS_processing_note"].fillna("").astype(str)
        mass_note = "Prediction used mass calibration outside the standards' m/z range"
        prepared.loc[valid & mass_extrapolated, "PFAS_processing_note"] = np.where(
            current_note[valid & mass_extrapolated] == "",
            mass_note,
            current_note[valid & mass_extrapolated] + "; " + mass_note,
        )

    pmean = np.full(len(prepared), np.nan)
    pstd = np.full(len(prepared), np.nan)
    p05 = np.full(len(prepared), np.nan)
    level = np.full(len(prepared), -1, dtype=int)
    level_name = np.full(len(prepared), "Not predictable", dtype=object)

    bundle = load_bundle(mode)
    predictable_count = int(valid.sum())
    if predictable_count:
        notify(f"Running {mode} model for {predictable_count:,} predictable rows...")
        valid_raw = raw.loc[valid].reset_index(drop=True)
        predicted_mean, predicted_std, predicted_p05 = mc_predict(
            bundle, valid_raw, config.mc_iterations, notify
        )
        valid_indices = np.where(valid.to_numpy())[0]
        pmean[valid_indices] = predicted_mean
        pstd[valid_indices] = predicted_std
        p05[valid_indices] = predicted_p05
        predicted_level, predicted_names = assign_tier(
            predicted_mean,
            predicted_p05,
            float(bundle["t_low_fdr10"]),
            float(bundle.get("t_high_fdr18", bundle["t_high_fdr20"])),
        )
        level[valid_indices] = predicted_level
        level_name[valid_indices] = predicted_names

    prepared["PFAS_prob_mean"] = pmean
    prepared["PFAS_prob_std"] = pstd
    prepared["PFAS_prob_p05"] = p05
    prepared["PFAS_MC_uncertainty"] = pmean - p05
    prepared["PFAS_level"] = level
    prepared["PFAS_level_name"] = level_name
    prepared["PFAS_candidate"] = np.where(valid, (level >= 1).astype(int), np.nan)
    prepared["PFAS_high_confidence"] = np.where(
        valid, (level == 3).astype(int), np.nan
    )

    if "_PFAS_intensity" in prepared:
        prepared["PFAS_M_intensity"] = _numeric(prepared["_PFAS_intensity"])
    else:
        intensity_column = _selected_or_detected(
            prepared,
            config.intensity_column,
            "intensity",
            mode,
            required=False,
        )
        prepared["PFAS_M_intensity"] = (
            _numeric(prepared[intensity_column])
            if intensity_column
            else np.nan
        )

    prepared = prepared.drop(
        columns=[
            "_PFAS_input_row",
            "_PFAS_mz",
            "_PFAS_rt",
            "_PFAS_tc",
            "_PFAS_intensity",
            "_PFAS_CCS_input",
        ],
        errors="ignore",
    )

    prioritized = prioritize_pfas_peaks(
        prepared,
        minimum_intensity=float(config.minimum_m_intensity),
        top_n=int(config.priority_top_n),
    )
    prepared["PFAS_priority_selected"] = 0
    prepared["PFAS_intensity_rank"] = np.nan
    if not prioritized.empty:
        prepared.loc[prioritized.index, "PFAS_priority_selected"] = 1
        prepared.loc[prioritized.index, "PFAS_intensity_rank"] = prioritized[
            "PFAS_intensity_rank"
        ]

    database_matches = pd.DataFrame()
    database_path_used = config.database_path or default_database_path()
    if not prioritized.empty:
        if not database_path_used.exists():
            raise ProcessingError(
                f"PFAS database file was not found: {database_path_used}"
            )
        notify(
            f"Matching {len(prioritized):,} prioritized peaks to "
            f"{database_path_used.name}..."
        )
        database_sheet = (
            0
            if database_path_used.suffix.lower() == ".csv"
            else config.database_sheet_name
        )
        database = prepare_pfas_database(
            read_input_table(database_path_used, database_sheet)
        )
        database_matches = match_pfas_database(
            prioritized,
            database,
            mode,
            float(config.database_mass_tolerance_ppm),
        )

    prepared["PFAS_database_match_count"] = 0
    if not database_matches.empty:
        match_counts = database_matches.groupby(
            "PFAS_peak_output_index"
        ).size()
        prepared.loc[
            match_counts.index,
            "PFAS_database_match_count",
        ] = match_counts.astype(int)
        prioritized["PFAS_database_match_count"] = (
            prioritized.index.to_series().map(match_counts).fillna(0).astype(int)
        )

        best = (
            database_matches.sort_values(
                [
                    "PFAS_peak_output_index",
                    "PFAS_absolute_mass_error_ppm",
                    "PFAS_mean_absolute_isotope_delta",
                ],
                kind="mergesort",
            )
            .drop_duplicates("PFAS_peak_output_index")
            .set_index("PFAS_peak_output_index")
        )
        best_columns = {
            "PFAS_ion_hypothesis": "PFAS_best_ion_hypothesis",
            "PFAS_mass_error_ppm": "PFAS_best_mass_error_ppm",
            "preferred_name": "PFAS_best_database_name",
            "molecular_formula": "PFAS_best_molecular_formula",
            "SMILES": "PFAS_best_SMILES",
            "theoretical_rMp1": "PFAS_best_theoretical_rMp1",
            "theoretical_rMp2": "PFAS_best_theoretical_rMp2",
            "theoretical_rMp3": "PFAS_best_theoretical_rMp3",
            "theoretical_rMp4": "PFAS_best_theoretical_rMp4",
        }
        prioritized = prioritized.join(
            best[list(best_columns)].rename(columns=best_columns),
            how="left",
        )
    else:
        prioritized["PFAS_database_match_count"] = 0

    counts = {
        name: int((level_name == name).sum())
        for name in [
            "Level 0 Not PFAS",
            "Level 1 Candidate PFAS",
            "Level 2 Medium confidence",
            "Level 3 High-confidence PFAS",
            "Not predictable",
        ]
    }
    summary_table = pd.DataFrame(
        [
            ("Input file", str(config.input_path)),
            ("Input sheet", str(config.sheet_name)),
            ("Ion mode", mode),
            ("Input rows", len(source)),
            ("Output rows", len(prepared)),
            ("Predictable rows", predictable_count),
            ("Priority minimum M intensity", config.minimum_m_intensity),
            ("Priority Top N", config.priority_top_n),
            ("Prioritized Level >=1 M peaks", len(prioritized)),
            ("PFAS database", str(database_path_used)),
            (
                "PFAS database mass tolerance (+/- ppm)",
                config.database_mass_tolerance_ppm,
            ),
            ("PFAS database match rows", len(database_matches)),
            ("First isotopic peak ratio method", used_rimp1_method),
            ("Exact 13C spacing (Da)", FIRST_ISOTOPE_SPACING_DA),
            ("Exact 13C audit tolerance (+/- Da)", MASS_TOLERANCE_DA),
            (
                "M+1 to M+4 nominal cluster window (+/- Da)",
                HIGHER_ISOTOPE_WINDOW_DA,
            ),
            ("Isotope RT tolerance (+/- min)", RT_TOLERANCE_MIN),
            ("Isotope 3D_TC tolerance (+/-)", TC_TOLERANCE),
            ("Mass calibration", mass_formula),
            ("Mass calibration R2", mass_r2),
            ("First-stage calibration", first_formula),
            ("First-stage calibration R2", first_r2),
            ("Calibration gas", "Nitrogen"),
            ("Nitrogen mass (Da)", NITROGEN_MASS_DA),
            (
                "Calibration transformation",
                "reduced_CCS = CCS * sqrt(mu_N2) / |z|",
            ),
            ("Second-stage calibration applied", second_calibration_applied),
            (
                "Second-stage calibration",
                prepared["PFAS_CCS_stage2_formula"].iloc[0],
            ),
            ("Monte Carlo iterations", config.mc_iterations),
            ("Level 0", counts["Level 0 Not PFAS"]),
            ("Level 1", counts["Level 1 Candidate PFAS"]),
            ("Level 2", counts["Level 2 Medium confidence"]),
            ("Level 3", counts["Level 3 High-confidence PFAS"]),
            ("Not predictable", counts["Not predictable"]),
        ],
        columns=["Item", "Value"],
    )

    notify("Writing labeled Excel output...")
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with pd.ExcelWriter(config.output_path, engine="openpyxl") as writer:
            prepared.to_excel(writer, sheet_name="Labeled data", index=False)
            summary_table.to_excel(writer, sheet_name="Run summary", index=False)
            if mass_calibration_table is not None:
                mass_calibration_table.to_excel(
                    writer, sheet_name="Mass calibration", index=False
                )
            if calibration_table is not None:
                calibration_table.to_excel(
                    writer, sheet_name="Calibration curve", index=False
                )
                add_calibration_chart(
                    writer.book["Calibration curve"],
                    calibration_table,
                    float(first_r2),
                )
            prioritized.to_excel(
                writer,
                sheet_name="Prioritized PFAS peaks",
                index=False,
            )
            database_matches.to_excel(
                writer,
                sheet_name="PFAS database matches",
                index=False,
            )
    except Exception as exc:
        raise ProcessingError(f"Cannot write output workbook: {exc}") from exc
    notify("Complete.")
    return ProcessingSummary(
        input_rows=len(source),
        predictable_rows=predictable_count,
        level_counts=counts,
        rimp1_method=used_rimp1_method,
        first_calibration_formula=first_formula,
        first_calibration_r2=first_r2,
        second_calibration_applied=second_calibration_applied,
        output_path=config.output_path,
    )
