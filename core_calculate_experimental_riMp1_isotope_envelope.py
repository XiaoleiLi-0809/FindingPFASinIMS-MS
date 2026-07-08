"""Calculate experimental M+1 to M+4 isotope ratios for singly charged features.

The algorithm builds mutually exclusive isotope envelopes in ascending m/z
order. Candidate peaks must match retention time and 3D_TC, the measured
mobility coordinate used to derive CCS, and each peak can belong to only one
envelope. Candidate conflicts are resolved by retention-time error, 3D_TC
error, and nominal mass-spacing error, in that order. M+1 to M+4 ratios use
the summed intensity of all assigned fine-structure peaks in each nominal
cluster. Direct CCS columns are not used for pairing; 3D_TC separates
coeluting equal-mass features into distinct M envelopes.

The reported ratios are:

    rMp1 = 100 * I(M+1) / I(M)
    rMp2 = 100 * sum(I(M+2 candidates)) / I(M)
    rMp3 = 100 * sum(I(M+3 candidates)) / I(M)
    rMp4 = 100 * sum(I(M+4 candidates)) / I(M)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FIRST_ISOTOPE_SPACING_DA = 1.00335483507
MASS_TOLERANCE_DA = 0.010
HIGHER_ISOTOPE_WINDOW_DA = 0.020
RT_TOLERANCE_MIN = 0.020
TC_TOLERANCE = 0.06
ISOTOPE_ORDERS = (1, 2, 3, 4)
HIGHER_ISOTOPE_ORDERS = (2, 3, 4)


def prepare_input(data: pd.DataFrame) -> pd.DataFrame:
    """Map supported source columns and sort features by m/z."""
    aliases = {
        "mz": ["3D_m_z"],
        "rt": ["3D_RetTime"],
        "intensity": ["3D_Intensity", "M", "3D_Mobility_Mass"],
        "tc": ["3D_TC", "Corrected drift time"],
    }
    rename_map: dict[str, str] = {}
    for target, candidates in aliases.items():
        if target in data.columns:
            continue
        source = next((name for name in candidates if name in data.columns), None)
        if source is not None:
            rename_map[source] = target
    prepared = data.rename(columns=rename_map)
    required = ["mz", "rt", "intensity", "tc"]
    missing = [column for column in required if column not in prepared.columns]
    if missing:
        raise ValueError(f"Missing required input columns: {missing}")

    for column in required:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    prepared = prepared.dropna(subset=required).copy()
    prepared["_original_row"] = prepared.index
    return prepared.sort_values("mz", kind="mergesort").reset_index(drop=True)


def best_first_isotope_candidate(
    mz_sorted: np.ndarray,
    rt_sorted: np.ndarray,
    base_mz: float,
    base_rt: float,
) -> tuple[int | None, int, float, float]:
    """Select the best candidate by mass error and then retention-time error."""
    expected_mz = base_mz + FIRST_ISOTOPE_SPACING_DA
    epsilon = np.finfo(float).eps * max(abs(expected_mz), 1.0) * 4
    start = int(
        np.searchsorted(
            mz_sorted,
            expected_mz - MASS_TOLERANCE_DA - epsilon,
            side="left",
        )
    )
    stop = int(
        np.searchsorted(
            mz_sorted,
            expected_mz + MASS_TOLERANCE_DA + epsilon,
            side="right",
        )
    )

    candidates: list[tuple[float, float, int, float, float]] = []
    for candidate_index in range(start, stop):
        mass_error = (
            mz_sorted[candidate_index] - base_mz - FIRST_ISOTOPE_SPACING_DA
        )
        rt_error = rt_sorted[candidate_index] - base_rt
        if (
            abs(mass_error) <= MASS_TOLERANCE_DA
            and abs(rt_error) <= RT_TOLERANCE_MIN
        ):
            candidates.append(
                (
                    abs(mass_error),
                    abs(rt_error),
                    candidate_index,
                    mass_error,
                    rt_error,
                )
            )

    if not candidates:
        return None, 0, np.nan, np.nan

    _, _, candidate_index, mass_error, rt_error = min(candidates)
    return candidate_index, len(candidates), mass_error, rt_error


def higher_isotope_cluster_candidates(
    mz_sorted: np.ndarray,
    rt_sorted: np.ndarray,
    base_mz: float,
    base_rt: float,
    isotope_order: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find all coeluting candidates in a nominal higher-isotope cluster."""
    expected_mz = base_mz + float(isotope_order)
    epsilon = np.finfo(float).eps * max(abs(expected_mz), 1.0) * 4
    start = int(
        np.searchsorted(
            mz_sorted,
            expected_mz - HIGHER_ISOTOPE_WINDOW_DA - epsilon,
            side="left",
        )
    )
    stop = int(
        np.searchsorted(
            mz_sorted,
            expected_mz + HIGHER_ISOTOPE_WINDOW_DA + epsilon,
            side="right",
        )
    )
    indices = np.arange(start, stop, dtype=int)
    if indices.size == 0:
        return indices, np.array([], dtype=float), np.array([], dtype=float)

    mass_offsets = mz_sorted[indices] - base_mz
    rt_errors = rt_sorted[indices] - base_rt
    eligible = (
        (np.abs(mass_offsets - isotope_order) <= HIGHER_ISOTOPE_WINDOW_DA)
        & (np.abs(rt_errors) <= RT_TOLERANCE_MIN)
    )
    return indices[eligible], mass_offsets[eligible], rt_errors[eligible]


def assign_isotope_envelopes(data: pd.DataFrame) -> pd.DataFrame:
    """Assign each feature to at most one M+1 to M+4 isotope envelope."""
    result = prepare_input(data)
    mz_values = result["mz"].to_numpy(float)
    rt_values = result["rt"].to_numpy(float)
    tc_values = result["tc"].to_numpy(float)
    intensity_values = result["intensity"].to_numpy(float)
    row_count = len(result)

    envelope_base = np.full(row_count, -1, dtype=int)
    isotope_order = np.zeros(row_count, dtype=int)
    mass_offset = np.full(row_count, np.nan)
    rt_error = np.full(row_count, np.nan)
    tc_error = np.full(row_count, np.nan)

    # Process candidate peaks from low to high m/z. A feature assigned as an
    # isotope cannot seed another envelope in the same 3D_TC-resolved trace.
    # Equal-mass, coeluting peaks at distinct 3D_TC values are never assigned to
    # that trace and remain eligible as independent M features.
    for candidate_index in range(row_count):
        candidate_mz = mz_values[candidate_index]
        possible_assignments: list[tuple[float, float, float, int, int, float, float, float]] = []

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
                observed_offset = candidate_mz - mz_values[base_index]
                observed_rt_error = rt_values[candidate_index] - rt_values[base_index]
                observed_tc_error = tc_values[candidate_index] - tc_values[base_index]
                observed_mass_error = observed_offset - float(order)
                if (
                    abs(observed_mass_error) <= HIGHER_ISOTOPE_WINDOW_DA
                    and abs(observed_rt_error) <= RT_TOLERANCE_MIN
                    and abs(observed_tc_error) <= TC_TOLERANCE
                ):
                    possible_assignments.append(
                        (
                            abs(observed_rt_error),
                            abs(observed_tc_error),
                            abs(observed_mass_error),
                            base_index,
                            order,
                            observed_offset,
                            observed_rt_error,
                            observed_tc_error,
                        )
                    )

        if not possible_assignments:
            continue
        (
            _,
            _,
            _,
            selected_base,
            selected_order,
            selected_offset,
            selected_rt_error,
            selected_tc_error,
        ) = min(possible_assignments)
        envelope_base[candidate_index] = selected_base
        isotope_order[candidate_index] = selected_order
        mass_offset[candidate_index] = selected_offset
        rt_error[candidate_index] = selected_rt_error
        tc_error[candidate_index] = selected_tc_error

    result["isotope_envelope_base_sorted_index"] = envelope_base
    result["isotope_order"] = isotope_order
    result["isotope_mass_offset_Da"] = mass_offset
    result["isotope_rt_error_min"] = rt_error
    result["isotope_tc_error"] = tc_error
    result["isotope_role"] = np.where(
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
        rt_error_min = np.full(row_count, np.nan)
        rt_error_max = np.full(row_count, np.nan)
        tc_error_min = np.full(row_count, np.nan)
        tc_error_max = np.full(row_count, np.nan)

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
            rt_error_min[base_index] = np.min(rt_errors)
            rt_error_max[base_index] = np.max(rt_errors)
            tc_error_min[base_index] = np.min(tc_errors)
            tc_error_max[base_index] = np.max(tc_errors)

        prefix = f"M+{order}"
        result[f"{prefix}_intensity_sum"] = intensity_sum
        result[f"{prefix}_candidate_count"] = candidate_count
        result[f"{prefix}_mz_min"] = mz_min
        result[f"{prefix}_mz_max"] = mz_max
        result[f"{prefix}_mass_offset_min_Da"] = offset_min
        result[f"{prefix}_mass_offset_max_Da"] = offset_max
        result[f"{prefix}_rt_error_min"] = rt_error_min
        result[f"{prefix}_rt_error_max"] = rt_error_max
        result[f"{prefix}_tc_error_min"] = tc_error_min
        result[f"{prefix}_tc_error_max"] = tc_error_max
        result[f"{prefix}_ambiguous"] = candidate_count > 1
        result[f"rMp{order}"] = 100.0 * intensity_sum / denominator

    # Retain the exact-carbon peak as an audit quantity. It is selected only
    # from peaks already assigned to the base feature's M+1 cluster.
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
        ranked = sorted(
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
        selected = ranked[0]
        carbon_mz[base_index] = mz_values[selected]
        carbon_intensity[base_index] = intensity_values[selected]
        carbon_mass_error[base_index] = (
            mz_values[selected] - mz_values[base_index] - FIRST_ISOTOPE_SPACING_DA
        )
        carbon_rt_error[base_index] = rt_values[selected] - rt_values[base_index]
        carbon_tc_error[base_index] = tc_values[selected] - tc_values[base_index]

    result["first_13C_peak_mz"] = carbon_mz
    result["first_13C_peak_intensity"] = carbon_intensity
    result["first_13C_peak_mass_error_Da"] = carbon_mass_error
    result["first_13C_peak_rt_error_min"] = carbon_rt_error
    result["first_13C_peak_tc_error"] = carbon_tc_error
    result["first_13C_peak_candidate_count"] = carbon_candidate_count
    result["first_13C_peak_ratio"] = 100.0 * carbon_intensity / denominator
    result["first_isotopic_peak_ratio"] = result["rMp1"]
    return result


def calculate_first_isotopic_peak_ratio(data: pd.DataFrame) -> pd.DataFrame:
    """Build mutually exclusive isotope envelopes and calculate rMp1 to rMp4."""
    return assign_isotope_envelopes(data)


def read_table(path: Path, sheet_name: str | int = 0) -> pd.DataFrame:
    """Read an Excel or CSV feature table."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path, sheet_name=sheet_name)


def write_table(data: pd.DataFrame, path: Path) -> None:
    """Write the paired feature table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        data.to_csv(path, index=False)
    else:
        data.to_excel(path, index=False)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input feature table (.xlsx or .csv).")
    parser.add_argument("output", type=Path, help="Output paired table (.xlsx or .csv).")
    parser.add_argument("--sheet", default=0, help="Excel sheet name or zero-based index.")
    return parser.parse_args()


def main() -> None:
    """Run first-isotopic-peak pairing and ratio calculation."""
    args = parse_args()
    sheet = int(args.sheet) if str(args.sheet).isdigit() else args.sheet
    source = read_table(args.input, sheet)
    paired = calculate_first_isotopic_peak_ratio(source)
    write_table(paired, args.output)
    ratio_count = paired["first_isotopic_peak_ratio"].notna().sum()
    ambiguous_count = paired["M+1_ambiguous"].sum()
    print(f"Input rows: {len(source):,}")
    print(f"Rows with a first isotopic peak ratio: {ratio_count:,}")
    print(f"Rows with multiple eligible candidates: {ambiguous_count:,}")
    for isotope_order in ISOTOPE_ORDERS:
        count = paired[f"rMp{isotope_order}"].notna().sum()
        multiple = paired[f"M+{isotope_order}_ambiguous"].sum()
        print(
            f"Rows with rMp{isotope_order}: {count:,} "
            f"(multiple fine-structure candidates: {multiple:,})"
        )
    print(f"Output: {args.output.resolve()}")


if __name__ == "__main__":
    main()
