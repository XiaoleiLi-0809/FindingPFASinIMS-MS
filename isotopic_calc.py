# -*- coding: utf-8 -*-
"""
Robust isotopic ratios from an Excel column 'Molecula_formula' using RDKit only.

Outputs per row:
- MonoisotopicMass (from the binned pattern's mono anchor)
- ExactMonoisotopicMass (sum of monoisotopic isotope masses, high precision)
- M, M+1, M+2, ... (percent intensities)
- (M+k)/M ratios

This script uses hardcoded high-precision isotope masses + natural abundances
for common elements, so it works even if your RDKit build lacks abundance tables.

Install once (Conda recommended):
    conda install -c conda-forge rdkit pandas openpyxl numpy
"""

import os, re, math
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem

# ------------- USER SETTINGS (EDIT THESE) -------------
EXCEL_IN       = r"Erin_Baker_new.xlsx"   # <--- EDIT
SHEET_NAME     = 0                                # or "Sheet1"
FORMULA_COLUMN = "PrecursorFormula"               # your column name
EXCEL_OUT      = None  # if None -> input_isotopes.xlsx next to input

# Verbose prints
VERBOSE = True

# How many bins (M..M+5)
MAX_K = 6

# Binning / pruning — relaxed to avoid zeroing small peaks
ABUND_THRESH = 1e-10
MERGE_TOL_DA = 1e-4
BIN_TOL_DA   = 0.50
# ------------------------------------------------------

_pt = Chem.GetPeriodicTable()
_token = re.compile(r"([A-Z][a-z]?|\(|\)|\d+|\.|·)")

def vprint(*args, **kwargs):
    if VERBOSE: print(*args, **kwargs)

# ---------------- High-quality fallback data ----------------
# Natural abundances (fractional, sum≈1.0)
FALLBACK_ABUND = {
    "H":  {1: 0.999885, 2: 0.000115},
    "C":  {12: 0.9893,  13: 0.0107},
    "N":  {14: 0.99636, 15: 0.00364},
    "O":  {16: 0.99757, 17: 0.00038, 18: 0.00205},
    "S":  {32: 0.9499,  33: 0.0075,  34: 0.0425, 36: 0.0001},
    "F":  {19: 1.0},
    "Cl": {35: 0.7578,  37: 0.2422},
    "Br": {79: 0.5069,  81: 0.4931},
    "I":  {127: 1.0},
    "Si": {28: 0.92223, 29: 0.04685, 30: 0.03092},
    "P":  {31: 1.0},
    "Na": {23: 1.0},
    "K":  {39: 0.932581, 41: 0.067302, 40: 0.000117},
    "Ca": {40: 0.96941, 44: 0.02086, 42: 0.00647, 48: 0.00187, 43: 0.00135, 46: 0.00004},
}
# High-precision isotope masses (u)
FALLBACK_MASS = {
    "H":  {1: 1.00782503223, 2: 2.01410177812},
    "C":  {12: 12.0, 13: 13.00335483507},
    "N":  {14: 14.00307400443, 15: 15.00010889888},
    "O":  {16: 15.99491461957, 17: 16.99913175650, 18: 17.99915961286},
    "S":  {32: 31.9720711744, 33: 32.9714589098, 34: 33.967867004, 36: 35.96708071},
    "F":  {19: 18.99840316273},
    "Cl": {35: 34.968852682, 37: 36.965902602},
    "Br": {79: 78.9183376, 81: 80.9162906},
    "I":  {127: 126.9044719},
    "Si": {28: 27.97692653465, 29: 28.97649466490, 30: 29.973770136},
    "P":  {31: 30.97376199842},
    "Na": {23: 22.9897692820},
    "K":  {39: 38.9637064864, 40: 39.963998166, 41: 40.9618252579},
    "Ca": {40: 39.962590863, 42: 41.95861783, 43: 42.95876644,
           44: 43.95548156, 46: 45.9536890, 48: 47.95252276},
}

# For ExactMonoisotopicMass (preferred monoisotopes)
MONO_ISO_NUM = {
    "H": 1,  "C": 12, "N": 14, "O": 16, "F": 19, "P": 31, "S": 32,
    "Cl": 35, "Br": 79, "I": 127, "Si": 28, "Na": 23, "K": 39, "Ca": 40,
}
MONO_ISO_MASS = {
    "H": 1.00782503223, "C": 12.0, "N": 14.00307400443, "O": 15.99491461957,
    "F": 18.99840316273, "P": 30.97376199842, "S": 31.9720711744,
    "Cl": 34.968852682,  "Br": 78.9183376,   "I": 126.9044719,
    "Si": 27.97692653465,"Na": 22.9897692820,"K": 38.9637064864,
    "Ca": 39.962590863,
}

# ---------------- Formula parsing ----------------
def parse_formula(formula: str) -> Dict[str, int]:
    """Parse 'C12H10Cl2O', 'Al2(SO4)3', 'FeSO4·7H2O' -> dict {element: count}."""
    if not isinstance(formula, str) or not formula.strip():
        return {}
    s = formula.replace("·", ".")
    parts = s.split(".")
    total: Dict[str, int] = defaultdict(int)

    def _parse_one(one: str) -> Dict[str, int]:
        tokens = _token.findall(one)
        stack: List[Dict[str, int]] = [defaultdict(int)]
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok == "(":
                stack.append(defaultdict(int)); i += 1
            elif tok == ")":
                mult = 1; i += 1
                if i < len(tokens) and tokens[i].isdigit():
                    mult = int(tokens[i]); i += 1
                group = stack.pop()
                for el, cnt in group.items():
                    stack[-1][el] += cnt * mult
            elif tok.isdigit():
                i += 1
            else:
                el = tok; cnt = 1
                if i+1 < len(tokens) and tokens[i+1].isdigit():
                    cnt = int(tokens[i+1]); i += 1
                stack[-1][el] += cnt; i += 1
        return stack[0]

    for p in parts:
        if not p: 
            continue
        m = re.match(r"^(\d+)([A-Za-z(].*)$", p)  # e.g., "7H2O"
        mult, core = (int(m.group(1)), m.group(2)) if m else (1, p)
        sub = _parse_one(core)
        for el, cnt in sub.items():
            total[el] += cnt * mult
    return dict(total)

# --------- Isotope patterns & convolutions ---------
def element_isotope_pattern(Z: int) -> List[Tuple[float, float]]:
    """
    Single-atom isotope pattern using hardcoded masses+abundances when available.
    If an element is not in the tables, we try RDKit; last resort is average mass.
    Returns [(mass, abundance_fraction)] with sum ~ 1.0.
    """
    sym = _pt.GetElementSymbol(Z)

    # 1) Preferred: hardcoded tables (robust, independent of RDKit build)
    if sym in FALLBACK_ABUND and sym in FALLBACK_MASS:
        masses, abunds = [], []
        for iso, frac in FALLBACK_ABUND[sym].items():
            m = FALLBACK_MASS[sym].get(iso)
            if m is None:
                try:
                    m = _pt.GetIsotopeMass(Z, iso)
                except Exception:
                    m = float(_pt.GetAtomicWeight(Z))
            masses.append(m); abunds.append(frac)
        s = sum(abunds); abunds = [a/s for a in abunds]
        return list(zip(masses, abunds))

    # 2) Otherwise, try RDKit natural abundances
    masses, abunds = [], []
    for iso in range(1, 300):
        try:
            ab = _pt.GetNaturalAbundance(Z, iso)
            if ab and ab > 0.0:
                masses.append(_pt.GetIsotopeMass(Z, iso))
                abunds.append(ab)
        except Exception:
            continue
    if masses:
        s = sum(abunds); abunds = [a/s for a in abunds]
        return list(zip(masses, abunds))

    # 3) Last resort: average atomic weight only (no isotopic structure)
    return [(float(_pt.GetAtomicWeight(Z)), 1.0)]

def _merge_close(peaks: List[Tuple[float,float]], tol=MERGE_TOL_DA) -> List[Tuple[float,float]]:
    if not peaks: return peaks
    peaks.sort(key=lambda x: x[0])
    merged = []
    cm, ca = peaks[0]
    for m, a in peaks[1:]:
        if abs(m - cm) <= tol:
            tot = ca + a
            cm = (cm*ca + m*a) / tot
            ca = tot
        else:
            merged.append((cm, ca))
            cm, ca = m, a
    merged.append((cm, ca))
    return merged

def convolve(patA: List[Tuple[float,float]], patB: List[Tuple[float,float]], thresh=ABUND_THRESH):
    out = defaultdict(float)
    for m1, a1 in patA:
        if a1 < thresh: continue
        for m2, a2 in patB:
            aa = a1*a2
            if aa >= thresh:
                out[m1+m2] += aa
    peaks = list(out.items())
    peaks = _merge_close(peaks, MERGE_TOL_DA)
    total = sum(a for _, a in peaks)
    if total == 0: return []
    peaks = [(m, a/total) for m, a in peaks if a/total >= thresh]
    return peaks

def fast_pow_pattern(pat: List[Tuple[float,float]], n: int) -> List[Tuple[float,float]]:
    if n <= 0: return [(0.0, 1.0)]
    if n == 1: return pat
    if n % 2 == 0:
        half = fast_pow_pattern(pat, n//2)
        return convolve(half, half)
    return convolve(pat, fast_pow_pattern(pat, n-1))

def formula_isotope_pattern(formula: str) -> List[Tuple[float,float]]:
    comp = parse_formula(formula)
    if not comp: return []
    pattern = [(0.0, 1.0)]
    for el, count in comp.items():
        Z = _pt.GetAtomicNumber(el)
        if Z == 0: raise ValueError(f"Unknown element: {el}")
        single = element_isotope_pattern(Z)
        multi  = fast_pow_pattern(single, count)
        pattern = convolve(pattern, multi)
    s = sum(a for _, a in pattern)
    return [(m, a/s) for m, a in pattern if a/s >= ABUND_THRESH]

def isotopic_bins(pattern: List[Tuple[float,float]], max_k=MAX_K, bin_tol=BIN_TOL_DA):
    if not pattern:
        return [np.nan]*max_k, np.nan
    pattern = sorted(pattern, key=lambda x: (x[0], -x[1]))
    mono_mass = pattern[0][0]
    bins = [0.0]*max_k
    for mass, frac in pattern:
        k = int(round(mass - mono_mass))
        if 0 <= k < max_k and abs((mass - mono_mass) - k) <= bin_tol:
            bins[k] += frac
    tot = sum(bins)
    bins = [(b/tot*100.0) if tot>0 else 0.0 for b in bins]
    return bins, mono_mass

def exact_mono_mass(formula: str) -> float:
    comp = parse_formula(formula)
    if not comp:
        return float("nan")
    total = 0.0
    for el, n in comp.items():
        # Prefer our high-precision monoisotope table
        m = MONO_ISO_MASS.get(el)
        if m is None:
            # Try RDKit for the specific monoisotope number
            Z = _pt.GetAtomicNumber(el)
            iso = MONO_ISO_NUM.get(el)
            if iso is not None:
                try:
                    m = _pt.GetIsotopeMass(Z, iso)
                except Exception:
                    m = float(_pt.GetAtomicWeight(Z))
            else:
                m = float(_pt.GetAtomicWeight(Z))
        total += m * n
    return total

def formula_to_M_series(formula: str, max_k=MAX_K):
    pat = formula_isotope_pattern(formula)
    bins, mono = isotopic_bins(pat, max_k=max_k)
    out = {("M" if i==0 else f"M+{i}"): round(b, 6) for i, b in enumerate(bins)}
    out["MonoisotopicMass"] = mono
    out["ExactMonoisotopicMass"] = exact_mono_mass(formula)
    M0 = bins[0] if bins[0] else np.nan
    for i in range(1, max_k):
        out[f"(M+{i})/M"] = round(bins[i]/M0, 8) if (M0 and not math.isnan(M0)) else np.nan
    return out

# ----------------- RUN -----------------
def main():
    # Sanity checks (should NOT be zero beyond M)
    vprint("Sanity C6H6 :", formula_to_M_series("C6H6", max_k=6))      # expect M+1 ~ 7%
    vprint("Sanity C6H4Cl2:", formula_to_M_series("C6H4Cl2", max_k=6))  # expect big M+2

    if not os.path.isfile(EXCEL_IN):
        raise FileNotFoundError(f"Input Excel not found: {EXCEL_IN}")
    df = pd.read_excel(EXCEL_IN, sheet_name=SHEET_NAME)
    if FORMULA_COLUMN not in df.columns:
        raise KeyError(f"Column '{FORMULA_COLUMN}' not found. Columns: {list(df.columns)}")

    results = []
    for idx, f in enumerate(df[FORMULA_COLUMN].astype(str), start=1):
        f2 = f.strip()
        if not f2:
            results.append({})
            continue
        try:
            results.append(formula_to_M_series(f2, max_k=MAX_K))
        except Exception as e:
            vprint(f"[Row {idx}] Error on '{f2}': {e}")
            results.append({})

    out_df = pd.concat([df, pd.DataFrame(results)], axis=1)

    if EXCEL_OUT is None:
        root, ext = os.path.splitext(EXCEL_IN)
        out_path = root + "_isotopes.xlsx"
    else:
        out_path = EXCEL_OUT

    out_df.to_excel(out_path, index=False)
    print(f"✅ Saved isotopic ratios to: {out_path}")

if __name__ == "__main__":
    main()
