# PFAS CCS Screening App

This desktop app processes Excel or CSV feature tables for PFAS screening. It can calibrate experimental CCS from cIMS standards, calculate or import the `first_isotopic_peak_ratio`, extract an experimental M+1 to M+4 isotope envelope, run the trained positive- or negative-mode model, and export a labeled Excel workbook.

## What The App Does

For each input feature, the app can:

1. Read `m/z`, corrected arrival time `tc` or existing CCS, and isotope information.
2. Optionally correct measured `m/z` using mass calibration standards.
3. Convert `tc` to CCS using a calibration curve built from standards.
4. Optionally apply the fixed POS/NEG CCS transformation.
5. Calculate or use `rMp1`, with M+1 to M+4 envelope extraction during automatic pairing.
6. Predict PFAS probability with the trained random forest model.
7. Assign a PFAS confidence level:

- `Level 0 Not PFAS`
- `Level 1 Candidate PFAS`
- `Level 2 Medium confidence`
- `Level 3 High-confidence PFAS`
- `Not predictable`

## Start The App

Use the portable Windows build:

```text
dist\PFAS_CCS_Screening\PFAS_CCS_Screening.exe
```

Keep the whole `dist\PFAS_CCS_Screening` folder together. Do not move only the `.exe`, because the models and Python runtime are stored in the same folder.

You can also run the Python entry point during development:

```powershell
python launch_pfas_app.py
```

## Input Files

### Sample Feature Table

The sample table should contain, at minimum:

- `m/z`
- Either corrected arrival time `tc` or an existing calibrated CCS column
- A way to provide `rMp1`

Optional but recommended:

- Charge column, if ions are not singly charged
- Retention time, `3D_TC`, and intensity columns for automatic envelope pairing

If no charge column is selected, the app uses the `Default |charge|` setting. The default is `1`, which matches the `[M+H]+` and `[M-H]-` workflows.

### Standards Table For CCS Calibration

For cIMS calibration, provide a standards table with:

- Standard `tc`
- Standard `m/z`
- Reference CCS in nitrogen
- Charge, optional

Use standards measured under the same ion mode and instrument conditions as the samples. The standards should cover the sample `tc` and `m/z` range as much as possible.

### Mass Calibration Standards

Mass calibration is optional. If used, provide a standards table with:

- Observed or measured `m/z`
- Known exact `m/z`
- Optional compound name or note columns

These standards should be measured in the same run or under the same acquisition conditions as the samples.

## Optional Mass Calibration

Mass calibration is applied before CCS calibration, isotope pairing, and model prediction. The output always keeps both raw and calibrated m/z.

### No Mass Calibration

The app uses the input `m/z` directly.

### Lock Mass PPM Offset

The app estimates a single average ppm offset from all mass standards:

```text
ppm_error = (observed_mz - exact_mz) / exact_mz * 1e6
corrected_mz = observed_mz / (1 + ppm_offset / 1e6)
```

This is useful when the mass error is approximately constant across the run.

### Linear PPM Correction

The app fits mass error as a function of observed m/z:

```text
ppm_error = a * observed_mz + b
corrected_mz = observed_mz / (1 + ppm_error / 1e6)
```

This is the recommended option when multiple standards span the sample m/z range.

Rows outside the mass standards' observed m/z range are still processed, but they are flagged as extrapolated.

## CCS Calibration

### First Calibration: cIMS Standards To Experimental CCS

The app fits a calibration curve from standards using reduced CCS.

For each standard:

```text
m_ion = (m/z) * |z|
mu = m_ion * m_N2 / (m_ion + m_N2)
m_N2 = 28.0134 Da
reduced_CCS = CCS * sqrt(mu) / |z|
```

The fitted calibration is:

```text
reduced_CCS = a * tc + b
```

For each sample:

```text
CCS_stage1 = (a * tc + b) * |z| / sqrt(mu)
```

Rows outside the standards' `tc` or `m/z` range are still predicted when possible, but they are flagged as extrapolated in the output.

### Use Existing CCS Instead

If your sample table already contains calibrated experimental CCS, choose:

```text
First calibration: Use an existing CCS column
```

Then map the CCS column in the advanced column mapping section.

### Second Calibration: Fixed POS/NEG Transformation

The app also includes fixed transformations:

```text
POS: CCS_model = 0.8289 * CCS_stage1 + 38.258
NEG: CCS_model = 0.9585 * CCS_stage1 + 9.3196
```

These transformations were designed for correcting AllCCS2-predicted CCS values. For instrument-calibrated experimental CCS from standards, use:

```text
Second calibration: None (recommended for experimental CCS)
```

Only choose `Apply fixed POS/NEG transformation` when you intentionally want that second transformation.

## M+1 Isotope Ratio Options

The app supports three ways to provide `rMp1`. The historical external names
`first_isotopic_peak_ratio` and `riMp1` remain accepted for compatibility.

### 1. Use an Existing First Isotopic Peak Ratio

Select a column that already contains `first_isotopic_peak_ratio`. Legacy
`riMp1` columns are accepted as input aliases.

### 2. Calculate From Monoisotopic and First-Isotope Intensities

Select the monoisotopic and first isotopic peak intensity columns. The app calculates:

```text
first_isotopic_peak_ratio =
    100 * I(first isotopic peak) / I(monoisotopic peak)
```

### 3. Pair the Experimental Isotope Envelope Automatically

Select `m/z`, retention time, `3D_TC`, and intensity. CCS is not used for
isotope pairing. The app builds mutually exclusive singly charged M+1 to M+4
envelopes using:

```text
nominal mass shifts: +1, +2, +3, and +4 Da
nominal cluster window: +/-0.020 Da
RT difference: +/-0.020 min
3D_TC difference: +/-0.060
```

Each feature can belong to only one isotope envelope, and an assigned M+n peak
cannot seed another envelope. Assignment conflicts are resolved by absolute RT
error, absolute `3D_TC` error, and absolute nominal mass error, in that order.
All assigned fine-structure peaks within each nominal cluster are summed:

```text
rMp1 = 100 * sum(I(M+1 fine-structure candidates)) / I(M)
rMp2 = 100 * sum(I(M+2 fine-structure candidates)) / I(M)
rMp3 = 100 * sum(I(M+3 fine-structure candidates)) / I(M)
rMp4 = 100 * sum(I(M+4 fine-structure candidates)) / I(M)
```

The RF-MC model uses `rMp1`. The app also reports `first_13C_peak_ratio`, which
selects the assigned M+1 peak nearest `+1.00335483507 Da` within `+/-0.010 Da`.
This single-peak value is an audit measurement and is not the model input.

Candidate counts, m/z ranges, mass-offset ranges, RT error ranges, `3D_TC`
error ranges, isotope roles, and ambiguity flags are retained for auditing.
The M+2 to M+4 ratios support preliminary isotope-pattern or elemental-class
assessment; they do not constitute structural identification.

## Intensity Prioritization And PFAS Database Matching

The app can prioritize the most abundant likely PFAS features using both:

- `Minimum M intensity`
- `Top N intense PFAS peaks`

Only monoisotopic `M` features assigned `Level >=1` are eligible. They are
ranked by M intensity and then PFAS probability.

The bundled default database is `Chemical List PFAS.xlsx`. Another Excel or CSV
database with equivalent name, formula, monoisotopic-mass, and SMILES fields can
be selected in the GUI. The default mass tolerance is `+/-5 ppm`.

Two ion interpretations are searched:

```text
POS: observed m/z matches [M+H]+ or M+
NEG: observed m/z matches [M-H]- or M-
proton mass: 1.007276466621 Da
```

For every mass candidate, the app reports the preferred name, DTXSID, CASRN,
molecular formula, neutral monoisotopic mass, SMILES, ion hypothesis, and mass
error. It calculates theoretical nominal `rMp1` to `rMp4` from the molecular
formula and places them beside the experimental ratios.

SMILES represents a possible database structure. A mass or molecular formula
may match multiple structures, so the app retains every candidate within the
selected tolerance and does not declare a unique identification.

## Step-By-Step Use

1. Open `PFAS_CCS_Screening.exe`.
2. Choose the input Excel or CSV feature table.
3. Choose the worksheet.
4. Select the ion mode:
   - `Positive [M+H]+`
   - `Negative [M-H]-`
5. Choose the first isotopic peak ratio source.
6. Choose the optional mass calibration method:
   - `No mass calibration`
   - `Lock mass ppm offset`
   - `Linear ppm correction from standards`
7. Choose the first CCS calibration method:
   - `cIMS calibration from standard tc, m/z and CCS`
   - `Use an existing CCS column`
8. Keep the second calibration as `None` for experimental CCS unless you specifically need the fixed POS/NEG transformation.
9. Confirm or adjust the advanced column mappings.
10. If using mass calibration, choose the mass standards file and map observed and exact `m/z`.
11. If using cIMS calibration, choose the CCS standards file and map standard `tc`, `m/z`, CCS, and optional charge.
12. Choose an output file.
13. Click `Run screening`.

## Output Workbook

The app writes an Excel workbook with these sheets:

### Labeled data

Contains the original input rows plus app-generated columns, including:

- `PFAS_ion_mode`
- `PFAS_mz_raw`
- `PFAS_mz_calibrated`
- `PFAS_mz_used`
- `PFAS_mass_error_ppm_estimated`
- `PFAS_mass_calibration_extrapolated`
- `PFAS_tc_input`
- `PFAS_charge_used`
- `PFAS_CCS_stage1`
- `PFAS_CCS_model_input`
- `PFAS_calibration_extrapolated`
- `PFAS_first_isotopic_peak_ratio_used`
- `PFAS_isotope_role`
- `PFAS_isotope_order`
- `PFAS_isotope_envelope_base_sorted_index`
- `PFAS_first_13C_peak_ratio`
- `PFAS_first_13C_peak_mass_error_Da`
- `PFAS_first_13C_peak_rt_error_min`
- `PFAS_first_13C_peak_tc_error`
- `PFAS_rMp1`
- `PFAS_rMp2`
- `PFAS_rMp3`
- `PFAS_rMp4`
- `PFAS_M_plus_1_*` through `PFAS_M_plus_4_*` audit fields
- `PFAS_prob_mean`
- `PFAS_prob_std`
- `PFAS_prob_p05`
- `PFAS_MC_uncertainty`
- `PFAS_level`
- `PFAS_level_name`
- `PFAS_candidate`
- `PFAS_high_confidence`
- `PFAS_processing_note`

### Prioritized PFAS peaks

Contains the Level >=1 monoisotopic M features that pass the M-intensity
threshold and Top N setting. It includes experimental `rMp1` to `rMp4`, model
scores, intensity rank, database match count, and the best mass-ranked database
candidate for quick review.

### PFAS database matches

Contains all database candidates within the selected ppm tolerance. It includes
the ion hypothesis, mass error, formula, SMILES, experimental and theoretical
`rMp1` to `rMp4`, and their differences.

### Run summary

Records the run settings, calibration formula, R2, model mode, number of predictable rows, and level counts.

### Calibration curve

Created when cIMS calibration standards are used. It includes:

- Standard `tc`
- Standard `m/z`
- Reference CCS
- Reduced CCS
- Fitted reduced CCS
- Back-calculated CCS
- Residuals
- A calibration chart

### Mass calibration

Created when lock-mass or linear mass calibration is used. It includes:

- Observed m/z
- Exact m/z
- Observed ppm error
- Fitted ppm error
- Corrected m/z
- Residual ppm error

## Troubleshooting

If rows are `Not predictable`, check that `m/z`, final CCS, and the first
isotopic peak ratio are present and positive.

If many rows are extrapolated, use standards that better cover the sample `tc` and `m/z` ranges.

If many rows are mass-calibration extrapolated, use mass standards that cover the sample `m/z` range.

If the output probabilities look unexpected, first confirm that POS/NEG mode and second calibration settings match the data source.

If the app does not open, keep the whole portable folder together and launch:

```text
dist\PFAS_CCS_Screening\PFAS_CCS_Screening.exe
```
