######  M, M+1, M+2, M+3, M+4, M+5


import pandas as pd
import numpy as np

# === Load Data ===
file_path = "SRM 2585_cal_peaks202506.xlsx"  # Replace with your file path
df = pd.read_excel(file_path)

# === Rename columns for easier reference ===
df.rename(columns={
    '3D_m_z': 'mz',
    '3D_RetTime': 'rt',
    '3D_Intensity': 'intensity'
}, inplace=True)

# === Sort by mz for efficient search ===
df_sorted = df.sort_values(by='mz').reset_index(drop=True)

# === Loop through all rows to identify isotopologues M+1 to M+5 ===
isotope_results = []

for i, base in df_sorted.iterrows():
    base_mz = base['mz']
    base_rt = base['rt']
    base_intensity = base['intensity']
    
    # Store all isotopologue matches
    isotopes = {}
    for n in range(1, 6):  # M+1 to M+5
        candidate = df_sorted[
            (df_sorted['mz'] - base_mz).between(n - 0.02, n + 0.02) &
            (abs(df_sorted['rt'] - base_rt) <= 0.02)
        ]
        isotopes[f'M+{n}'] = candidate.iloc[0] if not candidate.empty else None

    # Build result dictionary
    result = base.to_dict()

    # Loop through each isotopologue and assign values
    fields = {'mz': 'mz', 'rt': 'rt', 'intensity': 'intensity', 'CCS': 'CCS_correct_linear'}
    for label, candidate in isotopes.items():
        for key, col in fields.items():
            result[f'{label}_{key}'] = candidate[col] if candidate is not None else np.nan
        # Calculate intensity ratio
       # result[f'{label}_ratio'] = candidate['intensity'] / base_intensity if candidate is not None else np.nan

    isotope_results.append(result)

# === Save final result to Excel ===
result_df = pd.DataFrame(isotope_results)
result_df.to_excel("data_grouped_SRM2585.xlsx", index=False)
print("Analysis completed. Output saved to data_grouped.xlsx")
######## identify the most intense Molecule peak
import pandas as pd

# === Load your Excel file ===
file_path = 'data_grouped_SRM2585.xlsx'  # Replace with your filename
df = pd.read_excel(file_path)

# === Rename columns for easier reference ===
df.rename(columns={
    'intensity': 'M',
    'M+1_intensity': 'M+1',
    'M+2_intensity': 'M+2',
    'M+3_intensity': 'M+3',
    'M+4_intensity': 'M+4',
    'M+5_intensity': 'M+5',
}, inplace=True)

# === Define intensity columns in order ===
intensity_cols = ['M', 'M+1', 'M+2', 'M+3', 'M+4', 'M+5']

# === Function to process each row ===
def compute_ratios(row):
    intensities = row[intensity_cols]
    
    # Identify most intense peak and its index (e.g., 0 for M, 1 for M+1, etc.)
    max_idx = intensities.idxmax()
    max_val = intensities[max_idx]
    n = intensity_cols.index(max_idx)  # n value corresponding to M+n

    # Safely calculate ratios based on available indices
    mminus1 = row[intensity_cols[n - 1]] / max_val if n >= 1 else None
    mminus2 = row[intensity_cols[n - 2]] / max_val if n >= 2 else None
    mminus3 = row[intensity_cols[n - 3]] / max_val if n >= 3 else None
    mplus1  = row[intensity_cols[n + 1]] / max_val if n + 1 <= 5 else None
    mplus2 = row[intensity_cols[n + 2]] / max_val if n + 2 <= 5 else None
    mplus3  = row[intensity_cols[n + 3]] / max_val if n + 3 <= 5 else None
    mplus4  = row[intensity_cols[n + 4]] / max_val if n + 4 <= 5 else None
    mplus5  = row[intensity_cols[n + 5]] / max_val if n + 5 <= 5 else None
    return pd.Series([
        max_idx, max_val, n, mminus1, mminus2, mminus3, mplus1,mplus2,mplus3,mplus4,mplus5
    ], index=[
        'Most_Intense_Peak', 'Most_Intense_Value', 'Peak_Index',
        'Mminus1_ratio', 'Mminus2_ratio', 'Mminus3_ratio','Mplus1_ratio','Mplus2_ratio','Mplus3_ratio','Mplus4_ratio','Mplus5_ratio',
    ])

# === Apply to each row ===
results = df.apply(compute_ratios, axis=1)

# === Combine with original dataframe ===
df_combined = pd.concat([df, results], axis=1)

# === Rename columns for easier reference ===
df.rename(columns={
    'M': 'M_intensity',
    'M+1': 'M+1_intensity',
    'M+2': 'M+2_intensity',
    'M+3': 'M+3_intensity',
    'M+4': 'M+4_intensity',
    'M+5': 'M+5_intensity',
}, inplace=True)

# === Save to Excel ===
df_combined.to_excel('data_preprocessed_SRM2585.xlsx', index=False)

print("✅ Done. Output saved as 'data_preprocessed_SRM2585.xlsx'.")
