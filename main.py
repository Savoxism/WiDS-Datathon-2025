import pandas as pd

# Load the files
df1 = pd.read_csv("train_processed_ver1.csv")
df2 = pd.read_csv("train_processed_ver2.csv")

# Ensure same order of columns
df1 = df1[df2.columns]

# Check shape first
assert df1.shape == df2.shape, f"Shape mismatch: {df1.shape} vs {df2.shape}"

# Compare rows
diff_mask = df1 != df2
num_diff_rows = diff_mask.any(axis=1).sum()
print(f"\nNumber of rows with differences: {num_diff_rows}")

# Show diffs
diff_rows = df1[diff_mask.any(axis=1)]
diff_details = df1[diff_mask] != df2[diff_mask]
print("\n=== Differences ===")
for idx in diff_rows.index:
    print(f"\nRow {idx}:")
    for col in df1.columns:
        if df1.loc[idx, col] != df2.loc[idx, col]:
            print(f" - Column '{col}': {df1.loc[idx, col]}  -->  {df2.loc[idx, col]}")
