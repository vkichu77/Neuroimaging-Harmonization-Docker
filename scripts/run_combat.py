#!/usr/bin/env python3
"""
run_combat.py
Cross-sectional ComBat harmonization using neuroHarmonize (python)
Inputs:
 - input_csv: CSV with columns [subject_id, scanner_id, age, sex, ROI_1, ROI_2, etc]
Outputs:
 - harmonized CSV saved to output_csv

python scripts/run_combat.py \
  --input_csv derivatives/preproc/anat/roi_features.csv \
  --output_csv derivatives/harmonized/structural/roi_features_combat.csv \
  --batch_col scanner_id \
  --covars age sex

"""

import pandas as pd
import argparse
from neuroHarmonize import harmonization
import os

def run_combat(input_csv, output_csv, batch_col="scanner_id", covars=["age","sex"], subj_col=None):
    df = pd.read_csv(input_csv)
    # features = all columns matching ROI pattern (you can change this)
    feature_cols = [c for c in df.columns if c not in [subj_col, batch_col] + covars]
    X = df[feature_cols].values
    batch = df[batch_col].values
    mod = df[covars].values

    print(f"Running ComBat: {X.shape[0]} subjects × {X.shape[1]} features. Batch levels: {len(set(batch))}")
    harmonized_data, model = harmonization.combat(X, batch, mod)
    out_df = df.copy()
    out_df[feature_cols] = harmonized_data
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print("Saved harmonized output to:", output_csv)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", required=True)
    p.add_argument("--output_csv", required=True)
    p.add_argument("--batch_col", default="scanner_id")
    p.add_argument("--covars", nargs="+", default=["age","sex"])
    p.add_argument("--subj_col", default="subject_id")
    args = p.parse_args()
    run_combat(args.input_csv, args.output_csv, args.batch_col, args.covars, args.subj_col)
