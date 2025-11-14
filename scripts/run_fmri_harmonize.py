#!/usr/bin/env python3
"""
fMRI Harmonization using ComBat / CovBat
Supports:
 - Functional connectivity matrices (FC)
 - ICA spatial maps (MELODIC / dual regression)
 - ALFF / fALFF / ReHo

Usage:
    python run_fmri_harmonize.py \
        --lookup config/scanners_lookup_fmri.csv \
        --matrices derivatives/preproc/fmri/connectivity_matrices \
        --outdir derivatives/harmonized/fmri/fc_combat \
        --batch_col scanner_id \
        --covars age sex
"""

import os, argparse
import numpy as np
import pandas as pd
from neuroHarmonize import harmonization

def load_matrix(path):
    return np.load(path)

def main(args):
    df = pd.read_csv(args.lookup)
    matrices_dir = args.matrices
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    batch = df[args.batch_col].values
    covars = df[args.covars].values

    mats = []
    filelist = []

    print("Loading FC matrices…")
    for _, row in df.iterrows():
        p = os.path.join(matrices_dir, row['fc_file'])
        mats.append(load_matrix(p))
        filelist.append(row['fc_file'])

    # shape: subjects × N × N
    mats = np.stack(mats, axis=0)

    # vectorize upper triangle
    triu_idx = np.triu_indices(mats.shape[1], 1)
    X = mats[:, triu_idx[0], triu_idx[1]]

    print("Running ComBat…")
    X_harmonized, model = harmonization.combat(X, batch, covars)

    # reconstruct matrices
    print("Saving outputs…")
    for i, fname in enumerate(filelist):
        mat = np.zeros_like(mats[0])
        mat[triu_idx] = X_harmonized[i]
        mat = mat + mat.T
        np.fill_diagonal(mat, 1)
        np.save(os.path.join(outdir, fname), mat)

    print("Done.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lookup", required=True)
    p.add_argument("--matrices", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--batch_col", default="scanner_id")
    p.add_argument("--covars", nargs="+", default=["age","sex"])
    args = p.parse_args()
    main(args)
