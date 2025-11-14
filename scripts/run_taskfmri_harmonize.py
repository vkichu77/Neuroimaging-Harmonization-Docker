#!/usr/bin/env python3
"""
Harmonize task fMRI beta maps for a single contrast across subjects.
Input lookup CSV: subject_id, scanner_id, beta_file, age, sex, etc.


python scripts/run_taskfmri_harmonize.py \
  --lookup config/task_lookup_contrast1.csv \
  --mask derivatives/preproc/fmri/brain_mask.nii.gz \
  --outdir derivatives/harmonized/fmri/beta_contrast1_combat \
  --batch_col scanner_id \
  --covars age sex

"""

import os, argparse
import pandas as pd
import nibabel as nib
import numpy as np
from neuroHarmonize import harmonization

def load_all_beta_vectors(df, mask):
    mats = []
    for _, r in df.iterrows():
        img = nib.load(r['beta_file'])
        data = img.get_fdata()
        mats.append(data[mask])
    return np.vstack(mats)

def main(args):
    df = pd.read_csv(args.lookup)
    mask_img = nib.load(args.mask)
    mask = mask_img.get_fdata().astype(bool)
    files = df['beta_file'].values

    X = load_all_beta_vectors(df, mask)  # nsub x nvox
    batch = df[args.batch_col].values
    covars = df[args.covars].values if args.covars else None

    X_h, model = harmonization.combat(X, batch, covars)

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    for i, r in df.iterrows():
        inimg = nib.load(r['beta_file'])
        data = inimg.get_fdata()
        data[mask] = X_h[i]
        outname = os.path.basename(r['beta_file']).replace(".nii.gz","_harm.nii.gz")
        nib.save(nib.Nifti1Image(data, inimg.affine, inimg.header), os.path.join(outdir, outname))

    print("Harmonized beta maps saved to", outdir)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lookup", required=True)
    p.add_argument("--mask", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--batch_col", default="scanner_id")
    p.add_argument("--covars", nargs="+", default=None)
    args = p.parse_args()
    main(args)
