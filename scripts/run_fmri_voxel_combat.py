#!/usr/bin/env python3
"""
Voxelwise ComBat for 4D fMRI maps (e.g., ALFF/NIfTI or resting-state maps).
Processes in chunks to avoid memory overload.

Inputs:
 - lookup CSV: columns [subject_id, scanner_id, file_path, age, sex, ...]
Outputs:
 - harmonized NIfTI files saved to outdir (same filenames)

python scripts/run_fmri_voxel_combat.py \
  --lookup config/scanners_lookup_fmri.csv \
  --files_dir derivatives/preproc/fmri/voxel_maps \
  --mask derivatives/preproc/fmri/brain_mask.nii.gz \
  --outdir derivatives/harmonized/fmri/voxel_combat \
  --batch_col scanner_id \
  --covars age sex \
  --chunk_size 20000

"""

import os, argparse
import nibabel as nib
import numpy as np
import pandas as pd
from neuroHarmonize import harmonization
from joblib import Parallel, delayed
import math

def load_mask(mask_path):
    img = nib.load(mask_path)
    return img.get_fdata().astype(bool), img.affine, img.header

def get_voxel_indices(mask):
    return np.array(np.nonzero(mask)).T

def process_chunk(voxel_idx_chunk, files, mask_idx, df, covars, batch):
    # create data matrix for that chunk: subjects x voxels_in_chunk
    nsub = len(files)
    nk = len(voxel_idx_chunk)
    arr = np.zeros((nsub, nk), dtype=np.float32)
    for i,f in enumerate(files):
        img = nib.load(f)
        data = img.get_fdata()
        # extract voxels in chunk
        arr[i,:] = np.array([data[tuple(idx)] for idx in voxel_idx_chunk])
    # run ComBat across subjects for these voxels (features)
    X_h, model = harmonization.combat(arr, batch, covars)
    return X_h

def main(args):
    df = pd.read_csv(args.lookup)
    files = [os.path.join(args.files_dir, p) if not os.path.isabs(p) else p for p in df['file_path'].values]
    mask, affine, header = load_mask(args.mask)
    voxel_idx = get_voxel_indices(mask)
    nvox = voxel_idx.shape[0]
    print("nvox in mask:", nvox)
    chunk_size = args.chunk_size
    batches = df[args.batch_col].values
    covars = df[args.covars].values if args.covars else None

    # allocate output arrays per subject as memmaps to avoid RAM explosion
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    sample_img = nib.load(files[0])
    shape = sample_img.shape
    # create memmap placeholders
    out_maps = []
    for i, f in enumerate(files):
        fname = os.path.basename(f).replace(".nii.gz","_harm.nii.gz")
        outpath = os.path.join(outdir, fname)
        out_maps.append(outpath)

    # process chunk by chunk
    for start in range(0, nvox, chunk_size):
        end = min(start+chunk_size, nvox)
        chunk_idx = voxel_idx[start:end]
        print(f"Processing voxels {start}..{end} ({end-start})")
        X_h = process_chunk(chunk_idx, files, chunk_idx, df, covars, batches)  # shape (nsub, n_chunk)
        # write back to subject outputs
        for si in range(len(files)):
            # load subject output (or original) and set voxels
            inimg = nib.load(files[si])
            data = inimg.get_fdata()
            for vi, v in enumerate(chunk_idx):
                data[tuple(v)] = X_h[si, vi]
            # save subject-specific temp to outdir (we overwrite per chunk)
            outpath = out_maps[si]
            nib.save(nib.Nifti1Image(data, inimg.affine, inimg.header), outpath)
    print("Done.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lookup", required=True)
    p.add_argument("--files_dir", required=True)
    p.add_argument("--mask", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--batch_col", default="scanner_id")
    p.add_argument("--covars", nargs="+", default=None)
    p.add_argument("--chunk_size", type=int, default=20000)
    args = p.parse_args()
    main(args)
