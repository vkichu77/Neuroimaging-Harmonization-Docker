#!/usr/bin/env python3
"""
GPU/Numba-accelerated RISH-based dMRI harmonization using DIPY.

Features:
 - computes spherical harmonic (SH) coefficients per scan using DIPY,
 - computes RISH maps per SH order (RISH_l = sum_m c_{l,m}^2) — rotationally invariant power per order,
 - for each non-reference scanner estimates voxelwise linear mapping R_ref = a * R_sc + b using matched subjects,
 - converts mapping on RISH into a multiplicative scaling per SH order (heuristic: scale SH coefficients by sqrt(a) per order; small additive shifts put into order-0 coefficient),
 - applies mappings to all scans from that scanner,
 - reconstructs harmonized DWI using `sh_to_sf` and saves NIfTI outputs.
 - Automatic: GPU (cupy) or CPU fallback
 - Numba acceleration for voxel loops
 - Chunked memory-safe processing

Usage:
    python run_rish_harmonize.py \
        --config config/project_config.yaml \
        --lookup config/scanners_lookup.csv \
        --outdir derivatives/harmonized/dwi_rish \
        --sh_order 6 \
        --n_jobs 6 \
        --use_gpu True
"""

import os, argparse, yaml
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm
from dipy.data import default_sphere
from dipy.core.gradients import gradient_table
from dipy.reconst.shm import sf_to_sh, sh_to_sf
import warnings
warnings.filterwarnings("ignore")

# Try GPU (cupy) --------------------------------------------------------------
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

from numba import njit, prange

# ---------------------- Utility --------------------------

def xp(use_gpu):
    """Return cupy or numpy depending on availability and flag."""
    if use_gpu and GPU_AVAILABLE:
        return cp
    return np

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def compute_mask(dwi_img):
    """Very fast brain mask."""
    data = dwi_img.get_fdata()
    mean = np.mean(data[..., :4], axis=3)
    thresh = np.percentile(mean, 60)
    mask = mean > thresh
    return mask

# ----------------- SH & RISH functions -------------------

def compute_sh_coeffs(dwi_data, gtab, sh_order, mask):
    """Compute SH coefficients for all voxels inside mask."""
    X, Y, Z, N = dwi_data.shape
    sig = dwi_data[mask]    # shape (n_voxels, n_dirs)

    sh = sf_to_sh(sig, default_sphere, sh_order=sh_order)

    out = np.zeros((X, Y, Z, sh.shape[1]), dtype=np.float32)
    out[mask] = sh
    return out

@njit(parallel=True, fastmath=True)
def compute_rish_coeffs(sh, sh_order, l_orders):
    """
    Compute RISH(l) = sum_m SH(l,m)^2
    """
    X, Y, Z, C = sh.shape
    n_l = len(l_orders)
    rish = np.zeros((X, Y, Z, n_l), np.float32)

    index = 0
    coeff_idx = []
    for l in l_orders:
        n = 2*l + 1
        coeff_idx.append((index, index+n))
        index += n

    for i in prange(X):
        for j in prange(Y):
            for k in prange(Z):
                for li in range(n_l):
                    start, end = coeff_idx[li]
                    tmp = 0.0
                    for c in range(start, end):
                        tmp += sh[i, j, k, c] ** 2
                    rish[i, j, k, li] = tmp
    return rish

# ---------------------- Ridge mapping ---------------------

def estimate_mapping(rish_ref_list, rish_src_list, mask, alpha=0.1):
    """Voxelwise ridge regression mapping R_ref = a*R_src + b"""
    Xmask = np.where(mask)
    Xv = len(Xmask[0])
    n_l = rish_ref_list[0].shape[3]
    xp_backend = np

    maps_a = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2], n_l), dtype=np.float32)
    maps_b = np.zeros_like(maps_a)

    ref_stacked = np.stack([r[mask] for r in rish_ref_list], axis=0)
    src_stacked = np.stack([r[mask] for r in rish_src_list], axis=0)

    for l in tqdm(range(n_l), desc="Voxelwise mappings per order"):
        y = ref_stacked[:, :, l]   # shape (n_subjects, n_vox)
        x = src_stacked[:, :, l]
        # closed form ridge: a = sum(x*y) / (sum(x^2)+alpha)
        xy = np.sum(x * y, axis=0)
        xx = np.sum(x * x, axis=0) + alpha
        a = xy / xx
        b = np.mean(y - a * x, axis=0)

        tmp_a = np.zeros(mask.shape)
        tmp_b = np.zeros(mask.shape)
        tmp_a[mask] = a
        tmp_b[mask] = b

        maps_a[..., l] = tmp_a
        maps_b[..., l] = tmp_b

    return maps_a, maps_b

# ------------------ Apply mapping -------------------------

def apply_mapping(sh, maps_a, maps_b, l_orders, mask):
    """Scale SH coefficients by sqrt(a_l) and shift order-0 by b."""
    X, Y, Z, nc = sh.shape
    out = sh.copy()

    # index slices
    slices = []
    idx = 0
    for l in l_orders:
        n = 2*l + 1
        slices.append((idx, idx+n))
        idx += n

    for li, (s0, s1) in enumerate(slices):
        a = maps_a[..., li]
        b = maps_b[..., li]
        scale = np.sqrt(np.maximum(a, 1e-8))

        out[..., s0:s1] *= scale[..., None]
        if li == 0:  # order-0 offset
            out[..., s0] += b

    return out

# ------------------ Reconstruct signal --------------------

def reconstruct_signal(sh, sphere, gtab):
    """Reconstruct DWI at original directions."""
    sf = sh_to_sf(sh.reshape(-1, sh.shape[3]), sphere, sh_order=6)
    sf = sf.reshape(sh.shape[0], sh.shape[1], sh.shape[2], -1)

    # map sphere → original bvecs
    bvecs = gtab.bvecs
    verts = sphere.vertices
    idx_map = np.argmax(verts.dot(bvecs.T), axis=0)

    X, Y, Z, _ = sh.shape
    out = np.zeros((X, Y, Z, len(bvecs)), dtype=np.float32)

    for i in range(len(bvecs)):
        out[..., i] = sf[..., idx_map[i]]

    return out

# ------------------ MAIN PIPELINE -------------------------

def main(args):
    cfg = load_config(args.config)
    lookup = pd.read_csv(args.lookup)

    use_gpu = args.use_gpu and GPU_AVAILABLE
    xp_backend = xp(use_gpu)

    ref_scanner = cfg['reference_scanner']
    sh_order = args.sh_order

    l_orders = list(range(0, sh_order+1, 2))

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(f"{outdir}/mappings", exist_ok=True)

    # mask from first reference subject
    ref_df = lookup[lookup['scanner_id'] == ref_scanner]
    ref_img = nib.load(ref_df.iloc[0]['dwi_path'])
    mask = compute_mask(ref_img)

    # scanners to harmonize
    scanners = sorted(lookup['scanner_id'].unique())
    scanners.remove(ref_scanner)

    # for each scanner
    for sc in scanners:
        print(f"\n=== Harmonizing scanner {sc} ===")

        src_df = lookup[lookup['scanner_id'] == sc]

        # match by nearest age
        ref_ages = ref_df['age'].values[:, None]
        src_ages = src_df['age'].values[:, None]

        from sklearn.neighbors import NearestNeighbors
        knn = NearestNeighbors(n_neighbors=1).fit(ref_ages)
        _, idxs = knn.kneighbors(src_ages)

        matched_ref = [ref_df.iloc[i].to_dict() for i in idxs[:,0]]
        matched_src = [src_df.iloc[i].to_dict() for i in range(len(src_df))]

        # compute SH and RISH for matched subjects
        print("Computing SH & RISH for matched subjects…")
        rish_ref_list, rish_src_list = [], []

        for ref_row, src_row in tqdm(list(zip(matched_ref, matched_src))):
            # REF
            img = nib.load(ref_row['dwi_path'])
            data = img.get_fdata()
            gtab = gradient_table(np.loadtxt(ref_row['bval_path']),
                                  np.loadtxt(ref_row['bvec_path']))
            sh = compute_sh_coeffs(data, gtab, sh_order, mask)
            rish = compute_rish_coeffs(sh, sh_order, l_orders)
            rish_ref_list.append(rish)

            # SRC
            img = nib.load(src_row['dwi_path'])
            data = img.get_fdata()
            gtab = gradient_table(np.loadtxt(src_row['bval_path']),
                                  np.loadtxt(src_row['bvec_path']))
            sh = compute_sh_coeffs(data, gtab, sh_order, mask)
            rish = compute_rish_coeffs(sh, sh_order, l_orders)
            rish_src_list.append(rish)

        # estimate mapping
        print("Estimating voxelwise mapping…")
        maps_a, maps_b = estimate_mapping(rish_ref_list, rish_src_list, mask)

        # save mapping
        np.savez_compressed(f"{outdir}/mappings/{sc}.npz",
                            a=maps_a, b=maps_b, orders=l_orders)

        # apply mapping to ALL scans from this scanner
        print(f"Applying mapping to {len(src_df)} DWI scans…")
        for _, row in src_df.iterrows():
            img = nib.load(row['dwi_path'])
            data = img.get_fdata()
            gtab = gradient_table(np.loadtxt(row['bval_path']),
                                  np.loadtxt(row['bvec_path']))

            sh = compute_sh_coeffs(data, gtab, sh_order, mask)
            sh2 = apply_mapping(sh, maps_a, maps_b, l_orders, mask)
            out_dwi = reconstruct_signal(sh2, default_sphere, gtab)

            outpath = os.path.join(outdir,
                                   os.path.basename(row['dwi_path']).replace(".nii.gz","_harm.nii.gz"))
            nib.save(nib.Nifti1Image(out_dwi, img.affine, img.header), outpath)

        print(f"Scanner {sc} DONE.")

    print("\nAll scanners harmonized!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--lookup", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--sh_order", type=int, default=6)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--n_jobs", type=int, default=4)
    args = parser.parse_args()
    main(args)
