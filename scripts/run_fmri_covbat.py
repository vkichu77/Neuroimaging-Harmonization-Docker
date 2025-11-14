#!/usr/bin/env python3
"""
Covariance harmonization (CovBat-style whitening/recoloring) for FC matrices.

Inputs:
 - lookup CSV: columns [subject_id, scanner_id, fc_file, age, sex, etc.]
 - matrices_dir: directory containing fc .npy files (subjects in lookup)
Outputs:
 - harmonized fc .npy files saved to outdir (same filenames)
 - mapping npz files per scanner (a,b, covariances) saved to outdir/mappings/

 python scripts/run_fmri_covbat.py \
  --lookup config/scanners_lookup_fmri.csv \
  --matrices_dir derivatives/preproc/fmri/connectivity_matrices \
  --outdir derivatives/harmonized/fmri/fc_covbat \
  --batch_col scanner_id \
  --covars age sex \
  --do_combat

"""

import os, argparse
import numpy as np
import pandas as pd
from sklearn.covariance import EmpiricalCovariance
from scipy.linalg import sqrtm
from neuroHarmonize import harmonization  # optional if doing ComBat for means
import warnings
warnings.filterwarnings("ignore")

def vec_upper(mat):
    iu = np.triu_indices(mat.shape[0], 1)
    return mat[iu]

def mat_from_vec(vec, n):
    M = np.zeros((n,n), dtype=vec.dtype)
    iu = np.triu_indices(n,1)
    M[iu] = vec
    M = M + M.T
    np.fill_diagonal(M, 1.0)
    return M

def load_all_vectors(df, matrices_dir):
    vecs = []
    for _, r in df.iterrows():
        path = os.path.join(matrices_dir, r['fc_file'])
        mat = np.load(path)
        vecs.append(vec_upper(mat))
    return np.stack(vecs, axis=0)

def pooled_covariance(X):
    # X shape (n_subjects, n_features)
    return np.cov(X, rowvar=False)

def main(args):
    df = pd.read_csv(args.lookup)
    matrices_dir = args.matrices_dir
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir,"mappings"), exist_ok=True)

    # load vectors
    X = load_all_vectors(df, matrices_dir)  # n x p
    n, p = X.shape
    print("Loaded", n, "subjects x", p, "features")

    # optional ComBat on means to remove covariate effects prior to covariance harmonization
    if args.do_combat:
        batch = df[args.batch_col].values
        covars = df[args.covars].values if args.covars is not None else None
        print("Running ComBat on vectorized FC means prior to covariance harmonization ...")
        Xc, model = harmonization.combat(X, batch, covars)

    # compute per-scanner mean & covariance
    scanners = df[args.batch_col].unique()
    mu = {}
    Sigma = {}
    for sc in scanners:
        idx = df[df[args.batch_col]==sc].index.values
        Xk = X[idx]
        mu[sc] = np.mean(Xk, axis=0)
        Sigma[sc] = pooled_covariance(Xk)

    # pooled mean & covariance
    mu_pooled = np.mean(X, axis=0)
    Sigma_pooled = pooled_covariance(X)

    # compute mapping matrices for each scanner: A_k = Sigma_p^{1/2} * Sigma_k^{-1/2}
    # use sqrtm from scipy.linalg; ensure numerical stability
    mapping = {}
    for sc in scanners:
        try:
            Sk = Sigma[sc]
            A_k = sqrtm(Sigma_pooled).dot(np.linalg.inv(sqrtm(Sk)))
            mapping[sc] = {'A': A_k.real.astype(np.float32), 'mu_sc': mu[sc].astype(np.float32)}
            np.savez_compressed(os.path.join(outdir,"mappings", f"mapping_{sc}.npz"),
                                A=A_k.real, mu_sc=mu[sc], mu_pooled=mu_pooled, Sigma_sc=Sigma[sc], Sigma_pooled=Sigma_pooled)
        except Exception as e:
            print("Failed mapping for", sc, e)
            raise

    # apply mapping per subject
    for i, r in df.iterrows():
        sc = r[args.batch_col]
        x = X[i]
        A = mapping[sc]['A']
        mu_sc = mapping[sc]['mu_sc']
        x_centered = x - mu_sc
        x_new = A.dot(x_centered) + mu_pooled
        # reconstruct and save
        nnodes = int((1 + np.sqrt(1 + 8*x_new.size))/2)  # solve p = n(n-1)/2
        mat = mat_from_vec(x_new, nnodes)
        outpath = os.path.join(outdir, r['fc_file'])
        np.save(outpath, mat)
    print("Covariance harmonization done. Outputs in", outdir)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lookup", required=True)
    p.add_argument("--matrices_dir", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--batch_col", default="scanner_id")
    p.add_argument("--covars", nargs="*", default=None)
    p.add_argument("--do_combat", action="store_true", help="run ComBat on means before covariance harmonization")
    args = p.parse_args()
    main(args)
