#!/usr/bin/env python3
"""
Automatic QC plots for fMRI harmonization 
Saves plots in outdir.

python scripts/fmri_qc.py \
  --lookup config/scanners_lookup_fmri.csv \
  --pre_dir derivatives/preproc/fmri/connectivity_matrices \
  --post_dir derivatives/harmonized/fmri/fc_covbat \
  --outdir results/validation/fmri_qc

"""

import os, argparse
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import statsmodels.formula.api as smf

sns.set(style="whitegrid")

def vec_upper(mat):
    iu = np.triu_indices(mat.shape[0],1)
    return mat[iu]

def run_qc(lookup, pre_dir, post_dir, outdir, n_sample=200):
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(lookup)
    df_s = df.head(n_sample)

    pre_mats = [np.load(os.path.join(pre_dir, r['fc_file'])) for _, r in df_s.iterrows()]
    post_mats = [np.load(os.path.join(post_dir, r['fc_file'])) for _, r in df_s.iterrows()]
    Xpre = np.stack([vec_upper(m) for m in pre_mats])
    Xpost = np.stack([vec_upper(m) for m in post_mats])
    y = df_s['scanner_id'].values

    # scanner classifier
    clf = RandomForestClassifier(n_estimators=200, random_state=0)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    score_pre = cross_val_score(clf, Xpre, y, cv=cv, scoring='balanced_accuracy').mean()
    score_post = cross_val_score(clf, Xpost, y, cv=cv, scoring='balanced_accuracy').mean()

    # save classifier results
    with open(os.path.join(outdir,"classifier_scores.txt"), "w") as f:
        f.write(f"pre_bal_acc: {score_pre}\npost_bal_acc: {score_post}\n")

    # plot boxplots of global FC by scanner
    df_s['global_pre'] = [m.mean() for m in pre_mats]
    df_s['global_post'] = [m.mean() for m in post_mats]

    plt.figure(figsize=(10,5))
    sns.boxplot(x='scanner_id', y='global_pre', data=df_s)
    plt.title("Global FC (pre)")
    plt.savefig(os.path.join(outdir,"box_global_pre.png"))
    plt.close()

    plt.figure(figsize=(10,5))
    sns.boxplot(x='scanner_id', y='global_post', data=df_s)
    plt.title("Global FC (post)")
    plt.savefig(os.path.join(outdir,"box_global_post.png"))
    plt.close()

    # age slope preservation
    df_s['age'] = df_s['age']
    res_pre = smf.ols("global_pre ~ age + C(scanner_id)", data=df_s).fit()
    res_post = smf.ols("global_post ~ age + C(scanner_id)", data=df_s).fit()
    with open(os.path.join(outdir,"age_slopes.txt"), "w") as f:
        f.write(f"pre_age_coef: {res_pre.params['age']}\npost_age_coef: {res_post.params['age']}\n")

    print("QC outputs saved in", outdir)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lookup", required=True)
    p.add_argument("--pre_dir", required=True)
    p.add_argument("--post_dir", required=True)
    p.add_argument("--outdir", required=True)
    args = p.parse_args()
    run_qc(args.lookup, args.pre_dir, args.post_dir, args.outdir)
