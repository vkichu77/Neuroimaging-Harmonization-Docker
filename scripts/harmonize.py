#!/usr/bin/env python3
"""
harmonize.py — CLI wrapper to run harmonization pipelines for multiple modalities.

Supported modalities:
  - struct      : cross-sectional ComBat on ROIs (run_combat.py)
  - struct-long : longitudinal ComBat (run_longcombat.R)
  - dmri-rish   : RISH-based dMRI signal harmonization (run_rish_harmonize.py)
  - fmri-covbat : covariance harmonization for FC matrices (run_fmri_covbat.py)
  - fmri-voxel  : voxelwise ComBat on 4D NIfTI (run_fmri_voxel_combat.py)
  - task-fmri   : harmonize task beta maps (run_taskfmri_harmonize.py)
  - qc          : run QC (fmri_qc.py or validation notebook)

Usage examples:
  python scripts/harmonize.py --modality struct --config config/project_config.yaml \
    --input derivatives/preproc/anat/roi_features.csv --output derivatives/harmonized/structural/

  python scripts/harmonize.py --modality dmri-rish --config config/project_config.yaml \
    --lookup config/scanners_lookup.csv --outdir derivatives/harmonized/dwi_rish --sh_order 6 --n_jobs 4

  python scripts/harmonize.py --modality fmri-covbat --lookup config/scanners_lookup_fmri.csv \
    --matrices_dir derivatives/preproc/fmri/connectivity_matrices --outdir derivatives/harmonized/fmri/fc_covbat

Notes:
 - This wrapper assumes the specific scripts exist at scripts/*.py or scripts/*.R (see repo).
 - Use --dry-run to see the commands without executing them.
"""

import argparse
import os
import sys
import yaml
import subprocess
import shlex
import time
from datetime import datetime
from pathlib import Path

# === Utility helpers ===

def now_str():
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def run_cmd(cmd, logfile=None, dry_run=False, shell=False):
    print(f"\n>> RUN: {cmd}")
    if dry_run:
        print("   (dry-run) skipping execution")
        return 0
    if logfile:
        lf = open(logfile, "a")
        lf.write(f"\n\n### {now_str()} ###\n$ {cmd}\n")
    try:
        # If command is string and not shell, split it
        if isinstance(cmd, str) and not shell:
            proc = subprocess.run(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        else:
            proc = subprocess.run(cmd, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out, err = proc.stdout, proc.stderr
        if logfile:
            lf.write("--- STDOUT ---\n")
            lf.write(out)
            lf.write("\n--- STDERR ---\n")
            lf.write(err)
            lf.flush()
            lf.close()
        if proc.returncode != 0:
            print(f"[ERROR] command failed (rc={proc.returncode}). See logfile: {logfile}")
            print(err)
            return proc.returncode
        print(out)
        return 0
    except Exception as e:
        print(f"[EXCEPTION] {e}")
        if logfile:
            with open(logfile, "a") as lf:
                lf.write(f"\nEXCEPTION: {e}\n")
        return 2

def check_path_exists(p, required=True):
    if p is None:
        return False
    if os.path.exists(p):
        return True
    if required:
        raise FileNotFoundError(f"Required path not found: {p}")
    return False

# === Dispatch functions ===

def run_struct(args, logdir):
    # Cross-sectional ComBat (Python)
    input_csv = args.input
    output_csv = args.output
    check_path_exists(input_csv)
    ensure_dir(os.path.dirname(output_csv))
    cmd = f"python scripts/run_combat.py --input_csv {shlex.quote(input_csv)} --output_csv {shlex.quote(output_csv)} --batch_col {args.batch_col} --covars {' '.join(args.covars)}"
    return run_cmd(cmd, logfile=os.path.join(logdir,"struct.log"), dry_run=args.dry_run)

def run_struct_long(args, logdir):
    input_csv = args.input
    output_csv = args.output
    check_path_exists(input_csv)
    ensure_dir(os.path.dirname(output_csv))
    cmd = f"Rscript scripts/run_longcombat.R --input {shlex.quote(input_csv)} --output {shlex.quote(output_csv)} --batch {args.batch_col} --subj {args.subject_col} --time {args.time_col}"
    return run_cmd(cmd, logfile=os.path.join(logdir,"struct_long.log"), dry_run=args.dry_run)

def run_dmri_rish(args, logdir):
    lookup = args.lookup
    check_path_exists(lookup)
    ensure_dir(args.outdir)
    cmd = f"python scripts/run_rish_harmonize.py --config {shlex.quote(args.config)} --lookup {shlex.quote(lookup)} --outdir {shlex.quote(args.outdir)} --sh_order {args.sh_order} --n_jobs {args.n_jobs}"
    if args.use_gpu:
        cmd += " --use_gpu"
    return run_cmd(cmd, logfile=os.path.join(logdir,"dmri_rish.log"), dry_run=args.dry_run)

def run_fmri_covbat(args, logdir):
    lookup = args.lookup
    check_path_exists(lookup)
    ensure_dir(args.outdir)
    cmd = f"python scripts/run_fmri_covbat.py --lookup {shlex.quote(lookup)} --matrices_dir {shlex.quote(args.matrices_dir)} --outdir {shlex.quote(args.outdir)} --batch_col {args.batch_col}"
    if args.covars:
        cmd += " --covars " + " ".join(args.covars)
    if args.do_combat:
        cmd += " --do_combat"
    return run_cmd(cmd, logfile=os.path.join(logdir,"fmri_covbat.log"), dry_run=args.dry_run)

def run_fmri_voxel_combat(args, logdir):
    lookup = args.lookup
    check_path_exists(lookup)
    files_dir = args.files_dir
    check_path_exists(files_dir)
    check_path_exists(args.mask)
    ensure_dir(args.outdir)
    cmd = f"python scripts/run_fmri_voxel_combat.py --lookup {shlex.quote(lookup)} --files_dir {shlex.quote(files_dir)} --mask {shlex.quote(args.mask)} --outdir {shlex.quote(args.outdir)} --batch_col {args.batch_col} --chunk_size {args.chunk_size}"
    if args.covars:
        cmd += " --covars " + " ".join(args.covars)
    return run_cmd(cmd, logfile=os.path.join(logdir,"fmri_voxel.log"), dry_run=args.dry_run)

def run_taskfmri(args, logdir):
    lookup = args.lookup
    check_path_exists(lookup)
    check_path_exists(args.mask)
    ensure_dir(args.outdir)
    cmd = f"python scripts/run_taskfmri_harmonize.py --lookup {shlex.quote(lookup)} --mask {shlex.quote(args.mask)} --outdir {shlex.quote(args.outdir)} --batch_col {args.batch_col}"
    if args.covars:
        cmd += " --covars " + " ".join(args.covars)
    return run_cmd(cmd, logfile=os.path.join(logdir,"taskfmri.log"), dry_run=args.dry_run)

def run_qc(args, logdir):
    # fmri QC or validation notebook (we call py QC script for now)
    if args.qc_mode == "fmri":
        lookup = args.lookup
        check_path_exists(lookup)
        ensure_dir(args.outdir)
        cmd = f"python scripts/fmri_qc.py --lookup {shlex.quote(lookup)} --pre_dir {shlex.quote(args.pre_dir)} --post_dir {shlex.quote(args.post_dir)} --outdir {shlex.quote(args.outdir)}"
        return run_cmd(cmd, logfile=os.path.join(logdir,"qc.log"), dry_run=args.dry_run)
    else:
        print("Unsupported qc_mode:", args.qc_mode)
        return 1

# === Main CLI ===

def parse_args():
    p = argparse.ArgumentParser(description="Harmonization CLI wrapper")

    p.add_argument("--modality", required=True, choices=[
        "struct", "struct-long", "dmri-rish", "fmri-covbat", "fmri-voxel", "task-fmri", "qc"
    ], help="Which harmonization modality to run")

    # generic paths
    p.add_argument("--config", default="config/project_config.yaml", help="Project config yaml")
    p.add_argument("--lookup", default=None, help="Lookup CSV for DWI/fMRI (scanner info)")
    p.add_argument("--input", default=None, help="Input CSV (structural ROI or longitudinal)")
    p.add_argument("--output", default=None, help="Output CSV/file")
    p.add_argument("--outdir", default=None, help="Output directory for harmonized files")
    p.add_argument("--batch_col", default="scanner_id", help="Batch column name")
    p.add_argument("--subject_col", default="subject_id", help="Subject id column name")
    p.add_argument("--time_col", default="age_at_scan", help="Time column name")
    p.add_argument("--covars", nargs="*", default=["age","sex"], help="Covariates for ComBat")
    p.add_argument("--sh_order", type=int, default=6)
    p.add_argument("--n_jobs", type=int, default=4)
    p.add_argument("--use_gpu", action="store_true", help="Enable GPU mode where supported")
    p.add_argument("--matrices_dir", default=None, help="Directory with FC .npy matrices")
    p.add_argument("--files_dir", default=None, help="Directory with voxel maps or NIfTIs")
    p.add_argument("--mask", default=None, help="Brain mask NIfTI path (for voxelwise)")
    p.add_argument("--chunk_size", type=int, default=20000)
    p.add_argument("--do_combat", action="store_true", help="If supported, run ComBat step before covbat")
    p.add_argument("--qc_mode", default="fmri", choices=["fmri"], help="QC mode to run")
    p.add_argument("--pre_dir", default=None, help="pre-harmonization dir for QC")
    p.add_argument("--post_dir", default=None, help="post-harmonization dir for QC")

    # control
    p.add_argument("--dry-run", action="store_true", help="Print commands but do not execute")
    p.add_argument("--skip-validate", action="store_true", help="Skip pre-checks")
    p.add_argument("--logdir", default="logs", help="Directory for logs")
    return p.parse_args()

def main():
    args = parse_args()
    ensure_dir(args.logdir)
    ensure_dir("logs")

    # load config YAML if present (not strictly required)
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    # Dispatch
    rc = 0
    try:
        if args.modality == "struct":
            rc = run_struct(args, args.logdir)
        elif args.modality == "struct-long":
            rc = run_struct_long(args, args.logdir)
        elif args.modality == "dmri-rish":
            rc = run_dmri_rish(args, args.logdir)
        elif args.modality == "fmri-covbat":
            rc = run_fmri_covbat(args, args.logdir)
        elif args.modality == "fmri-voxel":
            rc = run_fmri_voxel_combat(args, args.logdir)
        elif args.modality == "task-fmri":
            rc = run_taskfmri(args, args.logdir)
        elif args.modality == "qc":
            rc = run_qc(args, args.logdir)
        else:
            print("Unknown modality:", args.modality)
            rc = 3
    except FileNotFoundError as e:
        print("[ERROR] missing file:", e)
        rc = 4
    except Exception as e:
        print("[ERROR] unexpected exception:", e)
        rc = 5

    if rc == 0:
        print(f"\n=== Completed modality {args.modality} successfully ===")
    else:
        print(f"\n=== Modality {args.modality} exited with code {rc} ===")
    sys.exit(rc)

if __name__ == "__main__":
    main()

