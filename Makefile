# =============================================================================
# MAKEFILE — Harmonization Workflow Shortcuts
#
# Usage:
# Structural
# make harmonize-struct

# Longitudinal ComBat
# make harmonize-struct-long

# RISH dMRI
# make harmonize-dmri-rish USE_GPU=True N_JOBS=8

# fMRI FC CovBat
# make harmonize-fmri-covbat

# fMRI voxel-level
# make harmonize-fmri-voxel CHUNK_SIZE=50000

# Task-fMRI
# make harmonize-taskfmri TASK_LOOKUP=config/task_contrast2.csv

# QC
# make qc-fmri
#
# Override any variable via:
#   make harmonize-struct INPUT=path/to/myfile.csv
#
# =============================================================================

# -----------------------------
# Configurable variables
# -----------------------------

CONFIG          ?= config/project_config.yaml
LOOKUP          ?= config/scanners_lookup.csv
LOOKUP_FMRI     ?= config/scanners_lookup_fmri.csv

# Structural ROI inputs
STRUCT_IN       ?= derivatives/preproc/anat/roi_features.csv
STRUCT_OUT      ?= derivatives/harmonized/structural/roi_features_combat.csv

# Longitudinal inputs
STRUCT_LONG_IN  ?= derivatives/preproc/anat/roi_longitudinal.csv
STRUCT_LONG_OUT ?= derivatives/harmonized/structural/roi_longcombat.csv

# dMRI RISH
DMRI_OUTDIR     ?= derivatives/harmonized/dwi_rish
SH_ORDER        ?= 6
N_JOBS          ?= 4
USE_GPU         ?= False

# fMRI FC CovBat
FMRI_FC_DIR     ?= derivatives/preproc/fmri/connectivity_matrices
FMRI_FC_OUTDIR  ?= derivatives/harmonized/fmri/fc_covbat

# fMRI voxelwise
FMRI_VOX_DIR    ?= derivatives/preproc/fmri/voxel_maps
FMRI_VOX_MASK   ?= derivatives/preproc/fmri/brain_mask.nii.gz
FMRI_VOX_OUTDIR ?= derivatives/harmonized/fmri/voxel_combat
CHUNK_SIZE      ?= 20000

# Task fMRI
TASK_LOOKUP     ?= config/task_lookup_contrast1.csv
TASK_OUTDIR     ?= derivatives/harmonized/fmri/beta_contrast1_combat
TASK_MASK       ?= derivatives/preproc/fmri/brain_mask.nii.gz

# QC
QC_OUTDIR       ?= results/validation/fmri_qc
QC_PRE_DIR      ?= $(FMRI_FC_DIR)
QC_POST_DIR     ?= $(FMRI_FC_OUTDIR)

# General
BATCH_COL       ?= scanner_id
COVARS          ?= age sex
PYTHON          ?= python
R               ?= Rscript

WRAPPER         := python scripts/harmonize.py


# =============================================================================
# Utility targets
# =============================================================================

.PHONY: help
help:
	@echo "Available targets:"
	@echo "  make harmonize-struct"
	@echo "  make harmonize-struct-long"
	@echo "  make harmonize-dmri-rish"
	@echo "  make harmonize-fmri-covbat"
	@echo "  make harmonize-fmri-voxel"
	@echo "  make harmonize-taskfmri"
	@echo "  make qc-fmri"
	@echo "============================"
	@echo "Override variables like:"
	@echo "  make harmonize-dmri-rish USE_GPU=True N_JOBS=8"


# Cleaning logs
.PHONY: clean-logs
clean-logs:
	rm -rf logs/*


# =============================================================================
# Harmonization targets
# =============================================================================

# -----------------------------
# 1. Structural ComBat
# -----------------------------
.PHONY: harmonize-struct
harmonize-struct:
	$(WRAPPER) \
		--modality struct \
		--input $(STRUCT_IN) \
		--output $(STRUCT_OUT) \
		--batch_col $(BATCH_COL) \
		--covars $(COVARS)


# -----------------------------
# 2. Longitudinal ComBat
# -----------------------------
.PHONY: harmonize-struct-long
harmonize-struct-long:
	$(WRAPPER) \
		--modality struct-long \
		--input $(STRUCT_LONG_IN) \
		--output $(STRUCT_LONG_OUT) \
		--batch_col $(BATCH_COL) \
		--subject_col subject_id \
		--time_col age_at_scan


# -----------------------------
# 3. dMRI RISH Harmonization
# -----------------------------
.PHONY: harmonize-dmri-rish
harmonize-dmri-rish:
	$(WRAPPER) \
		--modality dmri-rish \
		--config $(CONFIG) \
		--lookup $(LOOKUP) \
		--outdir $(DMRI_OUTDIR) \
		--sh_order $(SH_ORDER) \
		--n_jobs $(N_JOBS) $(if $(filter True,$(USE_GPU)),--use_gpu,)


# -----------------------------
# 4. fMRI FC CovBat Harmonization
# -----------------------------
.PHONY: harmonize-fmri-covbat
harmonize-fmri-covbat:
	$(WRAPPER) \
		--modality fmri-covbat \
		--lookup $(LOOKUP_FMRI) \
		--matrices_dir $(FMRI_FC_DIR) \
		--outdir $(FMRI_FC_OUTDIR) \
		--batch_col $(BATCH_COL) \
		--covars $(COVARS) \
		--do_combat


# -----------------------------
# 5. fMRI voxelwise harmonization
# -----------------------------
.PHONY: harmonize-fmri-voxel
harmonize-fmri-voxel:
	$(WRAPPER) \
		--modality fmri-voxel \
		--lookup $(LOOKUP_FMRI) \
		--files_dir $(FMRI_VOX_DIR) \
		--mask $(FMRI_VOX_MASK) \
		--outdir $(FMRI_VOX_OUTDIR) \
		--batch_col $(BATCH_COL) \
		--covars $(COVARS) \
		--chunk_size $(CHUNK_SIZE)


# -----------------------------
# 6. Task fMRI beta-map harmonization
# -----------------------------
.PHONY: harmonize-taskfmri
harmonize-taskfmri:
	$(WRAPPER) \
		--modality task-fmri \
		--lookup $(TASK_LOOKUP) \
		--mask $(TASK_MASK) \
		--outdir $(TASK_OUTDIR) \
		--batch_col $(BATCH_COL) \
		--covars $(COVARS)


# =============================================================================
# QC
# =============================================================================

.PHONY: qc-fmri
qc-fmri:
	$(WRAPPER) \
		--modality qc \
		--qc_mode fmri \
		--lookup $(LOOKUP_FMRI) \
		--pre_dir $(QC_PRE_DIR) \
		--post_dir $(QC_POST_DIR) \
		--outdir $(QC_OUTDIR)


# =============================================================================
# Docker helper targets (optional)
# =============================================================================

.PHONY: build
build:
	docker compose build

.PHONY: shell
shell:
	docker compose run --rm harmonize
