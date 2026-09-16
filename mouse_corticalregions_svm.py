import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
import matplotlib
matplotlib.use("Agg")  # non-interactive backend: never opens a window, so plt.show()/figures can't block script execution waiting to be closed
import matplotlib.pyplot as plt
import os
import re
import gc
import pickle
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc
)
import seaborn as sns
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, leaves_list
from scipy.spatial.distance import squareform
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from matplotlib.patches import Patch

# ══════════════════════════════════════════════════════════════════════════════
# 0. PATHS AND DEVICE
# ══════════════════════════════════════════════════════════════════════════════
DATA_DIR = r"U:\Research\microarray"
DATA     = r"C:\Users\StujenskeLab\Documents\NAN151_workspace\microarray"

fpath1    = os.path.join(DATA_DIR, 'merfish_F1_log.csv')
fpath2    = os.path.join(DATA_DIR, 'merfish_F2_log.csv')
fpath3    = os.path.join(DATA_DIR, 'merfish_M1_log.csv')
fpath4    = os.path.join(DATA_DIR,     'MERFISH-C57BL6J-638850_excitatory.csv')
genes_csv = os.path.join(DATA,     'common_genes_4datasets.csv')

print("=" * 60)
print("MOUSE CORTICAL-STRUCTURE REGION CLASSIFICATION PIPELINE (6 celltypes)")
print("=" * 60)
print("18 base M_regions, defined PURELY by parcellation_structure membership")
print("(no hippocampus/striatum/hypothalamus/amygdala regions this time,")
print("and no fallback through parcellation_division at all -- includes TT")
print("and DP as plain, unsplit olfactory-adjacent regions). Each region gets")
print("AT MOST ONE split, no combinations: PL, ACAd, ACAv are split into")
print("rostral/caudal halves by per-animal x_CCF midpoint (Step 5). ILA is")
print("split into dorsal/ventral halves by per-animal y_CCF midpoint instead")
print("(Step 5B) -- NOT rostral/caudal, and not both. F1/F2/M1 share one")
print("common pooled midpoint per split; M2 (different coordinate scale)")
print("gets its own separate midpoint, for both split axes.")
print("22 distinct M_regions total:")
print("  M_FRP, M_ACAdr, M_ACAdc, M_ACAvr, M_ACAvc, M_PLr, M_PLc,")
print("  M_ILAd, M_ILAv, M_ORBl, M_ORBm, M_ORBvl,")
print("  M_MOs, M_MOp, M_RSPd, M_RSPv, M_RSPagl, M_AId, M_AIv, M_AIp,")
print("  M_TT, M_DP")
print()
print("Workflow: load ALL cells (whole brain) -> assign base region from")
print("parcellation_structure -> split PL/ACAd/ACAv into rostral/caudal,")
print("ILA into dorsal/ventral (log1p SKIPPED, input already log-transformed)")
print("-> identify ROI cells -> TRAIN/VAL/TEST SPLIT ON ROI CELLS FIRST ->")
print("fit ComBat on (all non-ROI whole-brain cells + ROI-TRAIN cells only;")
print("covariate = class) -> apply learned per-animal correction to the")
print("ENTIRE whole brain -> filter down to ROI cells -> RESTRICT to a fixed")
print("set of 6 celltypes -> train ONE SVM region classifier PER celltype,")
print("reusing the same train/val/test split -> for each celltype, cluster")
print("its regions by confusion pattern (permutation-importance gene analysis")
print("removed from this version — clustering only).")
print()
print("DIFFERENCE FROM THE AUTO-DISCOVERY VERSION: instead of scanning for")
print("every celltype with >=4,000 pooled ROI cells, this restricts training")
print("to a FIXED list of 6 celltypes you specified.")

SAVE_DIR = r"U:\Research\microarray\crossspecies\mouse_regions_6celltypes"
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Save directory : {SAVE_DIR}")

animal_paths = [
    (fpath1, "F1", "female"),
    (fpath2, "F2", "female"),
    (fpath3, "M1", "male"),
    (fpath4, "M2", "male"),
]

rename_map = {
    "CCF_level1":            "parcellation_division",
    "CCF_level2":            "parcellation_structure",
    "acronym":               "parcellation_substructure",
    "hrc_mmc_subclass_name": "subclass",
}

meta_cols_needed = [
    "parcellation_division", "parcellation_structure",
    "parcellation_substructure", "subclass",
    "z", "cell_id",
    "brain_section_label", "cluster", "neurotransmitter",
    "CCF_level1", "CCF_level2", "acronym", "hrc_mmc_subclass_name",
]
# The x-CCF (rostral-caudal) and y-CCF (dorsal-ventral) coordinate columns
# are named inconsistently across files -- 'x_CCF'/'y_CCF' in some,
# 'x_ccf'/'y_ccf' in others (and possibly plain 'x'/'y' in older files) --
# and the class-label column has the SAME kind of inconsistency: named
# 'hrc_mmc_class_name' in some files, but already just 'class' in others
# (e.g. M2). All three are detected per-file below (find_coord_col /
# find_class_col) rather than hardcoded to one fixed name, and renamed to
# consistent 'x'/'y'/'class' column names after loading.

print("Classifier backend: scikit-learn SVC (CPU), hyperparameter-tuned per celltype.")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
N_PER_ANIMAL_COMBAT_TRAIN = 150000   # TOTAL fit-pool budget per animal (ROI-train + non-ROI combined)
ROI_TRAIN_FIT_FRACTION    = 0.5      # target share of that budget drawn from ROI-train cells
MAX_COMBAT_FIT_POOL_TOTAL = 1000000  # hard cap on TOTAL fit-pool size across all animals combined
MIN_TRAIN_PER_REGION_CT   = 50       # min train cells a region needs, WITHIN a celltype, to be kept
# A region can pass MIN_TRAIN_PER_REGION_CT (plenty of training cells) but
# still end up with very few TEST cells for that celltype (e.g. DP/TT with
# 30-110 test cells in several celltypes) -- precision/recall/F1 computed
# from that few examples is not a reliable estimate. Rather than report an
# unreliable number, any region with fewer than this many TEST cells is
# excluded from that celltype's classification report, confusion matrix,
# and confusion-based clustering (treated as missing data for REPORTING
# purposes only -- it is still used during TRAINING, since more training
# data is never a problem the way a tiny test sample is).
MIN_TEST_CELLS_FOR_REPORTING = 30

# Fixed list of 6 celltypes to train (replaces auto-discovery via MIN_CELLS_PER_CELLTYPE)
CELLTYPES_OF_INTEREST = [
    "004 L6 IT CTX Glut",
    "006 L4/5 IT CTX Glut",
    "007 L2/3 IT CTX Glut",
    "022 L5 ET CTX Glut",
    "030 L6 CT CTX Glut",
    "032 L5 NP CTX Glut",
]

# --- SVM + hyperparameter tuning config -------------------------------------
# Kernel SVMs scale poorly (roughly O(n^2)-O(n^3)), so for celltypes whose
# training set exceeds MAX_SVM_TRAIN_CELLS we fit (and tune) on a stratified
# subsample of that size rather than the full training set. This is called
# out explicitly in the per-celltype log output, not applied silently.
MAX_SVM_TRAIN_CELLS = 15000
SVM_CV_FOLDS         = 3
SVM_TUNE_SCORING     = "balanced_accuracy"   # robust to M_region class imbalance
# GridSearchCV(n_jobs=-1) uses joblib's loky backend, which memory-maps the
# training array to a scratch file per worker process (typically under
# %TEMP% on Windows) on every parallel dispatch. Across dozens of celltypes
# x 16 hyperparameter combos x SVM_CV_FOLDS folds, those scratch files can
# accumulate faster than they're cleaned up and exhaust the temp drive
# ("No space left on device"), especially on a small/full system drive.
# Since each celltype's fitting set is already capped at MAX_SVM_TRAIN_CELLS
# (small by SVM standards), we default to serial fitting (no memmapping) to
# avoid this. Raise SVM_N_JOBS if you've confirmed %TEMP% has ample free
# space and want the speedup.
SVM_N_JOBS           = 1
SVM_PARAM_GRID = [
    {"svc__kernel": ["rbf"],    "svc__C": [0.1, 1, 10, 100], "svc__gamma": ["scale", 0.01, 0.1]},
    {"svc__kernel": ["linear"], "svc__C": [0.1, 1, 10, 100]},
]

# --- Confusion-based region clustering config --------------------------------
# After each celltype's confusion matrix is built, regions are clustered by
# confusion pattern; the dendrogram is then cut at an ABSOLUTE distance
# threshold (not a fraction of that celltype's own max linkage distance —
# a fraction-based cut is fragile: one fully-separable outlier region can
# inflate the max distance enough that genuinely tight, confusable pairs
# elsewhere no longer clear the cutoff). Since distance = 1 - avg symmetric
# confusion rate, CONFUSION_DISTANCE_THRESHOLD=0.85 means "cluster together
# any two regions with more than 15% average mutual misclassification."
CONFUSION_DISTANCE_THRESHOLD = 0.85

# ══════════════════════════════════════════════════════════════════════════════
# 1. DETERMINE GENE PANEL
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 1: Mouse gene panel")
print("=" * 60)

mouse_panel = pd.read_csv(genes_csv).iloc[:, 0].tolist()
print(f"Mouse panel genes: {len(mouse_panel)}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. DETERMINE COLUMNS TO LOAD PER FILE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: Determining columns to load per file")
print("=" * 60)

def find_coord_col(all_cols, axis):
    """Find a spatial coordinate column (x or y) regardless of its exact
    casing/naming across files ('x_CCF'/'y_CCF', 'x_ccf'/'y_ccf', or plain
    'x'/'y')."""
    for candidate in (f"{axis}_ccf", axis):
        for c in all_cols:
            if c.lower() == candidate:
                return c
    return None

def find_class_col(all_cols):
    """Find the ComBat-covariate class-label column regardless of its exact
    naming across files ('hrc_mmc_class_name' in most, but already just
    'class' in others, e.g. M2), case-insensitive either way."""
    for candidate in ("hrc_mmc_class_name", "class"):
        for c in all_cols:
            if c.lower() == candidate:
                return c
    return None

def get_load_info(fpath, gene_panel, meta_needed):
    hdr       = pd.read_csv(fpath, nrows=0)
    all_cols  = hdr.columns.tolist()
    idx_col   = all_cols[0]
    gene_cols = [g for g in gene_panel if g in all_cols]
    mcols     = [c for c in meta_needed  if c in all_cols]
    x_col     = find_coord_col(all_cols, "x")
    y_col     = find_coord_col(all_cols, "y")
    class_col = find_class_col(all_cols)
    if x_col is None:
        raise ValueError(
            f"Could not find an x-coordinate column (looked for 'x_CCF', "
            f"'x_ccf', or 'x', case-insensitive) in {fpath}"
        )
    if y_col is None:
        raise ValueError(
            f"Could not find a y-coordinate column (looked for 'y_CCF', "
            f"'y_ccf', or 'y', case-insensitive) in {fpath}"
        )
    if class_col is None:
        raise ValueError(
            f"Could not find a class-label column (looked for "
            f"'hrc_mmc_class_name' or 'class', case-insensitive) in {fpath} "
            f"-- this animal would silently get an all-NaN 'class' column "
            f"after concatenation, which breaks the ComBat covariate "
            f"regression for every animal (not just this one)."
        )
    for c in (x_col, y_col, class_col):
        if c not in mcols:
            mcols = mcols + [c]
    keep      = [idx_col] + gene_cols + mcols
    print(f"  {os.path.basename(fpath):<46}: "
          f"{len(gene_cols)} genes + {len(mcols)} meta cols  "
          f"(x-coord: '{x_col}', y-coord: '{y_col}', class: '{class_col}')")
    return idx_col, keep, gene_cols, x_col, y_col, class_col

cols_info = {}
x_col_map = {}
y_col_map = {}
class_col_map = {}
for fpath, animal_id, _ in animal_paths:
    idx_col, keep_cols, gene_cols_in_file, x_col, y_col, class_col = get_load_info(fpath, mouse_panel, meta_cols_needed)
    cols_info[animal_id] = (idx_col, keep_cols, gene_cols_in_file)
    x_col_map[animal_id]     = x_col
    y_col_map[animal_id]     = y_col
    class_col_map[animal_id] = class_col

# Print each animal's actual class values side by side, so a category that's
# spelled differently across files (e.g. "Glutamatergic" vs "glutamatergic")
# is visible directly, rather than only showing up later as a mysterious
# ComBat NaN failure.
print("\n  Checking class-label values per animal (column name varies per file, see above):")
class_values_by_animal = {}
for fpath, animal_id, _ in animal_paths:
    this_class_col = class_col_map[animal_id]
    class_col_sample = pd.read_csv(fpath, usecols=[this_class_col])[this_class_col]
    unique_vals = sorted(class_col_sample.dropna().unique().tolist())
    class_values_by_animal[animal_id] = set(unique_vals)
    print(f"    {animal_id} ('{this_class_col}'): {len(unique_vals)} distinct class values: {unique_vals}")

all_class_values = set().union(*class_values_by_animal.values())
for animal_id, vals in class_values_by_animal.items():
    missing = all_class_values - vals
    if missing:
        print(f"    NOTE: {animal_id} is missing {len(missing)} class value(s) present in "
              f"other animals: {sorted(missing)} -- if these are the SAME underlying "
              f"biological classes just spelled differently, ComBat's covariate "
              f"regression will treat them as animal-exclusive categories and can fail.")

verified_genes = [
    g for g in mouse_panel
    if all(g in cols_info[aid][2] for _, aid, _ in animal_paths)
]
print(f"\nVerified genes present in all 4 files: {len(verified_genes)}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. REGION LABEL ASSIGNMENT FUNCTION
#    Regions are defined PURELY by parcellation_structure membership in this
#    list — no hierarchical fallback through parcellation_division (no
#    hippocampus/striatum/hypothalamus/amygdala regions this time, and no use
#    of parcellation_division at all). A cell qualifies as ROI iff its
#    parcellation_structure is exactly one of these 18 isocortex/olfactory
#    areas. TT (taenia tecta) and DP (dorsal peduncular area) added as plain,
#    unsplit regions -- no rostral/caudal or dorsal/ventral split applied to
#    either (only PL/ACAd/ACAv/ILA get r/c, and only ILA additionally gets d/v).
# ══════════════════════════════════════════════════════════════════════════════
structure_regions = [
    "FRP", "ACAd", "ACAv", "PL", "ILA", "ORBl", "ORBm", "ORBvl",
    "MOs", "MOp", "RSPd", "RSPv", "RSPagl", "AId", "AIv", "AIp",
    "TT", "DP",
]

region_rename = {s: f"M_{s}" for s in structure_regions}

# Regions to further split into rostral (r) / caudal (c) halves by x_CCF,
# PER ANIMAL (midpoint = that animal's own min/max x_CCF within that region;
# x < midpoint -> rostral, x >= midpoint -> caudal). Done per-animal rather
# than with one global cutoff because different animals' x_CCF coordinates
# can sit on very different absolute scales — a per-animal relative midpoint
# is robust to that regardless of the underlying cause.
RC_SPLIT_REGIONS = ["PL", "ACAd", "ACAv"]

# Region(s) to split into dorsal (d) / ventral (v) by y_CCF, PER ANIMAL,
# instead of rostral/caudal. NOT in RC_SPLIT_REGIONS -- ILA receives this
# split ONLY, applied directly in Step 5B (immediately after the
# rostral/caudal split, same point in the pipeline), replacing M_ILA with
# M_ILAd/M_ILAv directly. No duplication: each ILA cell ends up with exactly
# one label. Same per-animal-relative-midpoint logic as the rostral/caudal
# split, just on the y axis: y < midpoint -> dorsal, y >= midpoint -> ventral.
DV_SPLIT_REGIONS = ["ILA"]

# Which animals share ONE common midpoint (per region) vs. which get their
# own SEPARATE midpoint, for both the x_CCF (rostral/caudal) and y_CCF
# (dorsal/ventral) splits. F1/F2/M1 sit on a similar absolute coordinate
# scale, so pooling them gives a single, better-supported midpoint estimate
# using more data. M2 sits on a genuinely different scale (~40-60x smaller
# range, per the Step 5 scale-consistency diagnostic) -- pooling it in would
# make the common midpoint meaningless for everyone, so it gets its own
# midpoint computed only from its own cells, same as the original
# per-animal design.
POOLED_MIDPOINT_ANIMALS   = ["F1", "F2", "M1"]
SEPARATE_MIDPOINT_ANIMALS = ["M2"]

def assign_region_vectorized(struct_arr):
    region = np.full(len(struct_arr), None, dtype=object)
    for s in structure_regions:
        region[struct_arr == s] = s
    return region

# ══════════════════════════════════════════════════════════════════════════════
# 4. FULL-DATASET PASS 1 — Load ALL cells (whole brain), assign M_region labels
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4: Loading ALL cells (whole brain) with M_region labels attached")
print("=" * 60)

META_CHUNK = 200000
all_pieces = []

for fpath, animal_id, sex in animal_paths:
    idx_col, keep_cols, gene_cols_in_file = cols_info[animal_id]
    gene_load_cols = [idx_col] + [g for g in verified_genes if g in gene_cols_in_file]

    print(f"\n  {animal_id}: loading ALL cells (genes + meta) in chunks...")

    meta_only_cols = [idx_col] + [
        c for c in keep_cols if c != idx_col and c not in verified_genes
    ]

    meta_reader = pd.read_csv(
        fpath, usecols=meta_only_cols, index_col=idx_col,
        low_memory=False, chunksize=META_CHUNK
    )
    gene_reader = pd.read_csv(
        fpath, usecols=gene_load_cols, index_col=idx_col,
        low_memory=False, chunksize=META_CHUNK,
        dtype={g: np.float32 for g in verified_genes if g in gene_load_cols}
    )

    n_seen = 0
    animal_X_chunks    = []
    animal_meta_chunks = []

    for chunk_meta, chunk_genes in zip(meta_reader, gene_reader):
        chunk_meta = chunk_meta.copy()
        chunk_meta.rename(columns=rename_map, inplace=True)
        x_col_this_animal = x_col_map[animal_id]
        y_col_this_animal = y_col_map[animal_id]
        class_col_this_animal = class_col_map[animal_id]
        if x_col_this_animal != "x":
            chunk_meta.rename(columns={x_col_this_animal: "x"}, inplace=True)
        if y_col_this_animal != "y":
            chunk_meta.rename(columns={y_col_this_animal: "y"}, inplace=True)
        if class_col_this_animal != "class":
            chunk_meta.rename(columns={class_col_this_animal: "class"}, inplace=True)
        chunk_meta["animal_id"] = animal_id
        chunk_meta["sex"]       = sex

        struct_arr = chunk_meta.get(
            "parcellation_structure",
            pd.Series([""] * len(chunk_meta), index=chunk_meta.index)
        ).fillna("").values

        raw_labels = assign_region_vectorized(struct_arr)
        m_labels   = np.where(
            raw_labels != None,
            np.vectorize(lambda r: region_rename.get(r, None))(raw_labels),
            None
        )
        chunk_meta["M_region"] = m_labels

        X_chunk = chunk_genes.loc[chunk_meta.index, verified_genes].values.astype(np.float32)

        animal_X_chunks.append(X_chunk)
        animal_meta_chunks.append(chunk_meta)
        n_seen += len(chunk_meta)

    del meta_reader, gene_reader; gc.collect()

    X_animal    = np.vstack(animal_X_chunks).astype(np.float32)
    meta_animal = pd.concat(animal_meta_chunks, axis=0)
    del animal_X_chunks, animal_meta_chunks; gc.collect()

    n_roi_animal = meta_animal["M_region"].notna().sum()
    print(f"  {animal_id}: {n_seen:,} total cells loaded "
          f"({n_roi_animal:,} will be kept as ROI, {n_roi_animal/n_seen:.1%})")

    all_pieces.append((X_animal, meta_animal))
    gc.collect()

X_all_cells   = np.vstack([X for X, _ in all_pieces]).astype(np.float32)
obs_all_cells = pd.concat([m for _, m in all_pieces], axis=0)
obs_all_cells.reset_index(drop=True, inplace=True)
del all_pieces; gc.collect()

n_total_all = len(obs_all_cells)
n_roi_all   = obs_all_cells["M_region"].notna().sum()
print(f"\nFull dataset loaded : {X_all_cells.shape}")
print(f"Total cells (all)   : {n_total_all:,}")
print(f"ROI cells (subset)  : {n_roi_all:,}  ({n_roi_all/n_total_all:.1%})")

# ══════════════════════════════════════════════════════════════════════════════
# 5. ROSTRAL/CAUDAL SPLIT (PL, ACAd, ACAv, ILA) — then LOG1P (skipped)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5: Rostral/caudal split for PL, ACAd, ACAv")
print(f"        (common midpoint pooled across {POOLED_MIDPOINT_ANIMALS}; "
      f"separate midpoint per-animal for {SEPARATE_MIDPOINT_ANIMALS})")
print("=" * 60)

rc_diagnostics = []
for region in RC_SPLIT_REGIONS:
    region_mask_all = obs_all_cells["M_region"] == region_rename[region]

    # ---- Common midpoint, pooled across POOLED_MIDPOINT_ANIMALS ----
    pooled_mask = region_mask_all & obs_all_cells["animal_id"].isin(POOLED_MIDPOINT_ANIMALS)
    n_pooled = int(pooled_mask.sum())
    if n_pooled == 0:
        print(f"  WARNING: no cells in {region} across {POOLED_MIDPOINT_ANIMALS} — "
              f"skipping common-midpoint split for these animals")
    else:
        pooled_x = obs_all_cells.loc[pooled_mask, "x"].values
        pooled_x_min, pooled_x_max = float(np.min(pooled_x)), float(np.max(pooled_x))
        common_midpoint = (pooled_x_min + pooled_x_max) / 2.0
        print(f"  {region:<6} COMMON midpoint (pooled {POOLED_MIDPOINT_ANIMALS}, n={n_pooled:,}): "
              f"x range [{pooled_x_min:.2f}, {pooled_x_max:.2f}]  mid={common_midpoint:.2f}")

        for animal_id in POOLED_MIDPOINT_ANIMALS:
            mask = region_mask_all & (obs_all_cells["animal_id"] == animal_id)
            n_cells = int(mask.sum())
            if n_cells == 0:
                print(f"    WARNING: {animal_id} has 0 cells in {region} — skipping")
                continue

            x_vals = obs_all_cells.loc[mask, "x"].values
            is_rostral = x_vals < common_midpoint
            new_labels = np.where(is_rostral, f"M_{region}r", f"M_{region}c")
            obs_all_cells.loc[mask, "M_region"] = new_labels

            n_r = int(is_rostral.sum())
            n_c = n_cells - n_r
            rc_diagnostics.append({
                "region": region, "animal_id": animal_id, "midpoint_type": "common",
                "n_cells": n_cells, "x_min": float(np.min(x_vals)), "x_max": float(np.max(x_vals)),
                "midpoint": common_midpoint, "n_rostral": n_r, "n_caudal": n_c,
            })
            print(f"    {region:<6} {animal_id}: n={n_cells:,}  -> rostral={n_r:,}  caudal={n_c:,}")

    # ---- Separate midpoint, computed independently per animal ----
    for animal_id in SEPARATE_MIDPOINT_ANIMALS:
        mask = region_mask_all & (obs_all_cells["animal_id"] == animal_id)
        n_cells = int(mask.sum())
        if n_cells == 0:
            print(f"  WARNING: {animal_id} has 0 cells in {region} — skipping (no split possible)")
            continue

        x_vals = obs_all_cells.loc[mask, "x"].values
        x_min, x_max = float(np.min(x_vals)), float(np.max(x_vals))
        midpoint = (x_min + x_max) / 2.0

        is_rostral = x_vals < midpoint
        new_labels = np.where(is_rostral, f"M_{region}r", f"M_{region}c")
        obs_all_cells.loc[mask, "M_region"] = new_labels

        n_r = int(is_rostral.sum())
        n_c = n_cells - n_r
        rc_diagnostics.append({
            "region": region, "animal_id": animal_id, "midpoint_type": "separate",
            "n_cells": n_cells, "x_min": x_min, "x_max": x_max,
            "midpoint": midpoint, "n_rostral": n_r, "n_caudal": n_c,
        })
        print(f"  {region:<6} SEPARATE {animal_id}: n={n_cells:,}  x range [{x_min:.2f}, {x_max:.2f}]  "
              f"mid={midpoint:.2f}  -> rostral={n_r:,}  caudal={n_c:,}")

rc_diag_df = pd.DataFrame(rc_diagnostics)
rc_diag_df.to_csv(os.path.join(SAVE_DIR, "rostral_caudal_split_diagnostics.csv"), index=False)

print("\nScale consistency check (max range / min range across animals, per region):")
for region in RC_SPLIT_REGIONS:
    sub = rc_diag_df[rc_diag_df["region"] == region].copy()
    if sub.empty:
        continue
    sub["range"] = sub["x_max"] - sub["x_min"]
    ratio = sub["range"].max() / max(sub["range"].min(), 1e-9)
    flag = "  <-- WARNING: >5x scale difference across animals" if ratio > 5 else ""
    print(f"  {region:<6}: range ratio = {ratio:.1f}x{flag}")

n_regions_after_rc = obs_all_cells["M_region"].nunique()
print(f"\nRegions after rostral/caudal split: {n_regions_after_rc} "
      f"({len(structure_regions) - len(RC_SPLIT_REGIONS) - len(DV_SPLIT_REGIONS)} still-unsplit + "
      f"{len(RC_SPLIT_REGIONS)} x 2 r/c-split = "
      f"{(len(structure_regions) - len(RC_SPLIT_REGIONS) - len(DV_SPLIT_REGIONS)) + 2*len(RC_SPLIT_REGIONS) + len(DV_SPLIT_REGIONS)} "
      f"before the dorsal/ventral split just below)")

# ══════════════════════════════════════════════════════════════════════════════
# 5B. DORSAL/VENTRAL SPLIT (ILA) — a direct split, exactly like the
#     rostral/caudal split above, just on the y axis instead of x. ILA is NOT
#     in RC_SPLIT_REGIONS, so it never gets rostral/caudal labels at all —
#     this is the ONLY split ILA receives, replacing M_ILA with M_ILAd /
#     M_ILAv directly (no duplication, no rostral/caudal version kept
#     alongside).
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5B: Dorsal/ventral split for ILA (the ONLY split ILA gets)")
print(f"         (common midpoint pooled across {POOLED_MIDPOINT_ANIMALS}; "
      f"separate midpoint per-animal for {SEPARATE_MIDPOINT_ANIMALS})")
print("=" * 60)

dv_diagnostics = []
for region in DV_SPLIT_REGIONS:
    region_mask_all = obs_all_cells["M_region"] == region_rename[region]

    pooled_mask = region_mask_all & obs_all_cells["animal_id"].isin(POOLED_MIDPOINT_ANIMALS)
    n_pooled = int(pooled_mask.sum())
    if n_pooled == 0:
        print(f"  WARNING: no cells in {region} across {POOLED_MIDPOINT_ANIMALS} — "
              f"skipping common-midpoint split for these animals")
    else:
        pooled_y = obs_all_cells.loc[pooled_mask, "y"].values
        pooled_y_min, pooled_y_max = float(np.min(pooled_y)), float(np.max(pooled_y))
        common_midpoint = (pooled_y_min + pooled_y_max) / 2.0
        print(f"  {region:<6} COMMON midpoint (pooled {POOLED_MIDPOINT_ANIMALS}, n={n_pooled:,}): "
              f"y range [{pooled_y_min:.2f}, {pooled_y_max:.2f}]  mid={common_midpoint:.2f}")

        for animal_id in POOLED_MIDPOINT_ANIMALS:
            mask = region_mask_all & (obs_all_cells["animal_id"] == animal_id)
            n_cells = int(mask.sum())
            if n_cells == 0:
                print(f"    WARNING: {animal_id} has 0 cells in {region} — skipping")
                continue

            y_vals = obs_all_cells.loc[mask, "y"].values
            is_dorsal = y_vals < common_midpoint
            new_labels = np.where(is_dorsal, f"M_{region}d", f"M_{region}v")
            obs_all_cells.loc[mask, "M_region"] = new_labels

            n_d = int(is_dorsal.sum())
            n_v = n_cells - n_d
            dv_diagnostics.append({
                "region": region, "animal_id": animal_id, "midpoint_type": "common",
                "n_cells": n_cells, "y_min": float(np.min(y_vals)), "y_max": float(np.max(y_vals)),
                "midpoint": common_midpoint, "n_dorsal": n_d, "n_ventral": n_v,
            })
            print(f"    {region:<6} {animal_id}: n={n_cells:,}  -> dorsal={n_d:,}  ventral={n_v:,}")

    for animal_id in SEPARATE_MIDPOINT_ANIMALS:
        mask = region_mask_all & (obs_all_cells["animal_id"] == animal_id)
        n_cells = int(mask.sum())
        if n_cells == 0:
            print(f"  WARNING: {animal_id} has 0 cells in {region} — skipping (no split possible)")
            continue

        y_vals = obs_all_cells.loc[mask, "y"].values
        y_min, y_max = float(np.min(y_vals)), float(np.max(y_vals))
        midpoint = (y_min + y_max) / 2.0

        is_dorsal = y_vals < midpoint
        new_labels = np.where(is_dorsal, f"M_{region}d", f"M_{region}v")
        obs_all_cells.loc[mask, "M_region"] = new_labels

        n_d = int(is_dorsal.sum())
        n_v = n_cells - n_d
        dv_diagnostics.append({
            "region": region, "animal_id": animal_id, "midpoint_type": "separate",
            "n_cells": n_cells, "y_min": y_min, "y_max": y_max,
            "midpoint": midpoint, "n_dorsal": n_d, "n_ventral": n_v,
        })
        print(f"  {region:<6} SEPARATE {animal_id}: n={n_cells:,}  y range [{y_min:.2f}, {y_max:.2f}]  "
              f"mid={midpoint:.2f}  -> dorsal={n_d:,}  ventral={n_v:,}")

dv_diag_df = pd.DataFrame(dv_diagnostics)
dv_diag_df.to_csv(os.path.join(SAVE_DIR, "dorsal_ventral_split_diagnostics.csv"), index=False)

print("\nScale consistency check (max range / min range across animals, per region):")
for region in DV_SPLIT_REGIONS:
    sub = dv_diag_df[dv_diag_df["region"] == region].copy()
    if sub.empty:
        continue
    sub["range"] = sub["y_max"] - sub["y_min"]
    ratio = sub["range"].max() / max(sub["range"].min(), 1e-9)
    flag = "  <-- WARNING: >5x scale difference across animals" if ratio > 5 else ""
    print(f"  {region:<6}: range ratio = {ratio:.1f}x{flag}")

n_regions_now = obs_all_cells["M_region"].nunique()
print(f"\nRegions after rostral/caudal + dorsal/ventral split: {n_regions_now} "
      f"({(len(structure_regions) - len(RC_SPLIT_REGIONS) - len(DV_SPLIT_REGIONS)) + 2*len(RC_SPLIT_REGIONS) + 2*len(DV_SPLIT_REGIONS)} expected)")

print("\nLog1p transform SKIPPED — input data is already log-transformed")
X_all_log = X_all_cells  # already log-transformed upstream; no np.log1p() applied here
del X_all_cells; gc.collect()

# ══════════════════════════════════════════════════════════════════════════════
# 6. IDENTIFY ROI CELLS — NOT FILTERING YET.
#    We need to know which whole-brain positions are ROI cells (and their
#    M_region labels) in order to define the train/val/test split next, but
#    we deliberately do NOT drop non-ROI cells here: ComBat (Step 8) is fit
#    using the whole-brain population, and only ROI cells get filtered out
#    afterward (Step 9).
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 6: Identifying ROI cells (whole brain retained for ComBat)")
print("=" * 60)

roi_mask = obs_all_cells["M_region"].notna().values
roi_positions = np.where(roi_mask)[0]
print(f"ROI mask: {roi_mask.sum():,} / {len(roi_mask):,} cells")

subclass_col = next((c for c in ["subclass", "supercluster",
                                   "parcellation_substructure", "cluster"]
                     if c in obs_all_cells.columns), None)

if "class" not in obs_all_cells.columns:
    raise ValueError(
        "'class' column not found after renaming — this should be unreachable "
        "now that Step 2's find_class_col() fails fast per-file if neither "
        "'hrc_mmc_class_name' nor 'class' is found."
    )

print(f"Subclass column : '{subclass_col}'")
print(f"\nM_region counts (ROI cells only, whole-brain array not yet filtered):")
print(obs_all_cells.loc[roi_mask, "M_region"].value_counts().to_string())

# ══════════════════════════════════════════════════════════════════════════════
# 7. TRAIN / VAL / TEST SPLIT — ROI CELLS ONLY, done BEFORE ComBat fitting.
#    Indices below (roi_idx_train_wb etc.) are positions in the WHOLE-BRAIN
#    array (obs_all_cells / X_all_log), since filtering to ROI hasn't
#    happened yet. They get remapped to ROI-only positions in Step 9.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 7: Train / Val / Test split on ROI cells (70/10/20) — PRE-ComBat")
print("=" * 60)

le_region_pre = LabelEncoder()
y_region_roi  = le_region_pre.fit_transform(obs_all_cells.loc[roi_mask, "M_region"].values)

n_total = len(roi_positions)
roi_idx_trainval, roi_idx_test = train_test_split(
    np.arange(n_total), test_size=0.20,
    stratify=y_region_roi, random_state=42
)
roi_idx_train, roi_idx_val = train_test_split(
    roi_idx_trainval, test_size=0.125,
    stratify=y_region_roi[roi_idx_trainval], random_state=42
)

# Convert from "position within roi_positions" to "position within the
# whole-brain array" (obs_all_cells / X_all_log row indices).
roi_idx_train_wb = roi_positions[roi_idx_train]
roi_idx_val_wb   = roi_positions[roi_idx_val]
roi_idx_test_wb  = roi_positions[roi_idx_test]

print(f"Total ROI cells : {n_total:,}")
print(f"Train : {len(roi_idx_train_wb):,}  ({len(roi_idx_train_wb)/n_total:.0%})")
print(f"Val   : {len(roi_idx_val_wb):,}   ({len(roi_idx_val_wb)/n_total:.0%})")
print(f"Test  : {len(roi_idx_test_wb):,}   ({len(roi_idx_test_wb)/n_total:.0%})")

# ══════════════════════════════════════════════════════════════════════════════
# 8. COMBAT — FIT ON (ALL NON-ROI WHOLE-BRAIN CELLS) + (ROI TRAIN CELLS),
#    APPLY TO EVERY CELL IN THE WHOLE BRAIN. ROI val/test cells are excluded
#    from the fit so they're never seen by it (same no-leakage guarantee as
#    before)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 8: ComBat batch correction — fit on WHOLE BRAIN minus ROI val/test")
print("=" * 60)
print("Batch variable : animal_id")
print("Covariate      : class")
print("Fit pool       : all non-ROI whole-brain cells + ROI TRAIN cells only")
print("                 (ROI val/test cells excluded from the fit)")
print("Applied to     : every cell in the whole brain, via learned per-animal")
print("                 linear params, THEN filtered down to ROI cells (Step 9)")

rng = np.random.default_rng(42)

non_roi_positions   = np.where(~roi_mask)[0]
non_roi_animal_ids  = obs_all_cells.iloc[non_roi_positions]["animal_id"].values
roi_train_animal_ids = obs_all_cells.iloc[roi_idx_train_wb]["animal_id"].values

sub_idx_list = []
for animal_id in obs_all_cells["animal_id"].unique():
    animal_roi_train_pool = roi_idx_train_wb[roi_train_animal_ids == animal_id]
    animal_non_roi_pool   = non_roi_positions[non_roi_animal_ids == animal_id]

    roi_target    = int(round(N_PER_ANIMAL_COMBAT_TRAIN * ROI_TRAIN_FIT_FRACTION))
    n_roi_use     = min(len(animal_roi_train_pool), roi_target)
    remaining     = N_PER_ANIMAL_COMBAT_TRAIN - n_roi_use
    n_non_roi_use = min(len(animal_non_roi_pool), remaining)
    leftover      = remaining - n_non_roi_use
    if leftover > 0:
        extra_roi  = min(len(animal_roi_train_pool) - n_roi_use, leftover)
        n_roi_use += extra_roi

    chosen_roi     = rng.choice(animal_roi_train_pool, size=n_roi_use, replace=False)
    chosen_non_roi = rng.choice(animal_non_roi_pool,   size=n_non_roi_use, replace=False)
    combined = np.concatenate([chosen_roi, chosen_non_roi])
    sub_idx_list.append(combined)

    print(f"  {animal_id}: {n_roi_use:,} / {len(animal_roi_train_pool):,} ROI-train + "
          f"{n_non_roi_use:,} / {len(animal_non_roi_pool):,} non-ROI sampled "
          f"= {len(combined):,} total for ComBat fit "
          f"(budget={N_PER_ANIMAL_COMBAT_TRAIN:,}, target ROI share={ROI_TRAIN_FIT_FRACTION:.0%})")

sub_idx = np.sort(np.concatenate(sub_idx_list))
print(f"\nTotal ComBat fit-pool size across all animals: {len(sub_idx):,}")

if len(sub_idx) > MAX_COMBAT_FIT_POOL_TOTAL:
    print(f"WARNING: total fit-pool size {len(sub_idx):,} exceeds the hard cap "
          f"MAX_COMBAT_FIT_POOL_TOTAL={MAX_COMBAT_FIT_POOL_TOTAL:,} — downsampling "
          f"(stratified by animal_id) to avoid an out-of-memory failure in ComBat.")
    sub_idx_animal_ids = obs_all_cells.iloc[sub_idx]["animal_id"].values
    keep_frac = MAX_COMBAT_FIT_POOL_TOTAL / len(sub_idx)
    trimmed = []
    for animal_id in obs_all_cells["animal_id"].unique():
        this_animal_idx = sub_idx[sub_idx_animal_ids == animal_id]
        n_keep = max(1, int(round(len(this_animal_idx) * keep_frac)))
        trimmed.append(rng.choice(this_animal_idx, size=n_keep, replace=False))
    sub_idx = np.sort(np.concatenate(trimmed))
    print(f"Total ComBat fit-pool size after cap: {len(sub_idx):,}")

X_sub   = X_all_log[sub_idx].astype(np.float32)
obs_sub = obs_all_cells.iloc[sub_idx][["animal_id", "class"]].copy()
obs_sub.reset_index(drop=True, inplace=True)

print(f"\nSubsample shape : {X_sub.shape}")
print(f"Memory          : {X_sub.nbytes/1e9:.2f} GB")

class_dummies = pd.get_dummies(obs_sub["class"],     drop_first=True)
batch_dummies = pd.get_dummies(obs_sub["animal_id"], drop_first=True)
intercept     = pd.Series(1, index=obs_sub.index, name="intercept")
design_check  = pd.concat(
    [intercept, batch_dummies, class_dummies], axis=1
).values.astype(float)
rank   = np.linalg.matrix_rank(design_check)
n_cols = design_check.shape[1]
print(f"\nDesign matrix: rank={rank}, cols={n_cols}, full_rank={rank == n_cols}")

combat_covariates = ["class"] if rank == n_cols else None
if combat_covariates is None:
    print("WARNING: design not full rank — running ComBat without covariates")
print(f"ComBat covariates: {combat_covariates}")

print(f"\nFitting ComBat on {len(X_sub):,} whole-brain fit-pool cells (log-transformed)...")
adata_sub           = ad.AnnData(X=X_sub.copy(), obs=obs_sub.copy())
adata_sub.var_names = verified_genes

sc.pp.combat(
    adata_sub, key="animal_id", covariates=combat_covariates, inplace=True
)
X_sub_corrected = adata_sub.X.copy().astype(np.float32)

n_nan = np.isnan(X_sub_corrected).sum()
if n_nan > 0:
    print(f"WARNING: ComBat with covariates={combat_covariates} produced "
          f"{n_nan:,} NaN values ({n_nan / X_sub_corrected.size:.4%} of entries).")
    print("This typically means some covariate category is so imbalanced across")
    print("animals (e.g. present almost entirely in one animal) that the per-batch")
    print("regression is singular for some genes. Refitting WITHOUT covariates as")
    print("a fallback (batch correction on animal_id only).")
    adata_sub = ad.AnnData(X=X_sub.copy(), obs=obs_sub.copy())
    adata_sub.var_names = verified_genes
    sc.pp.combat(adata_sub, key="animal_id", covariates=None, inplace=True)
    X_sub_corrected = adata_sub.X.copy().astype(np.float32)
    combat_covariates = None
    n_nan_retry = np.isnan(X_sub_corrected).sum()
    if n_nan_retry > 0:
        raise RuntimeError(
            f"ComBat still produced {n_nan_retry:,} NaNs even without covariates — "
            f"inspect X_sub for zero-variance genes within a batch before proceeding."
        )
    print("Retry without covariates succeeded — no NaNs.")

print(f"ComBat complete.")
print(f"Corrected range: {X_sub_corrected.min():.3f} – {X_sub_corrected.max():.3f}")

print("\nExtracting per-batch linear correction parameters (from fit pool only)...")
batch_params = {}
for animal_id in obs_sub["animal_id"].unique():
    mask     = obs_sub["animal_id"].values == animal_id
    X_before = X_sub[mask]
    X_after  = X_sub_corrected[mask]
    std_b    = X_before.std(axis=0) + 1e-8
    std_a    = X_after.std(axis=0)
    mean_b   = X_before.mean(axis=0)
    mean_a   = X_after.mean(axis=0)
    alpha    = std_a / std_b
    beta     = mean_a - alpha * mean_b
    batch_params[animal_id] = {"alpha": alpha, "beta": beta}
    print(f"  {animal_id}: alpha mean={alpha.mean():.4f}  beta mean={beta.mean():.4f}")

del adata_sub, X_sub, X_sub_corrected; gc.collect()

print(f"\nApplying fit-pool-derived correction to the ENTIRE whole-brain array...")
X_all_corrected = np.empty_like(X_all_log)
CHUNK_CB = 200000
for start in range(0, len(X_all_log), CHUNK_CB):
    end        = min(start + CHUNK_CB, len(X_all_log))
    chunk_obs  = obs_all_cells.iloc[start:end]
    chunk_X    = X_all_log[start:end].copy()
    for animal_id in chunk_obs["animal_id"].unique():
        mask          = chunk_obs["animal_id"].values == animal_id
        chunk_X[mask] = (chunk_X[mask]
                          * batch_params[animal_id]["alpha"]
                          + batch_params[animal_id]["beta"])
    X_all_corrected[start:end] = np.clip(chunk_X, 0, None)

print(f"Corrected range: {X_all_corrected.min():.3f} – {X_all_corrected.max():.3f}")
print(f"ComBat covariates actually used: {combat_covariates}")

if np.isnan(X_all_corrected).any():
    raise RuntimeError(
        "X_all_corrected contains NaNs after applying the learned batch "
        "parameters — do not proceed to training on this data."
    )

print("\nBatch effect reduction check (WHOLE BRAIN, all cells):")
for label, X_check in [("pre-ComBat (log)",  X_all_log),
                         ("post-ComBat (log)", X_all_corrected)]:
    means = [X_check[obs_all_cells["animal_id"].values == aid].mean(axis=0)
             for aid in obs_all_cells["animal_id"].unique()]
    cv    = np.stack(means).var(axis=0).mean()
    print(f"  {label:<20}: cross-animal gene variance = {cv:.6f}")

del X_all_log; gc.collect()

# ══════════════════════════════════════════════════════════════════════════════
# 9. FILTER TO ROI CELLS — now that ComBat has run on the whole brain.
#    Remap the whole-brain train/val/test position arrays (Step 7) into
#    positions within the ROI-filtered array below.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 9: Filtering to ROI cells (post-ComBat)")
print("=" * 60)

X_roi_corrected = X_all_corrected[roi_mask].astype(np.float32)
obs_all         = obs_all_cells.loc[roi_mask].copy()
obs_all.reset_index(drop=True, inplace=True)

# Lookup: whole-brain position -> position within the ROI-filtered array
old_to_new_pos = np.full(len(roi_mask), -1, dtype=np.int64)
old_to_new_pos[roi_positions] = np.arange(len(roi_positions))

idx_train = old_to_new_pos[roi_idx_train_wb]
idx_val   = old_to_new_pos[roi_idx_val_wb]
idx_test  = old_to_new_pos[roi_idx_test_wb]
assert (idx_train >= 0).all() and (idx_val >= 0).all() and (idx_test >= 0).all()

print(f"X_roi_corrected (ROI only, post-ComBat) : {X_roi_corrected.shape}")
print(f"Train : {len(idx_train):,}  Val : {len(idx_val):,}  Test : {len(idx_test):,}")
print(f"\nM_region counts (ROI only, post-filter):")
print(obs_all["M_region"].value_counts().to_string())

del X_all_corrected, obs_all_cells; gc.collect()
X_encoder_input = X_roi_corrected

# ══════════════════════════════════════════════════════════════════════════════
# 10. VALIDATE THE 6 FIXED CELLTYPES ARE PRESENT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 10: Validating the 6 fixed celltypes are present")
print("=" * 60)

missing_ct = [ct for ct in CELLTYPES_OF_INTEREST if ct not in obs_all[subclass_col].unique()]
if missing_ct:
    raise ValueError(f"These celltypes aren't present in the ROI data: {missing_ct}")

qualifying_celltypes = CELLTYPES_OF_INTEREST
for ct in qualifying_celltypes:
    n_ct = (obs_all[subclass_col].values == ct).sum()
    print(f"  {ct:<28}: {n_ct:,} pooled ROI cells")

with open(os.path.join(SAVE_DIR, "qualifying_celltypes.txt"), "w") as f:
    for ct in qualifying_celltypes:
        f.write(ct + "\n")

# ══════════════════════════════════════════════════════════════════════════════
# 11. HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def sanitize(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))

def stratified_subsample(idx, y, max_n, rng):
    """Return a stratified-by-y subsample of idx capped at max_n, else idx unchanged."""
    if len(idx) <= max_n:
        return idx
    frac = max_n / len(idx)
    keep_parts = []
    for cls in np.unique(y):
        cls_idx = idx[y == cls]
        n_keep  = max(1, int(round(len(cls_idx) * frac)))
        n_keep  = min(n_keep, len(cls_idx))
        chosen  = rng.choice(cls_idx, size=n_keep, replace=False)
        keep_parts.append(chosen)
    out = np.concatenate(keep_parts)
    rng.shuffle(out)
    return out[:max_n] if len(out) > max_n else out

def plot_roc_multiclass(true_y, prob_y, class_names, title, save_path=None):
    n_cls = len(class_names)
    if n_cls == 2:
        # label_binarize collapses to a single column (not one-hot) when there
        # are exactly 2 classes, since one-vs-rest is degenerate for binary
        # classification. Build a proper 2-column one-hot manually so the
        # per-class loop below (indexing column i for each of the n_cls
        # classes) doesn't run out of columns.
        y_bin = np.zeros((len(true_y), 2), dtype=int)
        y_bin[np.arange(len(true_y)), true_y] = 1
    else:
        y_bin = label_binarize(true_y, classes=np.arange(n_cls))
    auc_dict = {}
    fig, ax  = plt.subplots(figsize=(8, 6.5))
    colors   = plt.cm.tab10(np.linspace(0, 1, max(n_cls, 2)))
    fpr_all, tpr_all = [], []
    for i, (cls, col) in enumerate(zip(class_names, colors)):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _   = roc_curve(y_bin[:, i], prob_y[:, i])
        roc_auc       = auc(fpr, tpr)
        auc_dict[cls] = roc_auc
        ax.plot(fpr, tpr, color=col, lw=1.5,
                label=f"{cls}  (AUC={roc_auc:.3f})")
        fpr_all.append(fpr); tpr_all.append(tpr)
    macro_auc = np.nan
    if fpr_all:
        all_fpr  = np.unique(np.concatenate(fpr_all))
        mean_tpr = np.zeros_like(all_fpr)
        for fpr_i, tpr_i in zip(fpr_all, tpr_all):
            mean_tpr += np.interp(all_fpr, fpr_i, tpr_i)
        mean_tpr /= len(fpr_all)
        macro_auc = auc(all_fpr, mean_tpr)
        ax.plot(all_fpr, mean_tpr, "k--", lw=2,
                label=f"Macro avg  (AUC={macro_auc:.3f})")
    ax.plot([0,1],[0,1],"gray",lw=0.8,linestyle=":")
    ax.set_xlim([-0.02,1.02]); ax.set_ylim([-0.02,1.05])
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.7)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return auc_dict, macro_auc

def cluster_regions_by_confusion(cm_norm, region_names, ct_dir, ct_tag):
    """Hierarchical clustering of regions using this celltype's OWN confusion
    matrix, PLUS spectral clustering on the same symmetrized similarity as a
    second, independent method, PLUS a version of the confusion matrix
    reordered by the dendrogram's own leaf order so real block structure (or
    the lack of it) is directly visible rather than inferred from the tree
    alone. Returns (Z linkage matrix, distance_df)."""
    similarity = (cm_norm + cm_norm.T) / 2
    np.fill_diagonal(similarity, 1.0)
    distance = 1 - similarity
    np.fill_diagonal(distance, 0.0)
    distance = (distance + distance.T) / 2

    distance_df = pd.DataFrame(distance, index=region_names, columns=region_names)
    distance_df.to_csv(os.path.join(ct_dir, "region_confusion_distance.csv"))

    n_reg = len(region_names)
    if n_reg < 2:
        return None, distance_df

    condensed = squareform(distance, checks=False)
    Z = linkage(condensed, method="average")

    fig, ax = plt.subplots(figsize=(max(7, 0.55 * n_reg + 3), 5.5))
    dendrogram(Z, labels=region_names, ax=ax, leaf_rotation=90, leaf_font_size=10)
    ax.set_title(f"Regions clustered by confusion pattern — {ct_tag}")
    ax.set_ylabel("Distance (1 \u2212 avg. symmetric confusion rate)")
    plt.tight_layout()
    plt.savefig(os.path.join(ct_dir, "region_confusion_dendrogram.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Dendrogram-reordered confusion matrix ────────────────────────────
    # Reorders BOTH axes of the raw (row-normalized) confusion matrix by the
    # dendrogram's own leaf order. If the clustering is capturing real
    # structure, confusion mass should visibly collapse into contiguous
    # blocks near the diagonal after reordering; if it doesn't, that's
    # direct evidence the tree isn't matching what the matrix shows.
    leaf_order = leaves_list(Z)
    ordered_names = [region_names[i] for i in leaf_order]
    cm_reordered = cm_norm[np.ix_(leaf_order, leaf_order)]

    fig, ax = plt.subplots(figsize=(1.1 * n_reg + 3, 1.0 * n_reg + 3))
    sns.heatmap(
        cm_reordered, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=ordered_names, yticklabels=ordered_names,
        linewidths=0.5, linecolor="white", ax=ax
    )
    ax.set_title(f"Confusion matrix reordered by dendrogram leaf order — {ct_tag}")
    ax.set_xlabel("Predicted (dendrogram order)"); ax.set_ylabel("True (dendrogram order)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(ct_dir, "confusion_matrix_dendrogram_ordered.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Spectral clustering on the same symmetrized similarity ───────────
    # Independent second method: hierarchical clustering forces a single
    # nested tree, which can bury a real-but-moderate pairwise relationship
    # if other pairs in the same plot are far more extreme. Spectral
    # clustering makes no such nesting assumption -- it groups by eigenvectors
    # of the similarity graph directly, so it can catch structure the
    # dendrogram's forced tree shape might obscure.
    spectral_summary = None
    if n_reg >= 4:
        k_range = range(2, min(8, n_reg))
        best_k, best_score, best_labels = None, -2, None
        for k in k_range:
            try:
                sc = SpectralClustering(
                    n_clusters=k, affinity="precomputed", random_state=42,
                    assign_labels="kmeans",
                )
                labels = sc.fit_predict(similarity)
                if len(set(labels)) < 2:
                    continue
                score = silhouette_score(distance, labels, metric="precomputed")
                if score > best_score:
                    best_k, best_score, best_labels = k, score, labels
            except Exception as e:
                print(f"    Spectral clustering failed at k={k}: {e}")
        if best_labels is not None:
            spectral_df = pd.DataFrame({
                "region": region_names, "spectral_cluster": best_labels,
            }).sort_values("spectral_cluster")
            spectral_df.to_csv(os.path.join(ct_dir, "spectral_clustering.csv"), index=False)
            spectral_summary = {"k": best_k, "silhouette": best_score,
                                 "clusters": spectral_df.groupby("spectral_cluster")["region"].apply(list).to_dict()}
            print(f"    Spectral clustering: best k={best_k} (silhouette={best_score:.3f})")
            for cid, regs in spectral_summary["clusters"].items():
                if len(regs) >= 2:
                    print(f"      Spectral cluster {cid}: {regs}")

            # Reorder + replot the confusion matrix by spectral cluster
            # assignment too, for direct visual comparison against the
            # dendrogram-ordered version above.
            spec_order = np.argsort(best_labels)
            spec_names = [region_names[i] for i in spec_order]
            cm_spec_ordered = cm_norm[np.ix_(spec_order, spec_order)]
            fig, ax = plt.subplots(figsize=(1.1 * n_reg + 3, 1.0 * n_reg + 3))
            sns.heatmap(
                cm_spec_ordered, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=spec_names, yticklabels=spec_names,
                linewidths=0.5, linecolor="white", ax=ax
            )
            ax.set_title(f"Confusion matrix reordered by spectral cluster (k={best_k}) — {ct_tag}")
            ax.set_xlabel("Predicted (spectral order)"); ax.set_ylabel("True (spectral order)")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(os.path.join(ct_dir, "confusion_matrix_spectral_ordered.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)
    else:
        print(f"    Skipping spectral clustering: only {n_reg} regions (need >= 4).")

    return Z, distance_df

# ══════════════════════════════════════════════════════════════════════════════
# 12. PER-CELLTYPE REGION CLASSIFICATION LOOP (SVM + hyperparameter tuning)
#     For each of the 6 fixed celltypes:
#       1. Reuse the SAME global train/val/test split from Step 7, filtered
#          to that celltype's cells (no new leakage introduced).
#       2. Drop M_region classes with too few training cells within this
#          celltype (MIN_TRAIN_PER_REGION_CT).
#       3. If n_train > MAX_SVM_TRAIN_CELLS, take a stratified subsample for
#          tuning + fitting (kernel SVMs don't scale past ~10-20k points).
#       4. GridSearchCV over kernel/C/gamma (StandardScaler -> SVC pipeline),
#          scoring=balanced_accuracy, tuned WITHOUT predict_proba (probability
#          estimation is expensive and unnecessary during search).
#       5. Refit the best hyperparameters once, WITH probability=True, to get
#          a model usable for ROC/AUC on val/test.
#       6. Evaluate on the FULL (non-subsampled) val + test splits.
#       7. Cluster this celltype's regions by confusion pattern (dendrogram).
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 12: Training one SVM region classifier PER celltype (6 total)")
print("=" * 60)
print(f"SVM param grid   : {SVM_PARAM_GRID}")
print(f"Tuning CV folds  : {SVM_CV_FOLDS}")
print(f"Tuning n_jobs    : {SVM_N_JOBS}  (1 = serial, avoids joblib temp-disk memmap usage)")
print(f"Tuning scoring   : {SVM_TUNE_SCORING}")
print(f"Max SVM train n  : {MAX_SVM_TRAIN_CELLS:,}  (stratified subsample above this)")
print(f"Cluster cut      : absolute distance threshold {CONFUSION_DISTANCE_THRESHOLD} "
      f"(regions with >{(1-CONFUSION_DISTANCE_THRESHOLD):.0%} avg mutual confusion get grouped)")

CELLTYPE_SAVE_ROOT = os.path.join(SAVE_DIR, "per_celltype_region_models")
os.makedirs(CELLTYPE_SAVE_ROOT, exist_ok=True)

rng_svm = np.random.default_rng(42)
celltype_summary = {}
all_celltype_cm_norm = {}  # ct -> cm_norm_df, collected for the aggregate dendrogram at the end

for ct in qualifying_celltypes:
    ct_tag  = sanitize(ct)
    ct_dir  = os.path.join(CELLTYPE_SAVE_ROOT, ct_tag)
    os.makedirs(ct_dir, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"# Cell type: {ct}")
    print(f"{'#'*70}")

    ct_mask     = (obs_all[subclass_col].values == ct)
    ct_pos_all  = np.where(ct_mask)[0]

    ct_idx_train_full = np.intersect1d(idx_train, ct_pos_all)
    ct_idx_val_full   = np.intersect1d(idx_val,   ct_pos_all)
    ct_idx_test_full  = np.intersect1d(idx_test,  ct_pos_all)

    # Only keep M_region classes with enough TRAINING cells within this celltype
    train_regions        = obs_all.iloc[ct_idx_train_full]["M_region"].values
    region_train_counts  = pd.Series(train_regions).value_counts()
    valid_regions = sorted(
        region_train_counts[region_train_counts >= MIN_TRAIN_PER_REGION_CT].index.tolist()
    )

    if len(valid_regions) < 2:
        print(f"  SKIPPING: fewer than 2 M_region classes with >= "
              f"{MIN_TRAIN_PER_REGION_CT} training cells for this celltype.")
        celltype_summary[ct] = {"status": "skipped_too_few_regions",
                                 "n_regions_available": len(valid_regions)}
        continue

    def filter_to_valid_regions(idx):
        regions = obs_all.iloc[idx]["M_region"].values
        return idx[np.isin(regions, valid_regions)]

    ct_idx_train = filter_to_valid_regions(ct_idx_train_full)
    ct_idx_val   = filter_to_valid_regions(ct_idx_val_full)
    ct_idx_test  = filter_to_valid_regions(ct_idx_test_full)

    if len(ct_idx_val) == 0 or len(ct_idx_test) == 0:
        print("  SKIPPING: empty val or test split after region filtering.")
        celltype_summary[ct] = {"status": "skipped_empty_val_or_test"}
        continue

    le_region_ct = LabelEncoder()
    le_region_ct.fit(valid_regions)
    n_regions_ct = len(le_region_ct.classes_)

    print(f"  M_regions used ({n_regions_ct}): {valid_regions}")
    print(f"  Train: {len(ct_idx_train):,}  Val: {len(ct_idx_val):,}  Test: {len(ct_idx_test):,}")

    y_train_ct_full = le_region_ct.transform(obs_all.iloc[ct_idx_train]["M_region"].values)

    # ---- Subsample training set for tuning/fitting if too large for kernel SVM ----
    n_train_ct = len(ct_idx_train)
    if n_train_ct > MAX_SVM_TRAIN_CELLS:
        ct_idx_train_fit = stratified_subsample(ct_idx_train, y_train_ct_full, MAX_SVM_TRAIN_CELLS, rng_svm)
        print(f"  NOTE: n_train ({n_train_ct:,}) exceeds MAX_SVM_TRAIN_CELLS "
              f"({MAX_SVM_TRAIN_CELLS:,}) — tuning/fitting on a stratified "
              f"subsample of {len(ct_idx_train_fit):,} cells instead. "
              f"Evaluation below still uses the FULL val/test splits.")
    else:
        ct_idx_train_fit = ct_idx_train

    X_train_fit = X_encoder_input[ct_idx_train_fit]
    y_train_fit = le_region_ct.transform(obs_all.iloc[ct_idx_train_fit]["M_region"].values)

    X_val   = X_encoder_input[ct_idx_val]
    y_val   = le_region_ct.transform(obs_all.iloc[ct_idx_val]["M_region"].values)
    X_test  = X_encoder_input[ct_idx_test]
    y_test  = le_region_ct.transform(obs_all.iloc[ct_idx_test]["M_region"].values)

    # ---- Hyperparameter tuning (no probability estimation during search) ----
    min_class_count = pd.Series(y_train_fit).value_counts().min()
    n_splits = min(SVM_CV_FOLDS, min_class_count)
    if n_splits < 2:
        print(f"  SKIPPING: smallest region class in the fitting set has only "
              f"{min_class_count} cell(s) — not enough for cross-validated tuning.")
        celltype_summary[ct] = {"status": "skipped_too_few_cells_for_cv"}
        continue

    search_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(class_weight="balanced", decision_function_shape="ovr", random_state=42)),
    ])
    cv_splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    grid = GridSearchCV(
        search_pipeline, SVM_PARAM_GRID, cv=cv_splitter,
        scoring=SVM_TUNE_SCORING, n_jobs=SVM_N_JOBS, refit=False
    )
    grid.fit(X_train_fit, y_train_fit)
    best_params = grid.best_params_
    print(f"  Best params ({SVM_TUNE_SCORING}={grid.best_score_:.3f}): {best_params}")

    # ---- Refit best hyperparameters once, WITH probability=True, for eval ----
    svc_kwargs = {k.replace("svc__", ""): v for k, v in best_params.items()}
    final_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(class_weight="balanced", decision_function_shape="ovr",
                    probability=True, random_state=42, **svc_kwargs)),
    ])
    final_pipeline.fit(X_train_fit, y_train_fit)

    # ---- Evaluate on VAL (sanity check) ----
    val_pred = final_pipeline.predict(X_val)
    val_acc  = (val_pred == y_val).mean()
    print(f"  VAL accuracy: {val_acc:.3f}  (n={len(y_val):,})")

    # ---- Evaluate on TEST ----
    test_pred = final_pipeline.predict(X_test)
    test_prob = final_pipeline.predict_proba(X_test)
    test_acc_all_regions = (test_pred == y_test).mean()

    # ---- Missing-data handling: regions with too few TEST cells are
    #      excluded from REPORTED metrics (classification report, confusion
    #      matrix, ROC, clustering) rather than having an unreliable
    #      precision/recall/F1 computed and presented as if trustworthy.
    #      Still fully used during training above -- this only affects what
    #      gets reported.
    test_region_labels_all = obs_all.iloc[ct_idx_test]["M_region"].values
    test_region_counts = pd.Series(test_region_labels_all).value_counts()
    reportable_regions = set(test_region_counts[test_region_counts >= MIN_TEST_CELLS_FOR_REPORTING].index)
    excluded_low_support = sorted(set(valid_regions) - reportable_regions)
    if excluded_low_support:
        excluded_counts = {r: int(test_region_counts.get(r, 0)) for r in excluded_low_support}
        print(f"  NOTE: excluding {excluded_low_support} from reported metrics/clustering "
              f"(< {MIN_TEST_CELLS_FOR_REPORTING} test cells: {excluded_counts} -- treated as "
              f"missing data, not computed as an unreliable metric). Still used in training above.")

    reportable_mask = np.isin(test_region_labels_all, list(reportable_regions))
    y_test_report    = y_test[reportable_mask]
    test_pred_report = test_pred[reportable_mask]
    test_prob_report = test_prob[reportable_mask]
    test_acc = (test_pred_report == y_test_report).mean() if reportable_mask.sum() > 0 else np.nan
    print(f"  TEST accuracy (all {len(valid_regions)} trained regions): {test_acc_all_regions:.3f}")
    print(f"  TEST accuracy (reportable regions only, >= {MIN_TEST_CELLS_FOR_REPORTING} test cells): {test_acc:.3f}")

    labels_present = sorted(np.unique(np.concatenate([y_test_report, test_pred_report])))
    report_txt = classification_report(
        y_test_report, test_pred_report, labels=labels_present,
        target_names=[le_region_ct.classes_[i] for i in labels_present],
        zero_division=0
    )
    print(f"\n  TEST — M_region classification report ({ct}), reportable regions only:")
    print(report_txt)
    with open(os.path.join(ct_dir, "classification_report_test.txt"), "w") as f:
        f.write(report_txt)

    # ROC (only meaningful with >=2 classes present in test)
    macro_auc_ct = np.nan
    # predict_proba columns follow final_pipeline.classes_ order, not necessarily 0..n-1
    class_order = final_pipeline.named_steps["svc"].classes_
    prob_aligned = np.zeros((len(y_test_report), n_regions_ct))
    for col_pos, cls_label in enumerate(class_order):
        prob_aligned[:, cls_label] = test_prob_report[:, col_pos]

    if len(np.unique(y_test_report)) >= 2:
        auc_dict, macro_auc_ct = plot_roc_multiclass(
            y_test_report, prob_aligned, le_region_ct.classes_,
            f"ROC — M_region within '{ct}'  (test, SVM, reportable regions)",
            save_path=os.path.join(ct_dir, "roc_test.png")
        )
    else:
        print("  Only one region class present in reportable test split — skipping ROC.")

    # Confusion matrix — image + BOTH raw-count and normalized CSV
    region_labels_present = [le_region_ct.classes_[i] for i in labels_present]
    cm      = confusion_matrix(y_test_report, test_pred_report, labels=labels_present)
    cm_norm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

    fig, ax = plt.subplots(figsize=(1.2 * n_regions_ct + 3, 1.0 * n_regions_ct + 3))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=region_labels_present,
        yticklabels=region_labels_present,
        linewidths=0.5, linecolor="white", ax=ax
    )
    ax.set_title(f"M_region confusion matrix — '{ct}' (test, SVM)")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(ct_dir, "confusion_matrix_test.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    cm_df      = pd.DataFrame(cm, index=region_labels_present, columns=region_labels_present)
    cm_norm_df = pd.DataFrame(cm_norm, index=region_labels_present, columns=region_labels_present)
    cm_df.to_csv(os.path.join(ct_dir, "confusion_matrix_test_counts.csv"))
    cm_norm_df.to_csv(os.path.join(ct_dir, "confusion_matrix_test_normalized.csv"))
    print(f"  Saved: confusion_matrix_test.png, confusion_matrix_test_counts.csv, "
          f"confusion_matrix_test_normalized.csv")
    all_celltype_cm_norm[ct] = cm_norm_df

    # ---- Save model + label encoder for this celltype ----
    with open(os.path.join(ct_dir, "svm_pipeline.pkl"), "wb") as f:
        pickle.dump(final_pipeline, f)
    with open(os.path.join(ct_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le_region_ct, f)
    with open(os.path.join(ct_dir, "best_params.pkl"), "wb") as f:
        pickle.dump(best_params, f)

    # ══════════════════════════════════════════════════════════════════════
    # Hierarchical clustering of THIS celltype's regions by confusion pattern
    # (permutation-importance testing on confusable clusters removed per
    # request — clustering itself is kept)
    # ══════════════════════════════════════════════════════════════════════
    Z_ct, distance_df_ct = cluster_regions_by_confusion(cm_norm_df.values, region_labels_present, ct_dir, ct)
    cluster_gene_summaries = []

    if Z_ct is not None:
        max_dist = Z_ct[:, 2].max()
        cluster_ids = fcluster(Z_ct, t=CONFUSION_DISTANCE_THRESHOLD, criterion="distance")
        clusters = {}
        for region_name, cid in zip(region_labels_present, cluster_ids):
            clusters.setdefault(cid, []).append(region_name)
        confusable_clusters = {cid: regs for cid, regs in clusters.items() if len(regs) >= 2}

        print(f"\n  Confusion-based clusters (cut at absolute distance "
              f"{CONFUSION_DISTANCE_THRESHOLD}, this celltype's max distance was {max_dist:.3f}):")
        for cid, regs in clusters.items():
            tag = " <-- confusable group" if len(regs) >= 2 else ""
            print(f"    Cluster {cid}: {regs}{tag}")

        cluster_gene_summaries = [{"cluster_regions": regs} for regs in confusable_clusters.values()]
    else:
        print("  Fewer than 2 regions — skipping confusion clustering for this celltype.")

    celltype_summary[ct] = {
        "status":              "trained",
        "best_params":         best_params,
        "cv_score":            grid.best_score_,
        "n_train_total":       n_train_ct,
        "n_train_used_for_fit": len(ct_idx_train_fit),
        "subsampled":          n_train_ct > MAX_SVM_TRAIN_CELLS,
        "n_val":               len(ct_idx_val),
        "n_test":              len(ct_idx_test),
        "n_regions_used":      n_regions_ct,
        "n_regions_reportable": len(reportable_regions),
        "regions_used":        valid_regions,
        "regions_excluded_low_support": excluded_low_support,
        "val_accuracy":        val_acc,
        "test_accuracy":       test_acc,
        "test_accuracy_all_regions": test_acc_all_regions,
        "test_macro_auc":      macro_auc_ct,
        "confusable_clusters": cluster_gene_summaries,
    }

    del grid, final_pipeline, search_pipeline
    gc.collect()

# ══════════════════════════════════════════════════════════════════════════════
# 13. SUMMARY ACROSS CELL TYPES
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 13: Summary across cell types")
print("=" * 60)

summary_rows = []
for ct, info in celltype_summary.items():
    if info.get("status") == "trained":
        summary_rows.append({
            "subclass":        ct,
            "best_params":     info["best_params"],
            "cv_score":        info["cv_score"],
            "n_train_total":   info["n_train_total"],
            "n_train_used":    info["n_train_used_for_fit"],
            "subsampled":      info["subsampled"],
            "n_val":           info["n_val"],
            "n_test":          info["n_test"],
            "n_regions":       info["n_regions_used"],
            "n_regions_reportable": info["n_regions_reportable"],
            "regions_excluded_low_support": ", ".join(info["regions_excluded_low_support"]) or None,
            "val_accuracy":    info["val_accuracy"],
            "test_accuracy":   info["test_accuracy"],
            "test_accuracy_all_regions": info["test_accuracy_all_regions"],
            "test_macro_auc":  info["test_macro_auc"],
            "n_confusable_clusters": len(info["confusable_clusters"]),
        })
    else:
        summary_rows.append({
            "subclass": ct, "best_params": None, "cv_score": None,
            "n_train_total": None, "n_train_used": None, "subsampled": None,
            "n_val": None, "n_test": None, "n_regions": None,
            "n_regions_reportable": None, "regions_excluded_low_support": None,
            "val_accuracy": None, "test_accuracy": None, "test_accuracy_all_regions": None,
            "test_macro_auc": None, "n_confusable_clusters": None,
        })

summary_df = pd.DataFrame(summary_rows).sort_values(
    "test_accuracy", ascending=False, na_position="last"
)
print(summary_df.to_string(index=False))

summary_csv_path = os.path.join(SAVE_DIR, "per_celltype_summary.csv")
summary_df.to_csv(summary_csv_path, index=False)

with open(os.path.join(SAVE_DIR, "per_celltype_summary.pkl"), "wb") as f:
    pickle.dump(celltype_summary, f)

with open(os.path.join(SAVE_DIR, "combat_batch_params.pkl"), "wb") as f:
    pickle.dump(batch_params, f)
with open(os.path.join(SAVE_DIR, "verified_genes.txt"), "w") as f:
    for g in verified_genes:
        f.write(g + "\n")

# ══════════════════════════════════════════════════════════════════════════════
# 14. AGGREGATE CONFUSION-BASED CLUSTERING ACROSS ALL CELLTYPES
#     Each celltype's SVM sees a different training set and can highlight
#     different confusable pairs by chance. This step checks the result
#     ACROSS all 6 celltypes at once: for every pair of regions, average the
#     symmetrized confusion rate over every celltype that includes BOTH
#     regions (celltypes missing one of the two, e.g. TT/DP dropped from
#     004/006, are simply skipped for that specific pair -- not treated as
#     zero confusion). A pair that clusters tightly here despite being
#     computed from up to 6 independently-trained models is a much more
#     robust signal than any single celltype's dendrogram alone.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 14: Aggregate confusion-based clustering across all celltypes")
print("=" * 60)

if len(all_celltype_cm_norm) < 2:
    print("Fewer than 2 trained celltypes — skipping aggregate clustering.")
else:
    all_regions_union = sorted(set().union(*[set(df.index) for df in all_celltype_cm_norm.values()]))
    n_all = len(all_regions_union)
    print(f"Union of regions across all {len(all_celltype_cm_norm)} trained celltypes: "
          f"{n_all} -> {all_regions_union}")

    sim_sum   = pd.DataFrame(0.0, index=all_regions_union, columns=all_regions_union)
    sim_count = pd.DataFrame(0,   index=all_regions_union, columns=all_regions_union)

    for ct, cm_norm_df in all_celltype_cm_norm.items():
        regs_ct = cm_norm_df.index.tolist()
        sym = (cm_norm_df.values + cm_norm_df.values.T) / 2
        sym_df = pd.DataFrame(sym, index=regs_ct, columns=regs_ct)
        for ri in regs_ct:
            for rj in regs_ct:
                sim_sum.loc[ri, rj]   += sym_df.loc[ri, rj]
                sim_count.loc[ri, rj] += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        avg_similarity = sim_sum / sim_count
    avg_similarity_arr = avg_similarity.values.copy()
    np.fill_diagonal(avg_similarity_arr, 1.0)
    avg_similarity = pd.DataFrame(avg_similarity_arr, index=all_regions_union, columns=all_regions_union)

    # Pairs with zero celltype overlap (no shared celltype includes both
    # regions) have no defined similarity -- can't be clustered meaningfully,
    # so they're treated as maximally distant rather than silently zero.
    offdiag_mask = ~np.eye(n_all, dtype=bool)
    n_undefined = int((sim_count.values[offdiag_mask] == 0).sum())
    if n_undefined > 0:
        print(f"NOTE: {n_undefined} region pair(s) share NO common celltype and have no "
              f"defined confusion rate -- set to maximum distance (least similar).")
    avg_similarity = avg_similarity.fillna(0.0)

    distance = 1 - avg_similarity.values
    np.fill_diagonal(distance, 0.0)
    distance = (distance + distance.T) / 2  # enforce exact symmetry

    distance_df = pd.DataFrame(distance, index=all_regions_union, columns=all_regions_union)
    distance_df.to_csv(os.path.join(SAVE_DIR, "aggregate_region_distance.csv"))
    sim_count.to_csv(os.path.join(SAVE_DIR, "aggregate_region_pair_support_count.csv"))

    condensed = squareform(distance, checks=False)
    Z_agg = linkage(condensed, method="average")

    fig, ax = plt.subplots(figsize=(max(10, 0.5 * n_all + 3), 6.5))
    dendrogram(Z_agg, labels=all_regions_union, ax=ax, leaf_rotation=90, leaf_font_size=9)
    ax.set_title(f"Regions clustered by confusion pattern, averaged across "
                 f"{len(all_celltype_cm_norm)} celltypes")
    ax.set_ylabel("Distance (1 \u2212 avg. symmetric confusion rate across celltypes)")
    plt.tight_layout()
    agg_dendro_path = os.path.join(SAVE_DIR, "aggregate_region_confusion_dendrogram.png")
    plt.savefig(agg_dendro_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Report the single most-confusable pair and most-distinct region, using
    # only pairs with real (non-zero-support) data behind them.
    off_diag = distance_df.values.copy()
    np.fill_diagonal(off_diag, np.inf)
    off_diag[sim_count.values == 0] = np.inf  # exclude undefined pairs from "most confusable"
    if np.isfinite(off_diag).any():
        min_idx = np.unravel_index(np.argmin(off_diag), off_diag.shape)
        print(f"\nMost confusable pair (aggregated across celltypes): "
              f"{all_regions_union[min_idx[0]]} <-> {all_regions_union[min_idx[1]]}  "
              f"(distance={off_diag[min_idx]:.3f}, "
              f"support={int(sim_count.values[min_idx])} celltypes)")

    print(f"\nSaved: aggregate_region_distance.csv, aggregate_region_pair_support_count.csv, "
          f"{os.path.basename(agg_dendro_path)}")

    # ── Aggregate spectral clustering — same idea as the per-celltype
    #    spectral clustering, just run on the AGGREGATE similarity matrix
    #    instead of one celltype's. Independent second method alongside the
    #    aggregate dendrogram above: hierarchical clustering forces a single
    #    nested tree even at the aggregate level, so a real-but-moderate
    #    cross-celltype relationship could still be buried the same way it
    #    can be within a single celltype's dendrogram.
    print("\nAggregate spectral clustering (across all celltypes' averaged similarity):")
    if n_all >= 4:
        k_range = range(2, min(10, n_all))
        best_k, best_score, best_labels = None, -2, None
        for k in k_range:
            try:
                sc = SpectralClustering(
                    n_clusters=k, affinity="precomputed", random_state=42,
                    assign_labels="kmeans",
                )
                labels = sc.fit_predict(avg_similarity.values)
                if len(set(labels)) < 2:
                    continue
                score = silhouette_score(distance, labels, metric="precomputed")
                if score > best_score:
                    best_k, best_score, best_labels = k, score, labels
            except Exception as e:
                print(f"  Spectral clustering failed at k={k}: {e}")

        if best_labels is not None:
            agg_spectral_df = pd.DataFrame({
                "region": all_regions_union, "spectral_cluster": best_labels,
            }).sort_values("spectral_cluster")
            agg_spectral_path = os.path.join(SAVE_DIR, "aggregate_spectral_clustering.csv")
            agg_spectral_df.to_csv(agg_spectral_path, index=False)

            print(f"  Best k={best_k} (silhouette={best_score:.3f})")
            agg_spectral_clusters = agg_spectral_df.groupby("spectral_cluster")["region"].apply(list).to_dict()
            for cid, regs in agg_spectral_clusters.items():
                if len(regs) >= 2:
                    print(f"    Aggregate spectral cluster {cid}: {regs}")
            print(f"  Saved: {os.path.basename(agg_spectral_path)}")

            # Reorder + plot the aggregate similarity matrix by spectral
            # cluster, for direct visual comparison against the dendrogram.
            spec_order = np.argsort(best_labels)
            spec_names = [all_regions_union[i] for i in spec_order]
            sim_spec_ordered = avg_similarity.values[np.ix_(spec_order, spec_order)]
            fig, ax = plt.subplots(figsize=(1.0 * n_all + 3, 0.9 * n_all + 3))
            sns.heatmap(
                sim_spec_ordered, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=spec_names, yticklabels=spec_names,
                linewidths=0.5, linecolor="white", ax=ax
            )
            ax.set_title(f"Aggregate region similarity reordered by spectral cluster (k={best_k})\n"
                         f"averaged across {len(all_celltype_cm_norm)} celltypes")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            plt.tight_layout()
            agg_spec_heatmap_path = os.path.join(SAVE_DIR, "aggregate_similarity_spectral_ordered.png")
            plt.savefig(agg_spec_heatmap_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {os.path.basename(agg_spec_heatmap_path)}")
        else:
            print("  No valid spectral clustering found across the k range tried.")
    else:
        print(f"  Skipping: only {n_all} regions in the union (need >= 4).")

    # Same reordering diagnostic as the per-celltype confusion matrices:
    # the aggregate similarity matrix reordered by the DENDROGRAM's own leaf
    # order, so real block structure (or its absence) is directly visible
    # rather than only inferred from the tree shape.
    leaf_order_agg = leaves_list(Z_agg)
    dendro_names_agg = [all_regions_union[i] for i in leaf_order_agg]
    sim_dendro_ordered = avg_similarity.values[np.ix_(leaf_order_agg, leaf_order_agg)]
    fig, ax = plt.subplots(figsize=(1.0 * n_all + 3, 0.9 * n_all + 3))
    sns.heatmap(
        sim_dendro_ordered, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=dendro_names_agg, yticklabels=dendro_names_agg,
        linewidths=0.5, linecolor="white", ax=ax
    )
    ax.set_title(f"Aggregate region similarity reordered by dendrogram leaf order\n"
                 f"averaged across {len(all_celltype_cm_norm)} celltypes")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    agg_dendro_heatmap_path = os.path.join(SAVE_DIR, "aggregate_similarity_dendrogram_ordered.png")
    plt.savefig(agg_dendro_heatmap_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(agg_dendro_heatmap_path)}")

print(f"\n{'─'*60}")
print("ALL DONE — per-celltype SVM M_region classifiers trained and evaluated")
print(f"{'─'*60}")
print(f"Output folder      : {SAVE_DIR}")
print(f"Per-celltype models: {CELLTYPE_SAVE_ROOT}")
print(f"Summary CSV        : {summary_csv_path}")
print(f"\nCelltypes trained: "
      f"{sum(1 for v in celltype_summary.values() if v.get('status') == 'trained')} "
      f"/ {len(qualifying_celltypes)}")