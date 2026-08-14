import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import os
import gc
import pickle
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc
)
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from matplotlib.patches import Patch
from scipy import stats

# ══════════════════════════════════════════════════════════════════════════════
# 0. PATHS AND DEVICE
# ══════════════════════════════════════════════════════════════════════════════
DATA_DIR = r"U:\Research\microarray"
DATA     = r"C:\Users\StujenskeLab\Documents\NAN151_workspace\microarray"

fpath1    = os.path.join(DATA_DIR, 'merfish_F1_raw.csv')
fpath2    = os.path.join(DATA_DIR, 'merfish_F2_raw.csv')
fpath3    = os.path.join(DATA_DIR, 'merfish_M1_raw.csv')
fpath4    = os.path.join(DATA_DIR,     'merfish_M2_Glut_raw.csv')
genes_csv = os.path.join(DATA,     'common_genes_4datasets.csv')

# UPDATED — points at the corrected, division-filtered human file
# (Hippocampus/Basal forebrain fixes) instead of the old Human_filtered.csv
human_filtered_path = os.path.join(DATA_DIR, 'Human_N_filtered.csv')
if not os.path.exists(human_filtered_path):
    human_filtered_path = os.path.join(DATA, 'Human_N_filtered.csv')
if not os.path.exists(human_filtered_path):
    raise FileNotFoundError("Human_N_filtered.csv not found.")
print(f"Human_filtered: {human_filtered_path}")
print("NOTE (v4): BATCH CORRECTION SKIPPED — no ComBat. Mouse ROI cells are")
print("      loaded directly (raw counts), log1p-transformed, and that's it.")
print("      Reintroduce ComBat later if per-animal batch effects turn out")
print("      to matter — everything downstream only depends on")
print("      X_encoder_input having the right shape, so it's a clean add-back.")
print("      No contrastive loss / no bridge — encoder trains directly on")
print("      the shared gene space (no PCA bottleneck).")
print("      Overfitting fixes applied: lower LR, higher weight decay/dropout,")
print("      smaller hidden dim (512→256) to reduce model capacity.")
print("      Mouse classification restricted to ISOCORTEX ONLY —")
print("      16 structure_regions (FRP/ACAd/ACAv/PL/ILA/ORBl/ORBm/ORBvl/")
print("      MOs/MOp/RSPd/RSPv/RSPagl/AId/AIv/AIp). Hippocampus, striatum,")
print("      hypothalamus, and extended amygdala are no longer classified —")
print("      those mouse cells are simply excluded from the ROI (region=None).")
print("      Human side unchanged (10 curated cortex regions from")
print("      Human_N_filtered.csv, plus whatever subcortical h_region labels")
print("      remain in that file — those just never get an 'expected match'")
print("      row and act as off-diagonal negative controls in the evaluation.)")

SAVE_DIR = os.path.join(DATA, "cross_species_v4_no_batch_correction")
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
    "parcellation_substructure", "subclass", "class",
    "x", "y", "z", "cell_id",
    "brain_section_label", "cluster", "neurotransmitter",
    "CCF_level1", "CCF_level2", "acronym", "hrc_mmc_subclass_name",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ══════════════════════════════════════════════════════════════════════════════
# 1. IDENTIFY SHARED GENES
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 1: Shared genes (mouse ∩ human)")
print("=" * 60)

mouse_panel = pd.read_csv(genes_csv).iloc[:, 0].tolist()
human_hdr   = pd.read_csv(human_filtered_path, index_col=0, nrows=0)
# UPDATED — exclusion list matches Human_N_filtered.csv's actual metadata
# columns (region_of_interest_label, anatomical_division_label, h_region).
# "supercluster" kept in the exclusion list too since it's harmless if
# absent and protective if some later version of the file adds it back.
hcols       = [c for c in human_hdr.columns
               if c not in ["region_of_interest_label",
                             "anatomical_division_label",
                             "supercluster", "h_region"]]
shared_genes = [g for g in mouse_panel if g in hcols]
not_in_human = [g for g in mouse_panel if g not in hcols]
print(f"Mouse panel    : {len(mouse_panel)}")
print(f"Shared genes   : {len(shared_genes)}")
print(f"Excluded       : {len(not_in_human)}  → {not_in_human}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. DETERMINE COLUMNS TO LOAD PER FILE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: Determining columns to load per file")
print("=" * 60)

def get_load_info(fpath, shared_genes, meta_needed):
    hdr       = pd.read_csv(fpath, nrows=0)
    all_cols  = hdr.columns.tolist()
    idx_col   = all_cols[0]
    gene_cols = [g for g in shared_genes if g in all_cols]
    mcols     = [c for c in meta_needed   if c in all_cols]
    keep      = [idx_col] + gene_cols + mcols
    print(f"  {os.path.basename(fpath):<46}: "
          f"{len(gene_cols)} genes + {len(mcols)} meta cols")
    return idx_col, keep, gene_cols

cols_info = {}
for fpath, animal_id, _ in animal_paths:
    cols_info[animal_id] = get_load_info(fpath, shared_genes, meta_cols_needed)

verified_shared = [
    g for g in shared_genes
    if all(g in cols_info[aid][2] for _, aid, _ in animal_paths)
]
print(f"\nVerified shared genes (all files): {len(verified_shared)}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. REGION LABEL ASSIGNMENT FUNCTION
#    UPDATED — isocortex-only classification (16 structure_regions).
#    Hippocampus/striatum/hypothalamus/extended-amygdala cells are no
#    longer assigned any M_region and are excluded from the ROI.
# ══════════════════════════════════════════════════════════════════════════════
hippocampus_labels = ["HIP", "HPF"]   # kept for reference only — NOT used below
striatum_labels    = ["STR", "STRv", "STRd"]   # kept for reference only — NOT used below
division_regions   = ["HY", "sAMY"]   # kept for reference only — NOT used below
structure_regions  = [
    "FRP", "ACAd", "ACAv", "PL", "ILA", "ORBl", "ORBm", "ORBvl",
    "MOs", "MOp", "RSPd", "RSPv", "RSPagl", "AId", "AIv", "AIp",
]
region_rename      = {
    "FRP":"M_FRP","ACAd":"M_ACAd","ACAv":"M_ACAv","PL":"M_PL","ILA":"M_ILA",
    "ORBl":"M_ORBl","ORBm":"M_ORBm","ORBvl":"M_ORBvl",
    "MOs":"M_MOs","MOp":"M_MOp",
    "RSPd":"M_RSPd","RSPv":"M_RSPv","RSPagl":"M_RSPagl",
    "AId":"M_AId","AIv":"M_AIv","AIp":"M_AIp",
}

# UPDATED — isocortex-only classification. Cells in hippocampus, striatum,
# hypothalamus, or extended amygdala are no longer assigned any M_region
# at all (region stays None -> excluded from ROI, same as any other
# out-of-scope area). Only the 16 isocortex structure_regions are matched.
def assign_region_vectorized(div_arr, struct_arr):
    region = np.full(len(div_arr), None, dtype=object)
    for s in structure_regions:
        region[struct_arr == s] = s
    return region

# ══════════════════════════════════════════════════════════════════════════════
# 4. LOAD META → FILTER ROI → LOAD GENE MATRIX FOR ROI ONLY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4: Load meta → filter ROI → load ROI genes only")
print("=" * 60)

META_CHUNK = 200000
roi_pieces = []

for fpath, animal_id, sex in animal_paths:
    idx_col, keep_cols, gene_cols_in_file = cols_info[animal_id]

    meta_only_cols = [idx_col] + [
        c for c in keep_cols
        if c != idx_col and c not in verified_shared
    ]

    print(f"\n  {animal_id}: scanning meta in chunks "
          f"({len(meta_only_cols)} cols, chunk={META_CHUNK:,})...")

    roi_index_labels = []
    meta_roi_chunks  = []
    total_rows_seen  = 0

    reader = pd.read_csv(
        fpath, usecols=meta_only_cols, index_col=idx_col,
        low_memory=False, chunksize=META_CHUNK
    )

    for chunk_meta in reader:
        chunk_meta.rename(columns=rename_map, inplace=True)
        chunk_meta["animal_id"] = animal_id
        chunk_meta["sex"]       = sex

        if "class" not in chunk_meta.columns:
            chunk_meta["class"] = "unknown"

        div_arr    = chunk_meta.get(
            "parcellation_division",
            pd.Series([""] * len(chunk_meta), index=chunk_meta.index)
        ).fillna("").values
        struct_arr = chunk_meta.get(
            "parcellation_structure",
            pd.Series([""] * len(chunk_meta), index=chunk_meta.index)
        ).fillna("").values

        raw_labels = assign_region_vectorized(div_arr, struct_arr)
        m_labels   = np.where(
            raw_labels != None,
            np.vectorize(lambda r: region_rename.get(r, None))(raw_labels),
            None
        )
        chunk_meta["M_region"] = m_labels
        roi_mask = chunk_meta["M_region"].notna().values

        if roi_mask.sum() > 0:
            roi_chunk = chunk_meta.loc[roi_mask].copy()
            meta_roi_chunks.append(roi_chunk)
            roi_index_labels.extend(roi_chunk.index.tolist())

        total_rows_seen += len(chunk_meta)

    del reader; gc.collect()

    n_roi   = len(roi_index_labels)
    n_total = total_rows_seen
    print(f"  {animal_id}: {n_roi:,} ROI / {n_total:,} total ({n_roi/n_total:.1%})")

    if n_roi == 0:
        print(f"  WARNING: no ROI cells in {animal_id}, skipping")
        gc.collect(); continue

    meta_roi = pd.concat(meta_roi_chunks, axis=0)
    del meta_roi_chunks; gc.collect()

    roi_id_set = set(roi_index_labels)
    print(f"  {animal_id}: scanning row positions of {n_roi:,} ROI cells...")

    keep_row_numbers = set()
    row_num          = 0

    id_reader = pd.read_csv(
        fpath, usecols=[idx_col], index_col=idx_col,
        low_memory=False, chunksize=META_CHUNK
    )
    for id_chunk in id_reader:
        for cell_id in id_chunk.index:
            row_num += 1
            if cell_id in roi_id_set:
                keep_row_numbers.add(row_num)
    del id_reader; gc.collect()

    total_data_rows = row_num
    skip_rows = [i for i in range(1, total_data_rows + 1)
                 if i not in keep_row_numbers]

    gene_load_cols = [idx_col] + [g for g in verified_shared if g in gene_cols_in_file]
    mem_est        = n_roi * len(verified_shared) * 4 / 1e9
    print(f"  {animal_id}: loading {n_roi:,} ROI gene rows (~{mem_est:.2f} GB)...")

    df_genes = pd.read_csv(
        fpath, usecols=gene_load_cols, index_col=idx_col,
        skiprows=skip_rows, low_memory=False,
        dtype={g: np.float32 for g in verified_shared if g in gene_load_cols}
    )

    common_idx = meta_roi.index.intersection(df_genes.index)
    if len(common_idx) < n_roi:
        print(f"  {animal_id}: WARNING — {n_roi - len(common_idx)} cells lost")

    meta_roi = meta_roi.loc[common_idx]
    X_roi    = df_genes.loc[common_idx, verified_shared].values.astype(np.float32)
    del df_genes; gc.collect()

    print(f"  {animal_id}: X_roi={X_roi.shape}  meta={meta_roi.shape}")
    print(f"    Raw count range: {X_roi.min():.1f} – {X_roi.max():.1f}")
    roi_pieces.append((X_roi, meta_roi))
    gc.collect()

# ══════════════════════════════════════════════════════════════════════════════
# 5. CONCATENATE ROI DATA
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5: Concatenating ROI data (RAW counts)")
print("=" * 60)

X_all   = np.vstack([X for X, _ in roi_pieces]).astype(np.float32)
obs_all = pd.concat([m for _, m in roi_pieces], axis=0)
obs_all.reset_index(drop=True, inplace=True)
del roi_pieces; gc.collect()

subclass_col = next((c for c in ["subclass", "supercluster",
                                   "parcellation_substructure", "cluster"]
                     if c in obs_all.columns), None)

print(f"X_all shape     : {X_all.shape}")
print(f"X_all range     : {X_all.min():.1f} – {X_all.max():.1f}  (RAW counts)")
print(f"Subclass column : '{subclass_col}'")
print(f"\nM_region counts:")
print(obs_all["M_region"].value_counts().to_string())
print(f"\nCells per M_region per animal:")
print(obs_all.groupby(
    ["M_region","animal_id"]).size().unstack(fill_value=0).to_string())

# ══════════════════════════════════════════════════════════════════════════════
# 6. LOG1P TRANSFORM — batch correction SKIPPED for now (per instruction).
#    This is simply log1p on the raw ROI counts, nothing else. No ComBat
#    fit, no per-animal correction. If batch effects turn out to matter,
#    re-add ComBat here later — the rest of the pipeline downstream
#    (classifier, human inference, similarity matrix) is unaffected either
#    way, since it only cares about X_encoder_input having the right shape.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 6: Log1p-transforming raw counts (NO ComBat — batch correction skipped)")
print("=" * 60)

X_all_log = np.log1p(X_all).astype(np.float32)
print(f"X_all_log range: {X_all_log.min():.3f} – {X_all_log.max():.3f}")

X_roi_log = X_all_log
X_encoder_input = X_roi_log.copy()

# ══════════════════════════════════════════════════════════════════════════════
# 7. TRAIN / VAL / TEST SPLIT — 70/10/20
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 7: Train / Val / Test split — 70/10/20")
print("=" * 60)

le_region_pre = LabelEncoder()
y_region_pre  = le_region_pre.fit_transform(obs_all["M_region"].values)

n_total = len(X_all)
idx_trainval, idx_test = train_test_split(
    np.arange(n_total), test_size=0.20,
    stratify=y_region_pre, random_state=42
)
idx_train, idx_val = train_test_split(
    idx_trainval, test_size=0.125,   # 0.125 of remaining 80% = 10% of total
    stratify=y_region_pre[idx_trainval], random_state=42
)

print(f"Total : {n_total:,}")
print(f"Train : {len(idx_train):,}  ({len(idx_train)/n_total:.0%})")
print(f"Val   : {len(idx_val):,}   ({len(idx_val)/n_total:.0%})")
print(f"Test  : {len(idx_test):,}   ({len(idx_test)/n_total:.0%})")

# ══════════════════════════════════════════════════════════════════════════════
# 9. BUILD ANNDATA FOR MOUSE QC UMAP
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 9: Building AnnData (ROI cells, log-transformed only)")
print("=" * 60)

obs_final       = obs_all.copy()
obs_final.index = obs_final.index.astype(str)

adata = ad.AnnData(X=X_roi_log.copy(), obs=obs_final)
adata.var_names                       = verified_shared
adata.layers["log_transformed"]       = X_roi_log

print("Running PCA + UMAP for QC visualization...")
sc.pp.scale(adata, max_value=10)
sc.pp.pca(adata, n_comps=50)
sc.pp.neighbors(adata, n_pcs=30)
sc.tl.umap(adata)

sc.pl.umap(
    adata,
    color=["M_region", "animal_id", "sex", subclass_col],
    ncols=2,
    title=["M_region (no batch correction)", "Animal ID",
           "Sex", "Cell subtype"],
    show=True
)
plt.savefig(os.path.join(SAVE_DIR, "umap_qc_mouse.png"),
            dpi=150, bbox_inches="tight")

# ══════════════════════════════════════════════════════════════════════════════
# 10. ENCODE LABELS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 10: Encoding labels")
print("=" * 60)

mouse_regions  = adata.obs["M_region"].values
mouse_subtypes = adata.obs[subclass_col].values

le_region  = LabelEncoder()
le_subtype = LabelEncoder()
y_region   = le_region.fit_transform(mouse_regions)
y_subtype  = le_subtype.fit_transform(mouse_subtypes)

n_regions  = len(le_region.classes_)
n_subtypes = len(le_subtype.classes_)

print(f"M_region classes ({n_regions}):")
for cls, n in zip(le_region.classes_, np.bincount(y_region)):
    print(f"  {cls:<12}: {n:,}")
print(f"\nSubtype classes  ({n_subtypes})")
print(f"Encoder input dim: {X_encoder_input.shape[1]}  "
      f"({len(verified_shared)} genes, log1p only, no ComBat)")

splits = {"train": idx_train, "val": idx_val, "test": idx_test}
print("\nM_region distribution per split:")
for sn, idx in splits.items():
    counts = pd.Series(
        le_region.inverse_transform(y_region[idx])
    ).value_counts()
    print(f"\n  {sn}:")
    for r, n in counts.items():
        print(f"    {r:<12}: {n:,}")

# ══════════════════════════════════════════════════════════════════════════════
# 11. RESIDUAL ENCODER + DUAL-HEAD MODEL
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 11: Residual Encoder + Dual-Head model (reduced capacity)")
print("=" * 60)

HIDDEN_DIM = 256

class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim),
        )
    def forward(self, x):
        return F.relu(self.net(x) + x)

class ResidualEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=512, hidden_dim=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
            nn.ReLU(), nn.Dropout(0.5),
        )
        self.res1 = ResBlock(hidden_dim, dropout=0.4)
        self.res2 = ResBlock(hidden_dim, dropout=0.4)
        self.out  = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim), nn.BatchNorm1d(latent_dim), nn.ReLU()
        )
    def forward(self, x):
        x = self.proj(x)
        x = self.res1(x)
        x = self.res2(x)
        return self.out(x)

class ResidualDualHeadClassifier(nn.Module):
    def __init__(self, input_dim, n_regions, n_subtypes, latent_dim=512, hidden_dim=256):
        super().__init__()
        self.encoder      = ResidualEncoder(input_dim, latent_dim, hidden_dim)
        self.head_region  = nn.Linear(latent_dim, n_regions)
        self.head_subtype = nn.Linear(latent_dim, n_subtypes)

    def forward(self, x):
        z = self.encoder(x)
        return self.head_region(z), self.head_subtype(z), z

    def encode(self, x):
        with torch.no_grad():
            return self.encoder(x)

input_dim  = X_encoder_input.shape[1]
latent_dim = 512
model      = ResidualDualHeadClassifier(
    input_dim, n_regions, n_subtypes, latent_dim, HIDDEN_DIM
).to(device)

total_p = sum(p.numel() for p in model.parameters())
print(f"Architecture     : Residual Encoder (reduced capacity)")
print(f"Input dim        : {input_dim}  ({len(verified_shared)} log1p genes, no ComBat)")
print(f"Hidden dim       : {HIDDEN_DIM}  (reduced from 512)")
print(f"Latent dim       : {latent_dim}")
print(f"Dropout          : 0.4 (residual blocks), 0.5 (input projection)")
print(f"M_region classes : {n_regions}")
print(f"Subtype classes  : {n_subtypes}")
print(f"Total parameters : {total_p:,}")
print(model)

# ══════════════════════════════════════════════════════════════════════════════
# 12. TRAINING SETUP
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 12: Training setup (lower LR, higher weight decay)")
print("=" * 60)

BATCH_SIZE = 1024
N_EPOCHS   = 200
LR         = 3e-4
LAMBDA_R   = 1.0
LAMBDA_S   = 0.5
PATIENCE   = 30
MIN_DELTA  = 1e-4

def make_loader(idx, shuffle=True):
    X  = torch.tensor(X_encoder_input[idx])
    yr = torch.tensor(y_region[idx],  dtype=torch.long)
    ys = torch.tensor(y_subtype[idx], dtype=torch.long)
    return DataLoader(TensorDataset(X, yr, ys),
                      batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=0, pin_memory=(device.type == "cuda"))

train_loader = make_loader(idx_train, shuffle=True)
val_loader   = make_loader(idx_val,   shuffle=False)
test_loader  = make_loader(idx_test,  shuffle=False)

def make_weights(y, n_cls):
    counts  = np.bincount(y)
    weights = 1.0 / (counts + 1e-6)
    return torch.tensor(
        weights / weights.sum() * n_cls, dtype=torch.float32
    ).to(device)

criterion_r = nn.CrossEntropyLoss(weight=make_weights(y_region,  n_regions))
criterion_s = nn.CrossEntropyLoss(weight=make_weights(y_subtype, n_subtypes))
optimizer   = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
scheduler   = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=12, verbose=True
)

print(f"Batch size    : {BATCH_SIZE}")
print(f"Max epochs    : {N_EPOCHS}")
print(f"LR            : {LR}")
print(f"Weight decay  : 1e-3")
print(f"λ_region      : {LAMBDA_R}")
print(f"λ_subtype     : {LAMBDA_S}")
print(f"Early stopping: patience={PATIENCE}, min_delta={MIN_DELTA}")
print(f"LR scheduler  : ReduceLROnPlateau patience=12, factor=0.5")

# ══════════════════════════════════════════════════════════════════════════════
# 13. TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 13: Training")
print("=" * 60)

history = {k: [] for k in [
    "train_loss","train_acc_r","train_acc_s",
    "val_loss","val_acc_r","val_acc_s"
]}
best_val_loss  = float("inf")
patience_count = 0
best_state     = None
best_epoch     = 0

for epoch in range(N_EPOCHS):
    model.train()
    t_loss = t_cr = t_cs = t_n = 0
    for X_b, yr_b, ys_b in train_loader:
        X_b  = X_b.to(device); yr_b = yr_b.to(device); ys_b = ys_b.to(device)
        optimizer.zero_grad()
        out_r, out_s, _ = model(X_b)
        loss = LAMBDA_R * criterion_r(out_r, yr_b) + \
               LAMBDA_S * criterion_s(out_s, ys_b)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        t_loss += loss.item()*len(X_b)
        t_cr   += (out_r.argmax(1)==yr_b).sum().item()
        t_cs   += (out_s.argmax(1)==ys_b).sum().item()
        t_n    += len(X_b)
    t_loss /= t_n; t_ar = t_cr/t_n; t_as = t_cs/t_n

    model.eval()
    v_loss = v_cr = v_cs = v_n = 0
    with torch.no_grad():
        for X_b, yr_b, ys_b in val_loader:
            X_b  = X_b.to(device); yr_b = yr_b.to(device); ys_b = ys_b.to(device)
            out_r, out_s, _ = model(X_b)
            loss = LAMBDA_R * criterion_r(out_r, yr_b) + \
                   LAMBDA_S * criterion_s(out_s, ys_b)
            v_loss += loss.item()*len(X_b)
            v_cr   += (out_r.argmax(1)==yr_b).sum().item()
            v_cs   += (out_s.argmax(1)==ys_b).sum().item()
            v_n    += len(X_b)
    v_loss /= v_n; v_ar = v_cr/v_n; v_as = v_cs/v_n

    scheduler.step(v_loss)
    for k, v in zip(history.keys(),
                    [t_loss,t_ar,t_as,v_loss,v_ar,v_as]):
        history[k].append(v)

    if (epoch+1) % 5 == 0 or epoch == 0:
        print(f"Ep {epoch+1:>3}/{N_EPOCHS} | "
              f"Tr loss:{t_loss:.4f} acc_r:{t_ar:.3f} acc_s:{t_as:.3f} | "
              f"Va loss:{v_loss:.4f} acc_r:{v_ar:.3f} acc_s:{v_as:.3f}")

    if v_loss < best_val_loss - MIN_DELTA:
        best_val_loss  = v_loss; patience_count = 0; best_epoch = epoch + 1
        best_state = {k: v.cpu().clone()
                      for k, v in model.state_dict().items()}
    else:
        patience_count += 1
        if patience_count >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1} "
                  f"(best epoch was {best_epoch}, "
                  f"{epoch+1-best_epoch} epochs since improvement)")
            break

model.load_state_dict(best_state)
print(f"\nBest val loss: {best_val_loss:.4f}  (epoch {best_epoch})")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
eps = range(1, len(history["train_loss"])+1)
for ax, tr_k, va_k, title in zip(
    axes,
    ["train_loss","train_acc_r","train_acc_s"],
    ["val_loss","val_acc_r","val_acc_s"],
    ["Loss","M_region accuracy","Subtype accuracy"]
):
    ax.plot(eps, history[tr_k], label="Train")
    ax.plot(eps, history[va_k], label="Val")
    ax.axvline(best_epoch, color="gray", linestyle=":", lw=1, label="Best epoch")
    ax.set_title(title); ax.set_xlabel("Epoch"); ax.legend()
plt.suptitle("Training curves — Residual encoder (12 regions, reduced capacity)",
             fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "training_curves.png"),
            dpi=150, bbox_inches="tight")
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# 14. COLLECT PREDICTIONS HELPER
# ══════════════════════════════════════════════════════════════════════════════
def collect_preds(idx):
    loader = make_loader(idx, shuffle=False)
    model.eval()
    tr, pr, pbr = [], [], []
    ts, ps, pbs = [], [], []
    with torch.no_grad():
        for X_b, yr_b, ys_b in loader:
            out_r, out_s, _ = model(X_b.to(device))
            p_r = torch.softmax(out_r, dim=1).detach().cpu().numpy()
            p_s = torch.softmax(out_s, dim=1).detach().cpu().numpy()
            tr.extend(yr_b.numpy())
            pr.extend(out_r.argmax(1).detach().cpu().numpy())
            ts.extend(ys_b.numpy())
            ps.extend(out_s.argmax(1).detach().cpu().numpy())
            pbr.append(p_r); pbs.append(p_s)
    return (np.array(tr), np.array(pr), np.vstack(pbr),
            np.array(ts), np.array(ps), np.vstack(pbs))

# ══════════════════════════════════════════════════════════════════════════════
# 15. CLASSIFICATION REPORTS — VAL + TEST
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 15: Classification reports — Val + Test")
print("=" * 60)

results = {}
for split_name, idx in [("val", idx_val), ("test", idx_test)]:
    tr, pr, pbr, ts, ps, pbs = collect_preds(idx)
    results[split_name] = dict(tr=tr, pr=pr, pbr=pbr, ts=ts, ps=ps, pbs=pbs)

    print(f"\n{'─'*55}")
    print(f"  {split_name.upper()}  —  M_region")
    print(f"{'─'*55}")
    region_labels_present = sorted(np.unique(np.concatenate([tr, pr])))
    print(classification_report(
        tr, pr, labels=region_labels_present,
        target_names=[le_region.classes_[i] for i in region_labels_present]
    ))

    print(f"\n  {split_name.upper()}  —  Subtype ({subclass_col})")
    print(f"{'─'*55}")
    subtype_labels_present = sorted(np.unique(np.concatenate([ts, ps])))
    n_present = len(subtype_labels_present)
    n_total_s = len(le_subtype.classes_)
    if n_present < n_total_s:
        print(f"  NOTE: {n_total_s - n_present} rare subtype(s) absent — "
              f"reporting {n_present} present")
    print(classification_report(
        ts, ps, labels=subtype_labels_present,
        target_names=[le_subtype.classes_[i] for i in subtype_labels_present]
    ))

# ══════════════════════════════════════════════════════════════════════════════
# 16. ROC HELPER
# ══════════════════════════════════════════════════════════════════════════════
def plot_roc_multiclass(true_y, prob_y, class_names, title):
    n_cls    = len(class_names)
    y_bin    = label_binarize(true_y, classes=np.arange(n_cls))
    auc_dict = {}
    fig, ax  = plt.subplots(figsize=(9, 7))
    colors   = plt.cm.tab20(np.linspace(0, 1, n_cls))
    fpr_all, tpr_all = [], []
    for i, (cls, col) in enumerate(zip(class_names, colors)):
        fpr, tpr, _   = roc_curve(y_bin[:, i], prob_y[:, i])
        roc_auc       = auc(fpr, tpr)
        auc_dict[cls] = roc_auc
        ax.plot(fpr, tpr, color=col, lw=1.5,
                label=f"{cls}  (AUC={roc_auc:.3f})")
        fpr_all.append(fpr); tpr_all.append(tpr)
    all_fpr  = np.unique(np.concatenate(fpr_all))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_cls):
        mean_tpr += np.interp(all_fpr, fpr_all[i], tpr_all[i])
    mean_tpr /= n_cls
    macro_auc = auc(all_fpr, mean_tpr)
    ax.plot(all_fpr, mean_tpr, "k--", lw=2,
            label=f"Macro avg  (AUC={macro_auc:.3f})")
    ax.plot([0,1],[0,1],"gray",lw=0.8,linestyle=":")
    ax.set_xlim([-0.02,1.02]); ax.set_ylim([-0.02,1.05])
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=7, framealpha=0.7)
    plt.tight_layout(); plt.show()
    return auc_dict, macro_auc

# ══════════════════════════════════════════════════════════════════════════════
# 17. ROC — M_REGION — VAL + TEST
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 17: ROC curves — M_region (val + test)")
print("=" * 60)

region_aucs = {}
for split_name in ["val", "test"]:
    res = results[split_name]
    auc_dict, macro = plot_roc_multiclass(
        res["tr"], res["pbr"], le_region.classes_,
        f"ROC — M_region  ({split_name})"
    )
    region_aucs[split_name] = auc_dict
    print(f"\n{split_name.upper()} — M_region AUCs:")
    for cls, v in auc_dict.items():
        print(f"  {cls:<12}: {v:.4f}")
    print(f"  {'Macro avg':<12}: {macro:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 18. ROC — SUBTYPE — VAL + TEST
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 18: ROC curves — Subtype (val + test)")
print("=" * 60)

subtype_aucs = {}
for split_name in ["val", "test"]:
    res = results[split_name]
    auc_dict, macro = plot_roc_multiclass(
        res["ts"], res["pbs"], le_subtype.classes_,
        f"ROC — Subtype  ({split_name})"
    )
    subtype_aucs[split_name] = auc_dict

# ══════════════════════════════════════════════════════════════════════════════
# 19. INDIVIDUAL ROC PER M_REGION — TEST SET
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 19: Individual ROC per M_region — test set")
print("=" * 60)

res_test = results["test"]
y_bin    = label_binarize(res_test["tr"], classes=np.arange(n_regions))

n_cols = 4
n_rows = int(np.ceil(n_regions / n_cols))
fig, axes = plt.subplots(n_rows, n_cols,
                          figsize=(4.5 * n_cols, 4 * n_rows))
axes = axes.flatten()

for i, cls in enumerate(le_region.classes_):
    ax          = axes[i]
    fpr, tpr, _ = roc_curve(y_bin[:, i], res_test["pbr"][:, i])
    roc_auc     = auc(fpr, tpr)
    color = ("#1D9E75" if roc_auc >= 0.90 else
             "#EF9F27" if roc_auc >= 0.75 else "#E24B4A")
    ax.plot(fpr, tpr, color=color, lw=2)
    ax.fill_between(fpr, tpr, alpha=0.08, color=color)
    ax.plot([0,1],[0,1],"gray",lw=0.8,linestyle=":")
    ax.set_xlim([-0.02,1.02]); ax.set_ylim([-0.02,1.05])
    ax.set_title(f"{cls}", fontsize=11, fontweight="bold", color=color)
    ax.set_xlabel("FPR", fontsize=9); ax.set_ylabel("TPR", fontsize=9)
    ax.text(0.62, 0.12, f"AUC = {roc_auc:.3f}", transform=ax.transAxes,
            fontsize=10, color=color, fontweight="bold")

for j in range(n_regions, len(axes)):
    axes[j].set_visible(False)

plt.suptitle("ROC per M_region — test set  (one-vs-rest)",
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "roc_per_region.png"),
            dpi=150, bbox_inches="tight")
plt.show()

print("\nTest set per-region AUC:")
print(f"{'Region':<12}  {'AUC':>6}  Grade")
print("─" * 35)
for i, cls in enumerate(le_region.classes_):
    fpr, tpr, _ = roc_curve(y_bin[:, i], res_test["pbr"][:, i])
    roc_auc     = auc(fpr, tpr)
    grade = "strong" if roc_auc >= 0.90 else "moderate" if roc_auc >= 0.75 else "weak"
    print(f"  {cls:<12} {roc_auc:.4f}  {grade}")

# ══════════════════════════════════════════════════════════════════════════════
# 20. CONFUSION MATRIX + AUC BAR
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 20: Confusion matrix + AUC bar — test set")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(20, 8))
for ax, true_y, pred_y, classes, title in zip(
    axes,
    [res_test["tr"], res_test["ts"]],
    [res_test["pr"], res_test["ps"]],
    [le_region.classes_, le_subtype.classes_],
    ["M_region — test confusion matrix",
     f"Subtype ({subclass_col}) — test confusion matrix"]
):
    cm      = confusion_matrix(true_y, pred_y)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=classes, yticklabels=classes,
        linewidths=0.5, linecolor="white", ax=ax
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "confusion_matrix.png"),
            dpi=150, bbox_inches="tight")
plt.show()

sorted_regions = sorted(region_aucs["test"].items(),
                         key=lambda x: x[1], reverse=True)
regions_sorted = [r for r, _ in sorted_regions]
aucs_sorted    = [v for _, v in sorted_regions]
bar_colors     = ["#1D9E75" if v >= 0.90 else
                  "#EF9F27" if v >= 0.75 else "#E24B4A" for v in aucs_sorted]

fig, ax = plt.subplots(figsize=(11, 8))
bars = ax.barh(regions_sorted, aucs_sorted,
               color=bar_colors, edgecolor="white", height=0.6)
for bar, v in zip(bars, aucs_sorted):
    ax.text(v + 0.005, bar.get_y() + bar.get_height()/2,
            f"{v:.3f}", va="center", ha="left", fontsize=10)
ax.axvline(0.90, color="#1D9E75", linestyle="--", lw=1, alpha=0.6)
ax.axvline(0.75, color="#EF9F27", linestyle="--", lw=1, alpha=0.6)
ax.set_xlim([0, 1.10])
ax.set_xlabel("AUC (one-vs-rest)", fontsize=12)
ax.set_title("M_region AUC — test set", fontsize=13)
ax.invert_yaxis()
ax.legend(handles=[
    Patch(facecolor="#1D9E75", label="AUC ≥ 0.90  (strong)"),
    Patch(facecolor="#EF9F27", label="0.75 ≤ AUC < 0.90  (moderate)"),
    Patch(facecolor="#E24B4A", label="AUC < 0.75  (weak)"),
], loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "auc_bar.png"),
            dpi=150, bbox_inches="tight")
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# 21. EXTRACT ENCODER FEATURES
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 21: Extracting encoder features")
print("=" * 60)

model.eval()

def encode_batched(X_np, batch_size=4096):
    all_z = []
    X_t   = torch.tensor(X_np)
    for i in range(0, len(X_t), batch_size):
        batch = X_t[i:i+batch_size].to(device)
        with torch.no_grad():
            z = model.encode(batch)
        all_z.append(z.detach().cpu().numpy())
    return np.vstack(all_z)

def predict_batched(X_np, batch_size=4096):
    pred_r, pred_s, prob_r = [], [], []
    X_t = torch.tensor(X_np)
    for i in range(0, len(X_t), batch_size):
        batch = X_t[i:i+batch_size].to(device)
        with torch.no_grad():
            out_r, out_s, _ = model(batch)
        prob_r.append(torch.softmax(out_r, dim=1).detach().cpu().numpy())
        pred_r.extend(out_r.argmax(1).detach().cpu().numpy())
        pred_s.extend(out_s.argmax(1).detach().cpu().numpy())
    return np.array(pred_r), np.array(pred_s), np.vstack(prob_r)

Z_mouse = encode_batched(X_encoder_input)
mouse_pred_r, mouse_pred_s, mouse_prob_r = predict_batched(X_encoder_input)

print(f"Mouse encoder features : {Z_mouse.shape}")
print(f"M_region accuracy (all): {(mouse_pred_r == y_region).mean():.3f}")
print(f"Subtype accuracy  (all): {(mouse_pred_s == y_subtype).mean():.3f}")

adata.obs["predicted_M_region"] = le_region.inverse_transform(mouse_pred_r)
adata.obs["predicted_subtype"]  = le_subtype.inverse_transform(mouse_pred_s)
adata.obs["region_confidence"]  = mouse_prob_r.max(axis=1)
adata.obsm["Z_encoder"]         = Z_mouse

# ══════════════════════════════════════════════════════════════════════════════
# 22. SAVE PIPELINE OBJECTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 22: Saving pipeline objects")
print("=" * 60)

model_path        = os.path.join(SAVE_DIR, "encoder_model.pt")
le_path           = os.path.join(SAVE_DIR, "label_encoders.pkl")
genes_path        = os.path.join(SAVE_DIR, "verified_shared_genes.txt")
z_mouse_path      = os.path.join(SAVE_DIR, "Z_mouse.npy")
mouse_labels_path = os.path.join(SAVE_DIR, "mouse_region_labels.npy")

torch.save({
    "model_state_dict": model.state_dict(),
    "input_dim":        input_dim,
    "latent_dim":       latent_dim,
    "hidden_dim":       HIDDEN_DIM,
    "n_regions":        n_regions,
    "n_subtypes":        n_subtypes,
    "best_val_loss":    best_val_loss,
    "best_epoch":       best_epoch,
    "architecture":     "ResidualDualHeadClassifier",
}, model_path)
print(f"Model saved              : {model_path}")

with open(le_path, "wb") as f:
    pickle.dump({"le_region": le_region, "le_subtype": le_subtype}, f)
with open(genes_path, "w") as f:
    for g in verified_shared: f.write(g + "\n")

np.save(z_mouse_path,      Z_mouse)
np.save(mouse_labels_path, mouse_regions)
print(f"Z_mouse saved            : {z_mouse_path}  {Z_mouse.shape}")

adata.write_h5ad(os.path.join(SAVE_DIR, "mouse_adata_corrected.h5ad"))
print(f"AnnData saved            : mouse_adata_corrected.h5ad")

# ══════════════════════════════════════════════════════════════════════════════
# 23. HUMAN INFERENCE
#     Human is projected DIRECTLY into the same gene space the encoder was
#     trained on — no PCA basis, no bridge.
#     UPDATED — expected_map now uses the granular h_region labels actually
#     present in Human_N_filtered.csv, not the old aggregated H_ACAd/H_ACAv/
#     H_PL/H_ILA names (which don't exist in that file). Ambiguous mouse
#     regions (M_ACAd, M_ACAv, M_PL) share candidate human labels (H_A32,
#     H_ACC) per the earlier homology discussion — only ONE primary
#     candidate is used here per mouse region since this script's design
#     assumes a 1:1 map; the full tiered/multi-candidate comparison lives
#     in the separate cross_species_similarity.py script.
#     M_FRP and M_MOs have no established human counterpart in the current
#     mapping and are excluded from the homology evaluation (still trained
#     and still shown in the full similarity heatmap).
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 23: Human inference")
print("=" * 60)

Human_filtered  = pd.read_csv(human_filtered_path, index_col=0)
available_genes = [g for g in verified_shared if g in Human_filtered.columns]
missing_genes   = [g for g in verified_shared if g not in Human_filtered.columns]

print(f"Shared genes expected : {len(verified_shared)}")
print(f"Present in human data : {len(available_genes)}")
if missing_genes:
    print(f"Missing               : {missing_genes[:5]}")

X_human_input = Human_filtered[available_genes].values.astype(np.float32)
human_regions = Human_filtered["h_region"].values

print(f"\nHuman gene matrix : {X_human_input.shape}  (already log-normalized)")
print(f"Range             : {X_human_input.min():.3f} – {X_human_input.max():.3f}")
print(f"\nHuman region counts:")
for r, n in zip(*np.unique(human_regions, return_counts=True)):
    print(f"  {r:<12}: {n:,}")

# UPDATED expected_map — isocortex-only. M_HP/M_HY/M_STR/M_sAMY removed
# since mouse is no longer classified on those regions at all. Human
# h_region still contains H_HP/H_HY/H_STR/H_sAMY/H_AMY (from
# Human_N_filtered.csv's full division-level keep) — those human cells
# remain in the data and get encoded/predicted like any other, they just
# have no "expected match" row here and will only ever show up as
# off-diagonal comparisons, which is fine as a negative control.
expected_map = {
    "M_ORBl":   "H_A13",    # strong — orbital cortex
    "M_ORBm":   "H_A14",    # strong — orbital cortex
    "M_ORBvl":  "H_A13",    # strong — orbital cortex, shares candidate with ORBl
    "M_MOp":    "H_M1C",    # strong — primary motor cortex
    "M_ILA":    "H_A25",    # moderate-strong
    "M_ACAd":   "H_A32",    # moderate — ambiguous, shares candidates with M_PL/H_ACC
    "M_ACAv":   "H_ACC",    # moderate — ambiguous
    "M_PL":     "H_A32",    # moderate — ambiguous, shares H_A32 with M_ACAd
    "M_RSPd":   "H_A29_A30",   # moderate — retrosplenial
    "M_RSPv":   "H_A29_A30",   # moderate — retrosplenial, shares candidate with RSPd
    "M_RSPagl": "H_A29_A30",   # moderate — retrosplenial, shares candidate
    # M_FRP, M_MOs, M_AId, M_AIv, M_AIp: no established human counterpart
}

print(f"\n{'H_region':<12} {'Expected':<10} {'N cells':>8} "
      f"{'% correct':>10} {'Top pred':<14} {'Top%':>7} {'Match'}")
print("─" * 70)

print(f"\nEncoding {len(X_human_input):,} human cells (no ComBat — single batch)...")
Z_human = encode_batched(X_human_input)
print(f"Z_human: {Z_human.shape}")

human_pred_r, human_pred_s, human_prob_r = predict_batched(X_human_input)
human_pred_region  = le_region.inverse_transform(human_pred_r)
human_pred_subtype = le_subtype.inverse_transform(human_pred_s)
human_confidence   = human_prob_r.max(axis=1)

print(f"\nPredicted M_region distribution for human cells:")
for r, n in zip(*np.unique(human_pred_region, return_counts=True)):
    print(f"  {r:<12}: {n:,}  ({n/len(human_pred_region):.1%})")

print(f"\n{'H_region':<12} {'Expected':<10} {'N cells':>8} "
      f"{'% correct':>10} {'Top pred':<14} {'Top%':>7} {'Match'}")
print("─" * 70)
for m_region, h_region in expected_map.items():
    mask = human_regions == h_region
    if mask.sum() == 0:
        continue
    preds        = human_pred_region[mask]
    correct_frac = (preds == m_region).mean()
    vc           = pd.Series(preds).value_counts()
    top_pred     = vc.index[0]
    top_frac     = vc.iloc[0] / mask.sum()
    match        = "✓" if top_pred == m_region else "✗"
    print(f"  {h_region:<12} {m_region:<10} {mask.sum():>8,} "
          f"{correct_frac:>9.1%}  {top_pred:<14} {top_frac:>6.1%}  {match}")

# ══════════════════════════════════════════════════════════════════════════════
# 24. CROSS-SPECIES SIMILARITY MATRIX  (direct Z_mouse vs Z_human, no bridge)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 24: Cross-species similarity matrix")
print("=" * 60)

h_regions_ordered = sorted(np.unique(human_regions).tolist())
m_regions_ordered = le_region.classes_.tolist()

def pseudobulk(Z, labels, region_list):
    return {r: Z[labels == r].mean(axis=0)
            for r in region_list if (labels == r).sum() > 0}

def cosine_sim(a, b):
    return float(np.dot(a, b) /
                 (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

pb_mouse = pseudobulk(Z_mouse, mouse_regions, m_regions_ordered)
pb_human = pseudobulk(Z_human, human_regions, h_regions_ordered)

sim_cosine  = pd.DataFrame(index=m_regions_ordered,
                            columns=h_regions_ordered, dtype=float)
sim_pearson = pd.DataFrame(index=m_regions_ordered,
                            columns=h_regions_ordered, dtype=float)

for mr in m_regions_ordered:
    for hr in h_regions_ordered:
        if mr in pb_mouse and hr in pb_human:
            sim_cosine.loc[mr, hr]  = cosine_sim(pb_mouse[mr], pb_human[hr])
            sim_pearson.loc[mr, hr] = float(
                np.corrcoef(pb_mouse[mr], pb_human[hr])[0, 1]
            )

print("Cosine similarity matrix:")
print(sim_cosine.round(3).to_string())
print("\nPearson similarity matrix:")
print(sim_pearson.round(3).to_string())

# UPDATED — diagonal_pairs now built directly from expected_map so it can
# never drift out of sync with the human-inference block above.
diagonal_pairs = [(m, h) for m, h in expected_map.items()]

print(f"\n{'Mouse':<12} {'Human':<12} {'Cosine':>8} {'Pearson':>8}  Grade")
print("─" * 55)
for mr, hr in diagonal_pairs:
    if (mr in sim_cosine.index and hr in sim_cosine.columns
            and pd.notna(sim_cosine.loc[mr, hr])):
        c = sim_cosine.loc[mr, hr]; p = sim_pearson.loc[mr, hr]
        grade = "strong" if c > 0.5 else "moderate" if c > 0.2 else "weak"
        print(f"  {mr:<12} {hr:<12} {c:>8.3f} {p:>8.3f}  {grade}")

# ══════════════════════════════════════════════════════════════════════════════
# 25. EVALUATION PLOTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 25: Cross-species evaluation plots")
print("=" * 60)

print("Computing joint mouse + human UMAP in latent space...")
rng_sub = np.random.default_rng(42)
m_idx   = rng_sub.choice(len(Z_mouse), min(30000, len(Z_mouse)), replace=False)
h_idx   = rng_sub.choice(len(Z_human), min(10000, len(Z_human)), replace=False)

Z_sub  = np.vstack([Z_mouse[m_idx], Z_human[h_idx]])
sp_sub = ["mouse"] * len(m_idx) + ["human"] * len(h_idx)
mr_sub = list(mouse_regions[m_idx]) + list(human_pred_region[h_idx])

adata_joint                = ad.AnnData(X=Z_sub.astype(np.float32))
adata_joint.obs["species"] = sp_sub
adata_joint.obs["region"]  = mr_sub

sc.pp.neighbors(adata_joint, use_rep="X", n_neighbors=15)
sc.tl.umap(adata_joint)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
sc.pl.umap(adata_joint, color="species", ax=axes[0], show=False,
           title="Joint latent space — by species\n(mouse + human cells)",
           palette={"mouse": "#5A9E8C", "human": "#E07B60"})
sc.pl.umap(adata_joint, color="region", ax=axes[1], show=False,
           title="Joint latent space — by predicted M_region",
           legend_fontsize=6)
plt.suptitle("Cross-species UMAP in encoder latent space",
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "umap_joint_species.png"),
            dpi=150, bbox_inches="tight")
plt.show()

print("Computing homologous region correlation heatmap...")
corr_matrix = pd.DataFrame(index=m_regions_ordered,
                            columns=h_regions_ordered, dtype=float)
for mr in m_regions_ordered:
    for hr in h_regions_ordered:
        if mr in pb_mouse and hr in pb_human:
            corr_matrix.loc[mr, hr] = cosine_sim(pb_mouse[mr], pb_human[hr])

fig, ax = plt.subplots(figsize=(13, 8))
sns.heatmap(
    corr_matrix.astype(float),
    annot=True, fmt=".2f", cmap="YlOrRd",
    vmin=0, vmax=1,
    linewidths=0.5, linecolor="white",
    ax=ax, annot_kws={"size": 9}
)
for mr in m_regions_ordered:
    hr = expected_map.get(mr)
    if hr and hr in h_regions_ordered:
        j = h_regions_ordered.index(hr)
        i = m_regions_ordered.index(mr)
        ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False,
                                    edgecolor="#2D3748", lw=3))
ax.set_title("Cross-species cosine similarity in latent space\n"
             "(black boxes = expected homologous pairs; M_FRP/M_MOs have no "
             "established human counterpart)",
             fontsize=11, pad=12)
ax.set_xlabel("Human regions", fontsize=11)
ax.set_ylabel("Mouse regions", fontsize=11)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "corr_heatmap_homologous.png"),
            dpi=150, bbox_inches="tight")
plt.show()

print("Computing diagonal vs random correlation boxplot...")
diagonal_vals = []
offdiag_vals  = []
for mr in m_regions_ordered:
    for hr in h_regions_ordered:
        if mr in pb_mouse and hr in pb_human:
            v = cosine_sim(pb_mouse[mr], pb_human[hr])
            if expected_map.get(mr) == hr:
                diagonal_vals.append(v)
            else:
                offdiag_vals.append(v)

t_stat, p_val = stats.ttest_ind(diagonal_vals, offdiag_vals)
sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else \
      "*"   if p_val < 0.05  else "ns"

fig, ax = plt.subplots(figsize=(6, 5))
ax.boxplot([diagonal_vals, offdiag_vals],
           tick_labels=["Homologous pairs\n(diagonal)",
                        "Non-homologous\n(off-diagonal)"],
           patch_artist=True,
           boxprops=dict(facecolor="#C8E6C9"),
           medianprops=dict(color="#2D3748", linewidth=2))
ax.set_ylabel("Cosine similarity (latent space)", fontsize=11)
ax.set_title("Homologous vs random region similarity", fontsize=12)
y_max = max(max(diagonal_vals), max(offdiag_vals)) + 0.05
ax.plot([1, 2], [y_max, y_max], "k-", lw=1.5)
ax.text(1.5, y_max + 0.01, f"{sig} (p={p_val:.3f})",
        ha="center", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "boxplot_diagonal_vs_random.png"),
            dpi=150, bbox_inches="tight")
plt.show()

print(f"\nHomologous mean : {np.mean(diagonal_vals):.3f} ± {np.std(diagonal_vals):.3f}")
print(f"Off-diagonal mean: {np.mean(offdiag_vals):.3f} ± {np.std(offdiag_vals):.3f}")
print(f"t-test p-value  : {p_val:.4f}  {sig}")

print("Computing per-region alignment fold change...")
fold_changes = {}
for mr, hr in diagonal_pairs:
    if mr in pb_mouse and hr in pb_human:
        diag_sim = cosine_sim(pb_mouse[mr], pb_human[hr])
        off_sims = [
            cosine_sim(pb_mouse[mr], pb_human[hr2])
            for hr2 in h_regions_ordered
            if hr2 != hr and hr2 in pb_human
        ]
        if off_sims:
            fold_changes[mr] = diag_sim / np.mean(off_sims)

regions_fc = list(fold_changes.keys())
fc_vals    = list(fold_changes.values())
colors_fc  = ["#5A9E8C" if v > 1.0 else "#E07B60" for v in fc_vals]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(regions_fc, fc_vals, color=colors_fc, edgecolor="white", width=0.6)
ax.axhline(1.0, color="gray", linestyle="--", lw=1.5, label="Fold change = 1.0")
for bar, v in zip(bars, fc_vals):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
            f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")
ax.set_ylabel("Fold change (diagonal / mean off-diagonal)", fontsize=11)
ax.set_title("Per-region alignment fold change", fontsize=12)
ax.set_xticks(range(len(regions_fc)))
ax.set_xticklabels(regions_fc, rotation=45, ha="right")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "fold_change_per_region.png"),
            dpi=150, bbox_inches="tight")
plt.show()

print("\nFold changes per region:")
for mr, fc in fold_changes.items():
    status = "↑ conserved" if fc > 1.1 else "→ neutral" if fc > 0.9 else "↓ divergent"
    print(f"  {mr:<12}: {fc:.3f}  {status}")

# ══════════════════════════════════════════════════════════════════════════════
# 26. PLOT AND SAVE SIMILARITY HEATMAPS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 26: Saving similarity heatmaps")
print("=" * 60)

def plot_and_save_similarity(sim_df, title, fname):
    fig, ax = plt.subplots(figsize=(15, 8))
    sns.heatmap(
        sim_df.astype(float), annot=True, fmt=".2f",
        cmap="RdYlGn", center=0, vmin=-1, vmax=1,
        linewidths=0.5, linecolor="white", ax=ax, annot_kws={"size": 8}
    )
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel("Human regions", fontsize=11)
    ax.set_ylabel("Mouse regions", fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved: {fname}")
    plt.show()

plot_and_save_similarity(
    sim_cosine,
    "Cross-species region similarity — cosine\n"
    "(encoder pseudobulk: mouse M_region vs human H_region)",
    os.path.join(SAVE_DIR, "similarity_cosine.png")
)
plot_and_save_similarity(
    sim_pearson,
    "Cross-species region similarity — Pearson\n"
    "(encoder pseudobulk: mouse M_region vs human H_region)",
    os.path.join(SAVE_DIR, "similarity_pearson.png")
)

# ══════════════════════════════════════════════════════════════════════════════
# 27. SAVE ALL RESULTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 27: Saving all results")
print("=" * 60)

np.save(os.path.join(SAVE_DIR, "Z_human.npy"), Z_human)

human_results_df = pd.DataFrame({
    "cell_id":           Human_filtered.index,
    "h_region":          human_regions,
    "predicted_region":  human_pred_region,
    "predicted_subtype": human_pred_subtype,
    "confidence":        human_confidence,
})
human_results_df.to_csv(
    os.path.join(SAVE_DIR, "human_predictions.csv"), index=False
)
print(f"Human predictions saved    : human_predictions.csv  "
      f"({len(human_results_df):,} cells)")

sim_cosine.to_csv(os.path.join(SAVE_DIR,  "similarity_matrix_cosine.csv"))
sim_pearson.to_csv(os.path.join(SAVE_DIR, "similarity_matrix_pearson.csv"))

print(f"\n{'─'*60}")
print(f"ALL DONE")
print(f"{'─'*60}")
print(f"Output folder : {SAVE_DIR}")
print(f"\nContents:")
for fname in sorted(os.listdir(SAVE_DIR)):
    fsize = os.path.getsize(os.path.join(SAVE_DIR, fname))
    unit  = "MB" if fsize > 1e6 else "KB"
    size  = fsize/1e6 if fsize > 1e6 else fsize/1e3
    print(f"  {fname:<45} {size:>8.1f} {unit}")

print(f"\nKey pipeline notes for this run:")
print(f"  Order                : RAW counts (ROI only) → log1p → split")
print(f"  Batch correction     : NONE — skipped for now, per instruction")
print(f"  ROI filter           : applied during load (step 4), before log1p")
print(f"  Encoder input        : {len(verified_shared)}-gene log1p expression, "
      f"no ComBat, no PCA bottleneck")
print(f"  Human projection     : direct — same gene space, no bridge")
print(f"  M_region set         : isocortex-only, 16 regions "
      f"(FRP/ACAd/ACAv/PL/ILA/ORBl/ORBm/ORBvl/MOs/MOp/RSPd/RSPv/RSPagl/AId/AIv/AIp)")
print(f"  h_region source      : Human_N_filtered.csv, granular naming "
      f"(H_A13/H_A14/H_M1C/H_A25/H_A32/H_ACC/... )")
print(f"  Contrastive loss     : REMOVED")
print(f"  Overfitting fixes    : LR 1e-3→3e-4, weight_decay 1e-4→1e-3, "
      f"hidden_dim 512→256, dropout 0.2/0.3→0.4/0.5")
print(f"  Early stopping       : patience={PATIENCE}, max epochs={N_EPOCHS}, "
      f"min_delta={MIN_DELTA}")
print(f"  Best epoch           : {best_epoch}")
print(f"  Homologous mean cosine : {np.mean(diagonal_vals):.3f}")
print(f"  Off-diagonal mean cosine: {np.mean(offdiag_vals):.3f}")
print(f"  t-test significance    : {sig} (p={p_val:.4f})")