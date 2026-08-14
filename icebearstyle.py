import pandas as pd
import numpy as np
import os
import gc
import pickle
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.stats import ttest_ind
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ══════════════════════════════════════════════════════════════════════════════
# 0. PATHS
# ══════════════════════════════════════════════════════════════════════════════
DATA_DIR = r"U:\Research\microarray"
DATA     = r"C:\Users\StujenskeLab\Documents\NAN151_workspace\microarray"

fpath1 = os.path.join(DATA_DIR, 'merfish_F1_raw.csv')
fpath2 = os.path.join(DATA_DIR, 'merfish_F2_raw.csv')
fpath3 = os.path.join(DATA_DIR, 'merfish_M1_raw.csv')
fpath4 = os.path.join(DATA_DIR, 'merfish_M2_Glut_raw.csv')
human_filtered_path = os.path.join(DATA_DIR, "Human_N_filtered.csv")

MOUSE_SAVE_DIR      = os.path.join(DATA, "mouse_region_classifier")
verified_genes_path = os.path.join(MOUSE_SAVE_DIR, "verified_genes.txt")

SAVE_DIR = os.path.join(DATA, "icebear_style_joint")
os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 60)
print("ICEBEAR-STYLE CONDITIONAL VAE — mouse (raw, 4 animals) + human")
print("(log-normalized, single/no-batch)")
print("=" * 60)
print("Design, adapted from Zhang/Yang et al. 'Icebear':")
print("  - Conditional VAE: reconstructs each cell from latent z + species")
print("    one-hot + batch one-hot (animal_id for mouse, donor_label for")
print("    human if available)")
print("  - DUAL reconstruction likelihood, since only mouse has raw counts:")
print("      mouse cells -> ZINB loss against raw counts (library-size")
print("                     corrected internally, no external normalization)")
print("      human cells -> Gaussian/MSE loss against the already-log-")
print("                     normalized values (no raw counts available)")
print("  - Optional adversarial species discriminator (alternating GAN-style")
print("    steps, per the paper) on top of the KL regularization")
print("  - z is NOT conditioned on region — region is what we evaluate for")
print("    alignment afterward (homolog alignment loss + pseudobulk cosine)")
print("  - REGENERATED for isocortex-only regions: mouse restricted to 16")
print("    isocortex structures (no hippocampus/striatum/hypothalamus/")
print("    extended amygdala), human restricted to the 10 curated Cerebral")
print("    cortex subregions from Human_N_filtered.csv. Both species are")
print("    now purely cortical, so STRONG_PAIRS only anchors on unambiguous")
print("    cortical homologs (M_MOp<->H_M1C, M_ORBl<->H_A13, M_ORBm<->H_A14).")
print(f"Device: {device}")
print(f"Save directory: {SAVE_DIR}")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
LATENT_DIM      = 50     # Icebear grid-searched {25,50,100}; 50 as a default
HIDDEN_DIM      = 256
MAX_CELLS_PER_ANIMAL = 150000
MAX_HUMAN_CELLS      = 300000
BATCH_SIZE      = 1024
N_EPOCHS        = 150
LR              = 3e-4
WEIGHT_DECAY    = 1e-4
PATIENCE        = 30       # Icebear used 45; adjust if you have the epoch budget
MIN_DELTA       = 1e-4
USE_ADVERSARIAL = True     # Icebear's optional GAN-style species discriminator
LAMBDA_ALIGN    = 1.0
ALIGN_MARGIN    = 1.0

# KL warmup — per the source: kl_weight ramps 0 -> KL_WEIGHT linearly over
# KL_WARMUP_EPOCHS starting at KL_START_EPOCH. Missing this causes posterior
# collapse (the encoder ignores the KL term entirely, z becomes meaningless).
KL_WEIGHT        = 1.0
KL_START_EPOCH   = 5
KL_WARMUP_EPOCHS = 20

# Two-phase training, per the source: Phase 1 trains the VAE alone
# (reconstruction + KL + region heads + alignment) to convergence. Phase 2
# then alternates discriminator/generator updates with a separate, lower
# LR, on a per-species-balanced subsample — not mixed in via gradient
# reversal from epoch 1 the way a single-phase GRL approach would do it.
DISC_LR           = 0.0001   # matches the source's hardcoded discriminator LR
DISC_BALANCE_CAP  = 5000     # per-species cap for discriminator training batches
PHASE2_MAX_EPOCHS = 150      # raised from 60 — last run stopped at epoch 21
                              # with domain_acc still climbing (0.41->0.53),
                              # meaning the adversarial dynamic hadn't
                              # converged yet, just ran out of patience
PHASE2_PATIENCE   = 30       # raised from 15, paired with the domain_acc-
                              # gap selection criterion above

STRONG_PAIRS = [
    ("M_MOp",  "H_M1C"),
    ("M_ORBl", "H_A13"),
    ("M_ORBm", "H_A14"),
]
# NOTE: with mouse restricted to isocortex-only, M_HP/M_HY/M_STR/M_sAMY no
# longer exist as classes (that homology was subcortical, and both species
# are now cortex-only — Human_N_filtered.csv also dropped its subcortical
# divisions). Only strong, unambiguous cortical pairs are used as training
# anchors here; the moderate-tier pairs (M_ACAd/M_ACAv/M_PL vs H_A32/H_ACC,
# and the new M_RSPd/v/agl vs H_A29_A30 retrosplenial claim) are
# deliberately left OUT of training and only checked post-hoc in the
# evaluation step below — see expected_pairs_eval.

# ══════════════════════════════════════════════════════════════════════════════
# 1. GENE PANEL (still case-insensitive name matching — see note at the end
#    about upgrading to a proper one-to-one ortholog graph, per the paper's
#    Methods 2.3; that is a separate, worthwhile fix not included here)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 1: Gene panel")
print("=" * 60)

with open(verified_genes_path) as f:
    verified_genes = [line.strip() for line in f if line.strip()]
hdr_h = pd.read_csv(human_filtered_path, index_col=0, nrows=0)
common_genes = [g for g in verified_genes if g in hdr_h.columns.tolist()]
print(f"Shared gene panel: {len(common_genes)} genes")

# ══════════════════════════════════════════════════════════════════════════════
# 2. MOUSE — LOAD RAW COUNTS (not log-transformed — ZINB needs raw)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: Loading mouse RAW counts (subsampled)")
print("=" * 60)

rename_map = {
    "CCF_level1": "parcellation_division", "CCF_level2": "parcellation_structure",
}
structure_regions  = [
    "FRP", "ACAd", "ACAv", "PL", "ILA", "ORBl", "ORBm", "ORBvl",
    "MOs", "MOp", "RSPd", "RSPv", "RSPagl", "AId", "AIv", "AIp",
]
region_rename = {
    "FRP": "M_FRP", "ACAd": "M_ACAd", "ACAv": "M_ACAv", "PL": "M_PL",
    "ILA": "M_ILA", "ORBl": "M_ORBl", "ORBm": "M_ORBm", "ORBvl": "M_ORBvl",
    "MOs": "M_MOs", "MOp": "M_MOp",
    "RSPd": "M_RSPd", "RSPv": "M_RSPv", "RSPagl": "M_RSPagl",
    "AId": "M_AId", "AIv": "M_AIv", "AIp": "M_AIp",
}

# Isocortex-only — matches the human side, which is now also cortex-only
# (Human_N_filtered.csv no longer keeps Hippocampus/Hypothalamus/Basal
# nuclei/Basal forebrain/Extended amygdala/Amygdaloid complex in full).
# Hippocampus, striatum, hypothalamus, and extended amygdala cells simply
# get no M_region at all and are excluded from the ROI.
def assign_region_vectorized(div_arr, struct_arr):
    region = np.full(len(div_arr), None, dtype=object)
    for s in structure_regions:
        region[struct_arr == s] = s
    return region

animal_paths = [(fpath1, "mouse_F1"), (fpath2, "mouse_F2"),
                 (fpath3, "mouse_M1"), (fpath4, "mouse_M2")]
rng = np.random.default_rng(42)
META_CHUNK = 200000

mouse_X_raw, mouse_region_raw, mouse_batch_raw = [], [], []

for fpath, batch_id in animal_paths:
    print(f"\n  {batch_id}: scanning ...")
    hdr_cols = pd.read_csv(fpath, nrows=0).columns.tolist()
    gene_cols_in_file = [g for g in common_genes if g in hdr_cols]
    idx_col = hdr_cols[0]

    # Detect the actual division/structure column names in THIS file — not
    # every mouse file uses "CCF_level1"/"CCF_level2" (e.g. merfish_M2_Glut_raw.csv
    # uses different column names). Try both the raw CCF names and the
    # already-renamed parcellation_* names before giving up.
    div_col_candidates    = ["CCF_level1", "parcellation_division"]
    struct_col_candidates = ["CCF_level2", "parcellation_structure"]
    div_col_in_file    = next((c for c in div_col_candidates    if c in hdr_cols), None)
    struct_col_in_file = next((c for c in struct_col_candidates if c in hdr_cols), None)
    if div_col_in_file is None or struct_col_in_file is None:
        raise KeyError(
            f"{batch_id}: could not find division/structure columns among "
            f"{div_col_candidates + struct_col_candidates}. "
            f"Actual columns in this file (first 30): {hdr_cols[:30]}\n"
            f"Add this file's real column names to the candidate lists above."
        )
    print(f"    division column: '{div_col_in_file}'   structure column: '{struct_col_in_file}'")

    meta_reader = pd.read_csv(fpath, usecols=[idx_col, div_col_in_file, struct_col_in_file],
                               index_col=idx_col, low_memory=False, chunksize=META_CHUNK)
    gene_reader = pd.read_csv(fpath, usecols=[idx_col] + gene_cols_in_file,
                               index_col=idx_col, low_memory=False, chunksize=META_CHUNK,
                               dtype={g: np.float32 for g in gene_cols_in_file})
    animal_X, animal_r = [], []
    for chunk_meta, chunk_genes in zip(meta_reader, gene_reader):
        chunk_meta = chunk_meta.rename(columns={div_col_in_file: "parcellation_division",
                                                  struct_col_in_file: "parcellation_structure"})
        div_arr = chunk_meta["parcellation_division"].fillna("").values
        struct_arr = chunk_meta["parcellation_structure"].fillna("").values
        raw_labels = assign_region_vectorized(div_arr, struct_arr)
        m_labels = np.array(
            [region_rename.get(r, None) if r is not None else None for r in raw_labels],
            dtype=object)
        roi_mask = m_labels != None
        if roi_mask.sum() == 0:
            continue
        X_chunk = chunk_genes.reindex(columns=common_genes).fillna(0).values.astype(np.float32)
        animal_X.append(X_chunk[roi_mask])   # RAW counts, no log1p here
        animal_r.append(m_labels[roi_mask])
    del meta_reader, gene_reader; gc.collect()

    X_animal = np.vstack(animal_X); r_animal = np.concatenate(animal_r)
    n_take = min(MAX_CELLS_PER_ANIMAL, len(X_animal))
    take_idx = rng.choice(len(X_animal), size=n_take, replace=False)
    mouse_X_raw.append(X_animal[take_idx])
    mouse_region_raw.append(r_animal[take_idx])
    mouse_batch_raw.append(np.full(n_take, batch_id))
    print(f"    {batch_id}: {len(X_animal):,} available, {n_take:,} sampled")

X_mouse_raw = np.vstack(mouse_X_raw).astype(np.float32)
mouse_regions = np.concatenate(mouse_region_raw)
mouse_batches = np.concatenate(mouse_batch_raw)
del mouse_X_raw, mouse_region_raw, mouse_batch_raw; gc.collect()
print(f"\nTotal mouse cells: {len(X_mouse_raw):,}")
print(f"Raw count range: {X_mouse_raw.min():.1f} - {X_mouse_raw.max():.1f}")

# encoder input needs a stable numeric scale even though the RECONSTRUCTION
# target is raw counts — log1p just for what goes into the encoder, exactly
# mirroring how Icebear's µ0 is a depth-corrected mean learned FROM raw
# counts, while any real model still needs a well-scaled encoder input.
X_mouse_enc_input = np.log1p(X_mouse_raw)
mouse_library_size = X_mouse_raw.sum(axis=1, keepdims=True).astype(np.float32)
mouse_library_size = np.clip(mouse_library_size, 1.0, None)

# ══════════════════════════════════════════════════════════════════════════════
# 3. HUMAN — LOAD LOG-NORMALIZED VALUES (no raw counts available)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: Loading human data — filtering to Cerebral cortex (in-line)")
print("=" * 60)
print("Human_N_filtered.csv now contains ALL human cells (all divisions),")
print("not a pre-filtered subset. Filtering to the full Cerebral cortex")
print("division and deriving h_region happens right here, rather than in")
print("a separate preprocessing script.")

DIV_COL = "anatomical_division_label"
ROI_COL = "region_of_interest_label"

all_cols_h = hdr_h.columns.tolist()
has_donor  = "donor_label" in all_cols_h
for col in [DIV_COL, ROI_COL]:
    if col not in all_cols_h:
        raise KeyError(
            f"'{col}' not found in {human_filtered_path}. Available columns "
            f"(first 20): {all_cols_h[:20]}"
        )

load_cols_h = [DIV_COL, ROI_COL] + (["donor_label"] if has_donor else []) + common_genes
Human = pd.read_csv(human_filtered_path, index_col=0,
                     usecols=[hdr_h.index.name or "Unnamed: 0"] + load_cols_h)
print(f"Human cells loaded (all divisions): {len(Human):,}")
print(f"\nDivision counts:")
print(Human[DIV_COL].value_counts().to_string())

# ── Filter to the full Cerebral cortex division ────────────────────────────
n_before = len(Human)
Human = Human[Human[DIV_COL] == "Cerebral cortex"].copy()
print(f"\nKept {len(Human):,} / {n_before:,} cells (Cerebral cortex only)")

# ── Derive h_region — general transform, covers whatever cortex labels are
#    actually present, not a fixed 10-label list. "Human A29-A30" ->
#    "H_A29_A30", "Human A13" -> "H_A13", etc. Verified to reproduce the
#    exact same names as the old curated-list version for those 10 labels,
#    so existing homology tables (STRONG_PAIRS, expected_pairs_eval) still
#    work unchanged.
def roi_label_to_h_region(label):
    if not isinstance(label, str):
        return None
    name = label
    if name.startswith("Human "):
        name = name[len("Human "):]
    name = name.replace("-", "_").replace(" ", "_")
    return f"H_{name}"

Human["h_region"] = Human[ROI_COL].apply(roi_label_to_h_region)
Human = Human[Human["h_region"].notna()].copy()
print(f"Distinct h_region values: {Human['h_region'].nunique()}")
print(f"(donor_label present: {has_donor})")

if len(Human) > MAX_HUMAN_CELLS:
    Human = Human.groupby("h_region", group_keys=False).apply(
        lambda g: g.sample(n=max(1, int(MAX_HUMAN_CELLS * len(g) / len(Human))), random_state=42))
    print(f"Subsampled to: {len(Human):,}")

X_human_enc_input = Human[common_genes].values.astype(np.float32)   # already log-normalized
human_regions = Human["h_region"].values
human_batches = (Human["donor_label"].astype(str).apply(lambda d: f"human_{d}").values
                  if has_donor else np.full(len(Human), "human_batch0"))
print(f"Human log-normalized range: {X_human_enc_input.min():.3f} - {X_human_enc_input.max():.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. BUILD JOINT DATASET — species + batch one-hots, region labels for
#    evaluation only (never fed to the model as input)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4: Building joint dataset")
print("=" * 60)

le_batch = LabelEncoder()
all_batches = np.concatenate([mouse_batches, human_batches])
y_batch_all = le_batch.fit_transform(all_batches)
n_batches = len(le_batch.classes_)
print(f"Batch vocabulary ({n_batches}): {le_batch.classes_.tolist()}")

species_all = np.concatenate([np.zeros(len(X_mouse_raw), dtype=np.int64),
                               np.ones(len(X_human_enc_input), dtype=np.int64)])
X_enc_all = np.concatenate([X_mouse_enc_input, X_human_enc_input], axis=0).astype(np.float32)

le_mouse = LabelEncoder(); y_mouse_full = le_mouse.fit_transform(mouse_regions)
le_human = LabelEncoder(); y_human_full = le_human.fit_transform(human_regions)
y_mouse_all = np.concatenate([y_mouse_full, np.full(len(X_human_enc_input), -1, dtype=np.int64)])
y_human_all = np.concatenate([np.full(len(X_mouse_raw), -1, dtype=np.int64), y_human_full])

lib_all = np.concatenate([mouse_library_size.squeeze(1),
                           np.ones(len(X_human_enc_input), dtype=np.float32)])

strong_pairs_encoded = []
for m_name, h_name in STRONG_PAIRS:
    if m_name in le_mouse.classes_ and h_name in le_human.classes_:
        strong_pairs_encoded.append(
            (int(np.where(le_mouse.classes_ == m_name)[0][0]),
             int(np.where(le_human.classes_ == h_name)[0][0])))
    else:
        print(f"  SKIP alignment pair {m_name}<->{h_name}: not present")
print(f"Active alignment anchor pairs: {len(strong_pairs_encoded)}")

n_mouse_regions = len(le_mouse.classes_)
n_human_regions = len(le_human.classes_)
input_dim = len(common_genes)

idx_all = np.arange(len(X_enc_all))
strat_key = np.array([f"{s}_{max(m,h)}" for s, m, h in zip(species_all, y_mouse_all, y_human_all)])
idx_trainval, idx_test = train_test_split(idx_all, test_size=0.10, stratify=strat_key, random_state=42)
idx_train, idx_val = train_test_split(idx_trainval, test_size=0.15,
                                       stratify=strat_key[idx_trainval], random_state=42)
print(f"Train: {len(idx_train):,}  Val: {len(idx_val):,}  Test: {len(idx_test):,}")

n_mouse = len(X_mouse_raw)

# ══════════════════════════════════════════════════════════════════════════════
# 5. MODEL — conditional VAE: encoder Q(z|x,s,b), decoder with dual heads
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5: Building conditional VAE")
print("=" * 60)

def zinb_nll(x, mu, theta, pi_logits, eps=1e-8):
    """Zero-inflated negative binomial negative log-likelihood, per-element.
    x: raw counts. mu: mean. theta: dispersion. pi_logits: dropout logit."""
    softplus_pi = F.softplus(-pi_logits)
    log_theta_eps = torch.log(theta + eps)
    log_theta_mu_eps = torch.log(theta + mu + eps)
    pi_theta_log = -pi_logits + theta * (log_theta_eps - log_theta_mu_eps)
    case_zero = F.softplus(pi_theta_log) - softplus_pi
    case_non_zero = (
        -softplus_pi + pi_theta_log
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta) - torch.lgamma(theta) - torch.lgamma(x + 1)
    )
    is_zero = (x < eps).float()
    return -(is_zero * case_zero + (1 - is_zero) * case_non_zero)

class ConditionalEncoder(nn.Module):
    def __init__(self, input_dim, cond_dim, latent_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + cond_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.LeakyReLU(),
        )
        self.mu_head     = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)
    def forward(self, x, cond):
        h = self.net(torch.cat([x, cond], dim=1))
        return self.mu_head(h), self.logvar_head(h)

class ConditionalDecoder(nn.Module):
    """Dual-head decoder: ZINB params for mouse (raw counts), Gaussian mean
    for human (log-normalized values)."""
    def __init__(self, latent_dim, cond_dim, gene_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.LeakyReLU(),
        )
        self.px_scale_decoder = nn.Linear(hidden_dim, gene_dim)   # -> softmax -> proportions
        self.px_dropout_decoder = nn.Linear(hidden_dim, gene_dim) # -> pi logits
        self.px_r = nn.Parameter(torch.randn(gene_dim) * 0.1)     # per-gene log-dispersion
        self.human_mean_decoder = nn.Linear(hidden_dim, gene_dim)
    def forward(self, z, cond, library_size):
        h = self.net(torch.cat([z, cond], dim=1))
        px_scale = F.softmax(self.px_scale_decoder(h), dim=-1)
        mu_mouse = library_size.unsqueeze(1) * px_scale
        theta = torch.exp(self.px_r).unsqueeze(0).expand_as(mu_mouse)
        pi_logits = self.px_dropout_decoder(h)
        mean_human = self.human_mean_decoder(h)
        return mu_mouse, theta, pi_logits, mean_human

class IcebearStyleModel(nn.Module):
    def __init__(self, input_dim, n_batches, latent_dim, hidden_dim):
        super().__init__()
        cond_dim = 2 + n_batches
        self.n_batches = n_batches
        self.encoder = ConditionalEncoder(input_dim, cond_dim, latent_dim, hidden_dim)
        self.decoder = ConditionalDecoder(latent_dim, cond_dim, input_dim, hidden_dim)
        # Discriminator sizing follows the source's sqrt(embed_dim * nlabel)
        # heuristic rather than a fixed width.
        disc_hidden = max(8, int(math.sqrt(latent_dim * 2)))
        self.domain_head = nn.Sequential(
            nn.Linear(latent_dim, disc_hidden), nn.LayerNorm(disc_hidden), nn.LeakyReLU(),
            nn.Linear(disc_hidden, 2)
        )
    def make_cond(self, species, batch):
        sp_1h = F.one_hot(species, num_classes=2).float()
        b_1h  = F.one_hot(batch, num_classes=self.n_batches).float()
        return torch.cat([sp_1h, b_1h], dim=1)
    def forward(self, x, species, batch, library_size):
        cond = self.make_cond(species, batch)
        mu_q, logvar_q = self.encoder(x, cond)
        std = torch.exp(0.5 * logvar_q)
        z = mu_q + std * torch.randn_like(std)
        mu_mouse, theta, pi_logits, mean_human = self.decoder(z, cond, library_size)
        return mu_q, logvar_q, z, mu_mouse, theta, pi_logits, mean_human
    def encode_mean(self, x, species, batch):
        cond = self.make_cond(species, batch)
        with torch.no_grad():
            mu_q, _ = self.encoder(x, cond)
        return mu_q

model = IcebearStyleModel(input_dim, n_batches, LATENT_DIM, HIDDEN_DIM).to(device)
print(f"Input dim: {input_dim}  Latent dim: {LATENT_DIM}  Batches: {n_batches}")
print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")

# ══════════════════════════════════════════════════════════════════════════════
# 6. DATA LOADERS
# ══════════════════════════════════════════════════════════════════════════════
X_raw_full = np.zeros_like(X_enc_all)
X_raw_full[:n_mouse] = X_mouse_raw

def make_loader(idx, shuffle):
    X_enc = torch.tensor(X_enc_all[idx])
    X_raw = torch.tensor(X_raw_full[idx])
    sp    = torch.tensor(species_all[idx], dtype=torch.long)
    b     = torch.tensor(y_batch_all[idx], dtype=torch.long)
    lib   = torch.tensor(lib_all[idx])
    ym    = torch.tensor(y_mouse_all[idx], dtype=torch.long)
    yh    = torch.tensor(y_human_all[idx], dtype=torch.long)
    return DataLoader(TensorDataset(X_enc, X_raw, sp, b, lib, ym, yh),
                       batch_size=BATCH_SIZE, shuffle=shuffle,
                       num_workers=0, pin_memory=(device.type=="cuda"))

train_loader = make_loader(idx_train, True)
val_loader   = make_loader(idx_val,   False)

opt_main = optim.AdamW(
    list(model.encoder.parameters()) + list(model.decoder.parameters()),
    lr=LR, weight_decay=WEIGHT_DECAY)
# Discriminator uses its own, much lower, fixed LR — matches the source's
# hardcoded 0.0001, decoupled from the main VAE learning rate.
opt_disc = optim.AdamW(model.domain_head.parameters(), lr=DISC_LR, weight_decay=WEIGHT_DECAY)
sched_main = optim.lr_scheduler.ReduceLROnPlateau(opt_main, mode="min", factor=0.5, patience=10)

def alignment_loss(z, species, y_mouse, y_human, pairs):
    z_m = z[species == 0]; ym_b = y_mouse[species == 0]
    z_h = z[species == 1]; yh_b = y_human[species == 1]
    if len(z_m) == 0 or len(z_h) == 0:
        return torch.tensor(0.0, device=z.device)
    m_c = {mi.item(): z_m[ym_b == mi].mean(0) for mi in torch.unique(ym_b) if mi >= 0}
    h_c = {hi.item(): z_h[yh_b == hi].mean(0) for hi in torch.unique(yh_b) if hi >= 0}
    pull, push = [], []
    pair_set = set(pairs)
    for mi, mc in m_c.items():
        for hi, hc in h_c.items():
            d = F.pairwise_distance(mc.unsqueeze(0), hc.unsqueeze(0)).squeeze(0)
            if (mi, hi) in pair_set:
                pull.append(d**2)
            else:
                push.append(F.relu(ALIGN_MARGIN - d)**2)
    loss = torch.tensor(0.0, device=z.device)
    if pull: loss = loss + torch.stack(pull).mean()
    if push: loss = loss + 0.5 * torch.stack(push).mean()
    return loss

# ══════════════════════════════════════════════════════════════════════════════
# 7. TRAINING LOOP — TWO PHASES, matching the source structure:
#    Phase 1: train the VAE alone (reconstruction + KL warmup + region
#             heads + alignment) to convergence, no discriminator at all.
#    Phase 2: alternate discriminator/generator updates on a per-species-
#             balanced subsample, using a separate low discriminator LR —
#             this only starts once Phase 1 has converged, not from epoch 1.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 7: Training — Phase 1 (VAE only, KL warmup)")
print("=" * 60)

def kl_weight_at(epoch):
    if epoch < KL_START_EPOCH:
        return 0.0
    return min(KL_WEIGHT, (epoch - KL_START_EPOCH) / float(KL_WARMUP_EPOCHS))

def vae_forward_loss(X_enc_b, X_raw_b, sp_b, b_b, lib_b, ym_b, yh_b, kl_w):
    mu_q, logvar_q, z, mu_mouse, theta, pi_logits, mean_human = model(X_enc_b, sp_b, b_b, lib_b)
    mask_mouse, mask_human = (sp_b == 0), (sp_b == 1)
    recon = torch.tensor(0.0, device=device)
    if mask_mouse.any():
        recon = recon + zinb_nll(X_raw_b[mask_mouse], mu_mouse[mask_mouse],
                                  theta[mask_mouse], pi_logits[mask_mouse]).sum(dim=1).mean()
    if mask_human.any():
        recon = recon + F.mse_loss(mean_human[mask_human], X_enc_b[mask_human],
                                    reduction="none").sum(dim=1).mean()
    kl = -0.5 * torch.sum(1 + logvar_q - mu_q.pow(2) - logvar_q.exp(), dim=1).mean()
    align = LAMBDA_ALIGN * alignment_loss(z, sp_b, ym_b, yh_b, strong_pairs_encoded)
    loss = recon + kl_w * kl + align
    return loss, z, mu_q

history = {k: [] for k in ["train_loss", "val_loss", "val_domain_acc", "phase"]}
best_val_loss, best_state, best_epoch, patience_ct = float("inf"), None, 0, 0

for epoch in range(N_EPOCHS):
    kl_w = kl_weight_at(epoch)
    model.train()
    t_loss_sum, t_n = 0.0, 0
    for X_enc_b, X_raw_b, sp_b, b_b, lib_b, ym_b, yh_b in train_loader:
        X_enc_b, X_raw_b = X_enc_b.to(device), X_raw_b.to(device)
        sp_b, b_b, lib_b = sp_b.to(device), b_b.to(device), lib_b.to(device)
        ym_b, yh_b = ym_b.to(device), yh_b.to(device)

        opt_main.zero_grad()
        loss, z, mu_q = vae_forward_loss(X_enc_b, X_raw_b, sp_b, b_b, lib_b, ym_b, yh_b, kl_w)
        loss.backward()
        nn.utils.clip_grad_norm_(list(model.encoder.parameters()) + list(model.decoder.parameters()), 1.0)
        opt_main.step()
        t_loss_sum += loss.item() * len(X_enc_b); t_n += len(X_enc_b)

    train_loss = t_loss_sum / t_n

    model.eval()
    v_loss_sum, v_n = 0.0, 0
    with torch.no_grad():
        for X_enc_b, X_raw_b, sp_b, b_b, lib_b, ym_b, yh_b in val_loader:
            X_enc_b, X_raw_b = X_enc_b.to(device), X_raw_b.to(device)
            sp_b, b_b, lib_b = sp_b.to(device), b_b.to(device), lib_b.to(device)
            ym_b, yh_b = ym_b.to(device), yh_b.to(device)
            loss, _, _ = vae_forward_loss(X_enc_b, X_raw_b, sp_b, b_b, lib_b, ym_b, yh_b, kl_w)
            v_loss_sum += loss.item() * len(X_enc_b); v_n += len(X_enc_b)

    val_loss = v_loss_sum / v_n
    sched_main.step(val_loss)
    for k, v in zip(["train_loss","val_loss","val_domain_acc","phase"], [train_loss, val_loss, float("nan"), 1]):
        history[k].append(v)

    if (epoch+1) % 5 == 0 or epoch == 0:
        print(f"[Phase 1] Ep {epoch+1:>3}/{N_EPOCHS} | kl_w:{kl_w:.3f} | "
              f"Tr loss:{train_loss:.2f} | Va loss:{val_loss:.2f}")

    if val_loss < best_val_loss - MIN_DELTA:
        best_val_loss, best_epoch, patience_ct = val_loss, epoch+1, 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    else:
        patience_ct += 1
        if patience_ct >= PATIENCE:
            print(f"\nPhase 1 early stopping at epoch {epoch+1} (best epoch {best_epoch})")
            break

model.load_state_dict(best_state)
print(f"\nPhase 1 best val loss: {best_val_loss:.2f}  (epoch {best_epoch})")

# ── Phase 2: adversarial discriminator/generator alternation ──────────────
if USE_ADVERSARIAL:
    print("\n" + "=" * 60)
    print("STEP 7B: Training — Phase 2 (discriminator/generator alternation)")
    print("=" * 60)
    print(f"Discriminator LR: {DISC_LR}  |  balanced subsample cap: "
          f"{DISC_BALANCE_CAP}/species  |  max epochs: {PHASE2_MAX_EPOCHS}")

    idx_train_arr = np.array(idx_train)
    species_train = species_all[idx_train_arr]
    mouse_train_idx = idx_train_arr[species_train == 0]
    human_train_idx = idx_train_arr[species_train == 1]

    # Selection metric: |domain_acc - 0.5|, NOT val_loss_p2. The combined
    # generator loss conflates reconstruction quality with species-
    # invariance — a checkpoint can look "best" on that metric while
    # domain_acc is quietly climbing away from 0.5 (species becoming MORE
    # separable), which is the opposite of what Phase 2 is for. Track
    # reconstruction loss too (recon_only) purely as a sanity check that
    # the encoder isn't degrading catastrophically while chasing invariance.
    best_invariance_gap, best_state_p2, best_epoch_p2, patience_ct2 = float("inf"), None, 0, 0
    kl_w_final = kl_weight_at(len(history["train_loss"]) - 1)

    for epoch in range(PHASE2_MAX_EPOCHS):
        model.train()

        # Balanced per-species subsample, fresh each epoch — matches the
        # source's sub_index_batch logic (5000-cell cap per label).
        n_m = min(DISC_BALANCE_CAP, len(mouse_train_idx))
        n_h = min(DISC_BALANCE_CAP, len(human_train_idx))
        disc_idx = np.concatenate([
            rng.choice(mouse_train_idx, size=n_m, replace=False),
            rng.choice(human_train_idx, size=n_h, replace=False),
        ])
        rng.shuffle(disc_idx)
        disc_loader = make_loader(disc_idx, shuffle=False)

        # -- discriminator step: train domain_head on the MEAN embedding --
        for X_enc_b, X_raw_b, sp_b, b_b, lib_b, ym_b, yh_b in disc_loader:
            X_enc_b, sp_b, b_b = X_enc_b.to(device), sp_b.to(device), b_b.to(device)
            with torch.no_grad():
                mu_q = model.encode_mean(X_enc_b, sp_b, b_b)
            opt_disc.zero_grad()
            d_out = model.domain_head(mu_q)
            d_loss = F.cross_entropy(d_out, sp_b)
            d_loss.backward()
            opt_disc.step()

        # -- generator step: VAE loss minus discriminator loss, i.e. push
        #    the encoder to make species harder to tell apart, computed on
        #    the MEAN embedding (matching the source, not the sampled z) --
        for X_enc_b, X_raw_b, sp_b, b_b, lib_b, ym_b, yh_b in disc_loader:
            X_enc_b, X_raw_b = X_enc_b.to(device), X_raw_b.to(device)
            sp_b, b_b, lib_b = sp_b.to(device), b_b.to(device), lib_b.to(device)
            ym_b, yh_b = ym_b.to(device), yh_b.to(device)
            opt_main.zero_grad()
            loss, z, mu_q = vae_forward_loss(X_enc_b, X_raw_b, sp_b, b_b, lib_b, ym_b, yh_b, kl_w_final)
            d_out = model.domain_head(mu_q)
            d_loss = F.cross_entropy(d_out, sp_b)
            loss_generator = loss - d_loss
            loss_generator.backward()
            nn.utils.clip_grad_norm_(list(model.encoder.parameters()) + list(model.decoder.parameters()), 1.0)
            opt_main.step()

        model.eval()
        v_loss_sum, v_n, v_correct_d, v_n_d = 0.0, 0, 0, 0
        v_recon_sum = 0.0
        with torch.no_grad():
            for X_enc_b, X_raw_b, sp_b, b_b, lib_b, ym_b, yh_b in val_loader:
                X_enc_b, X_raw_b = X_enc_b.to(device), X_raw_b.to(device)
                sp_b, b_b, lib_b = sp_b.to(device), b_b.to(device), lib_b.to(device)
                ym_b, yh_b = ym_b.to(device), yh_b.to(device)
                loss, z, mu_q = vae_forward_loss(X_enc_b, X_raw_b, sp_b, b_b, lib_b, ym_b, yh_b, kl_w_final)
                d_out = model.domain_head(mu_q)
                loss_val = (loss - F.cross_entropy(d_out, sp_b)).item()
                v_loss_sum += loss_val * len(X_enc_b); v_n += len(X_enc_b)
                v_recon_sum += loss.item() * len(X_enc_b)  # loss here excludes -d_loss, i.e. the "raw" VAE loss
                v_correct_d += (d_out.argmax(1) == sp_b).sum().item(); v_n_d += len(sp_b)

        val_loss_p2 = v_loss_sum / v_n
        val_recon_p2 = v_recon_sum / v_n
        val_domain_acc = v_correct_d / v_n_d
        invariance_gap = abs(val_domain_acc - 0.5)
        for k, v in zip(["train_loss","val_loss","val_domain_acc","phase"],
                         [float("nan"), val_loss_p2, val_domain_acc, 2]):
            history[k].append(v)

        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"[Phase 2] Ep {epoch+1:>3}/{PHASE2_MAX_EPOCHS} | "
                  f"Va loss (gen):{val_loss_p2:.2f} | VAE-only loss:{val_recon_p2:.2f} | "
                  f"domain_acc:{val_domain_acc:.3f} (0.5=species-invariant, "
                  f"gap={invariance_gap:.3f})")

        # Guard against the encoder "winning" by degrading reconstruction
        # quality rather than genuinely mixing species — only accept a
        # checkpoint if the invariance gap improved AND VAE-only loss
        # hasn't blown up relative to where Phase 1 ended.
        recon_degraded = val_recon_p2 > best_val_loss * 1.15
        if invariance_gap < best_invariance_gap - MIN_DELTA and not recon_degraded:
            best_invariance_gap, best_epoch_p2, patience_ct2 = invariance_gap, epoch+1, 0
            best_state_p2 = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ct2 += 1
            if patience_ct2 >= PHASE2_PATIENCE:
                print(f"\nPhase 2 early stopping at epoch {epoch+1} (best epoch {best_epoch_p2}, "
                      f"gap={best_invariance_gap:.3f})")
                break

    if best_state_p2 is not None:
        model.load_state_dict(best_state_p2)
        print(f"\nPhase 2 best domain_acc gap: {best_invariance_gap:.3f}  (epoch {best_epoch_p2})")
    else:
        print("\nWARNING: Phase 2 never found a checkpoint that improved species-"
              "invariance without degrading reconstruction — keeping the Phase 1 "
              "model. Consider raising PHASE2_PATIENCE/PHASE2_MAX_EPOCHS, or "
              "check whether the discriminator is simply too weak to provide "
              "useful adversarial signal.")
    best_epoch = f"{best_epoch} (phase1) / {best_epoch_p2} (phase2)"

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
eps = range(1, len(history["train_loss"])+1)
phase_arr = np.array(history["phase"])
phase2_start = int(np.argmax(phase_arr == 2)) + 1 if (phase_arr == 2).any() else None

axes[0].plot(eps, history["train_loss"], label="Train (phase 1 only)")
axes[0].plot(eps, history["val_loss"], label="Val")
if phase2_start:
    axes[0].axvline(phase2_start, color="gray", linestyle=":", label="Phase 2 starts")
axes[0].set_title("Loss (phase 1: recon+KL+align, phase 2: minus discriminator)")
axes[0].legend(fontsize=8)

axes[1].plot(eps, history["val_domain_acc"], color="purple", label="Val domain acc (phase 2 only)")
axes[1].axhline(0.5, color="gray", linestyle="--")
if phase2_start:
    axes[1].axvline(phase2_start, color="gray", linestyle=":")
axes[1].set_title("Species-discriminator accuracy")
axes[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "training_curves.png"), dpi=150, bbox_inches="tight")
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# 8. EXTRACT z, PSEUDOBULK COSINE EVALUATION (same as prior scripts)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 8: Pseudobulk evaluation in latent space")
print("=" * 60)

model.eval()

def encode_all_own_batch(X_enc, species_arr, batch_arr, batch_size=4096):
    all_z = []
    X_t = torch.tensor(X_enc)
    sp_t = torch.tensor(species_arr, dtype=torch.long)
    b_t  = torch.tensor(batch_arr, dtype=torch.long)
    for i in range(0, len(X_t), batch_size):
        z = model.encode_mean(X_t[i:i+batch_size].to(device), sp_t[i:i+batch_size].to(device),
                               b_t[i:i+batch_size].to(device))
        all_z.append(z.cpu().numpy())
    return np.vstack(all_z)

mouse_batch_ids = y_batch_all[:n_mouse]
human_batch_ids = y_batch_all[n_mouse:]
Z_mouse = encode_all_own_batch(X_mouse_enc_input, np.zeros(n_mouse, dtype=np.int64), mouse_batch_ids)
Z_human = encode_all_own_batch(X_human_enc_input, np.ones(len(X_human_enc_input), dtype=np.int64), human_batch_ids)

np.save(os.path.join(SAVE_DIR, "Z_mouse_icebear.npy"), Z_mouse)
np.save(os.path.join(SAVE_DIR, "Z_human_icebear.npy"), Z_human)
np.save(os.path.join(SAVE_DIR, "mouse_region_labels.npy"), mouse_regions)
np.save(os.path.join(SAVE_DIR, "human_region_labels.npy"), human_regions)

# ══════════════════════════════════════════════════════════════════════════════
# 8B. JOINT UMAP — mouse + human cells plotted together, colored by species
#     and by region. This is the direct visual check of whether the
#     adversarial training actually mixed species in latent space, rather
#     than just trusting the val_domain_acc number.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 8B: Joint UMAP (mouse + human)")
print("=" * 60)

import anndata as ad
import scanpy as sc

Z_joint = np.concatenate([Z_mouse, Z_human], axis=0)
species_label_joint = np.array(["mouse"] * len(Z_mouse) + ["human"] * len(Z_human))
region_joint = np.concatenate([mouse_regions, human_regions])

# UMAP on the FULL cell set can be slow/heavy for large n — subsample per
# species/region for the plot only (does not affect any saved Z arrays or
# the pseudobulk similarity results above, which use all cells).
MAX_UMAP_CELLS_PER_GROUP = 20000
rng_umap = np.random.default_rng(0)
keep_idx = []
for sp in ["mouse", "human"]:
    sp_idx = np.where(species_label_joint == sp)[0]
    n_take = min(MAX_UMAP_CELLS_PER_GROUP, len(sp_idx))
    keep_idx.append(rng_umap.choice(sp_idx, size=n_take, replace=False))
keep_idx = np.concatenate(keep_idx)
print(f"UMAP subsample: {len(keep_idx):,} cells "
      f"(mouse={np.sum(species_label_joint[keep_idx]=='mouse'):,}, "
      f"human={np.sum(species_label_joint[keep_idx]=='human'):,})")

adata_joint = ad.AnnData(X=Z_joint[keep_idx].copy())
adata_joint.obs["species"] = species_label_joint[keep_idx]
adata_joint.obs["region"]  = region_joint[keep_idx]

sc.pp.neighbors(adata_joint, use_rep="X", n_pcs=None)
sc.tl.umap(adata_joint)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sc.pl.umap(adata_joint, color="species", ax=axes[0], show=False, title="Joint latent UMAP — species")
sc.pl.umap(adata_joint, color="region", ax=axes[1], show=False, title="Joint latent UMAP — region",
           legend_fontsize=6)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "umap_joint_latent.png"), dpi=150, bbox_inches="tight")
plt.show()
print("Saved: umap_joint_latent.png")
print("If species-mixing worked, the LEFT panel should show mouse and human")
print("cells interleaved, not forming two separate blobs. The RIGHT panel")
print("shows whether region identity is still visible as structure despite")
print("that mixing — ideally same-region mouse/human cells cluster near")
print("each other rather than region structure disappearing entirely.")

def pseudobulk(Z, labels):
    return {r: Z[labels == r].mean(axis=0) for r in np.unique(labels)}

pb_m, pb_h = pseudobulk(Z_mouse, mouse_regions), pseudobulk(Z_human, human_regions)
mouse_region_list, human_region_list = sorted(pb_m.keys()), sorted(pb_h.keys())
M_pb = np.stack([pb_m[r] for r in mouse_region_list])
H_pb = np.stack([pb_h[r] for r in human_region_list])

def cosine_sim_matrix(A, B):
    A_n = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B_n = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return A_n @ B_n.T

sim = cosine_sim_matrix(M_pb, H_pb)
df_sim = pd.DataFrame(sim, index=mouse_region_list, columns=human_region_list)
df_sim.to_csv(os.path.join(SAVE_DIR, "similarity_matrix_icebear.csv"))

fig, ax = plt.subplots(figsize=(0.6*len(human_region_list)+3, 0.5*len(mouse_region_list)+3))
sns.heatmap(df_sim, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            linewidths=0.5, linecolor="white", ax=ax)
ax.set_title("Mouse vs Human — Icebear-style cVAE latent cosine similarity")
ax.set_xlabel("Human h_region"); ax.set_ylabel("Mouse M_region")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "heatmap_icebear.png"), dpi=150, bbox_inches="tight")
plt.show()

expected_pairs_eval = {
    "M_MOp": (["H_M1C"],"strong"),
    "M_ORBl": (["H_A13"],"strong"), "M_ORBm": (["H_A14"],"strong"), "M_ORBvl": (["H_A13"],"strong"),
    "M_ILA": (["H_A25"],"moderate-strong"),
    "M_ACAd": (["H_A32","H_ACC"],"moderate"), "M_ACAv": (["H_ACC"],"moderate"),
    "M_PL": (["H_A32"],"moderate"),
    "M_RSPd": (["H_A29_A30"],"moderate"), "M_RSPv": (["H_A29_A30"],"moderate"),
    "M_RSPagl": (["H_A29_A30"],"moderate"),
    # M_FRP, M_MOs, M_AId, M_AIv, M_AIp: no established human counterpart
    # among the 10 curated cortex regions
}
rows = []
for m, (hs, conf) in expected_pairs_eval.items():
    if m not in mouse_region_list: continue
    for h in hs:
        if h not in human_region_list: continue
        rows.append({"M_region": m, "h_region": h, "confidence": conf, "cosine": df_sim.loc[m, h]})
homolog_df = pd.DataFrame(rows)
print("\nHomolog pair similarities (Icebear-style cVAE):")
print(homolog_df.to_string(index=False))
homolog_df.to_csv(os.path.join(SAVE_DIR, "homolog_pair_similarities_icebear.csv"), index=False)

diag_set = set(zip(homolog_df["M_region"], homolog_df["h_region"]))
all_vals = [{"cosine": df_sim.loc[m,h], "is_diag": (m,h) in diag_set}
            for m in mouse_region_list for h in human_region_list]
full_df = pd.DataFrame(all_vals)
diag_vals, offdiag_vals = full_df[full_df["is_diag"]]["cosine"], full_df[~full_df["is_diag"]]["cosine"]
if len(diag_vals) >= 2:
    t_stat, p_val = ttest_ind(diag_vals, offdiag_vals, equal_var=False)
    print(f"\nDiagonal mean={diag_vals.mean():.3f}  Off-diagonal mean={offdiag_vals.mean():.3f}")
    print(f"t={t_stat:.3f}  p={p_val:.4g}")

# ══════════════════════════════════════════════════════════════════════════════
# 8C. FOCUSED UMAP — top 3 highest-cosine homolog pairs only. The joint
#     UMAP in Step 8B mixes ALL regions together, which can hide whether
#     any ONE pair's mouse and human cells actually overlap in latent
#     space — a pair could show high pseudobulk cosine similarity while
#     the individual cells still form separate (but nearby) clusters.
#     This isolates just the top 3 pairs by cosine and checks overlap
#     directly, with the rest of the data shown only as light context.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 8C: Focused UMAP — top 3 homolog pairs")
print("=" * 60)

top3 = homolog_df.sort_values("cosine", ascending=False).head(3).reset_index(drop=True)
print("Top 3 pairs by cosine similarity:")
print(top3.to_string(index=False))

MAX_PER_GROUP = 8000
rng_top3 = np.random.default_rng(0)

pair_cells_Z, pair_cells_species, pair_cells_label, pair_cells_pairidx = [], [], [], []
for i, row in top3.iterrows():
    mr, hr = row["M_region"], row["h_region"]
    m_idx_pair = np.where(mouse_regions == mr)[0]
    h_idx_pair = np.where(human_regions == hr)[0]
    m_take = rng_top3.choice(m_idx_pair, size=min(MAX_PER_GROUP, len(m_idx_pair)), replace=False)
    h_take = rng_top3.choice(h_idx_pair, size=min(MAX_PER_GROUP, len(h_idx_pair)), replace=False)
    pair_cells_Z.append(Z_mouse[m_take]); pair_cells_species.append(np.full(len(m_take), "mouse"))
    pair_cells_label.append(np.full(len(m_take), f"{mr} / {hr}")); pair_cells_pairidx.append(np.full(len(m_take), i))
    pair_cells_Z.append(Z_human[h_take]); pair_cells_species.append(np.full(len(h_take), "human"))
    pair_cells_label.append(np.full(len(h_take), f"{mr} / {hr}")); pair_cells_pairidx.append(np.full(len(h_take), i))
    print(f"  {mr} / {hr}: {len(m_take):,} mouse cells, {len(h_take):,} human cells")

Z_top3      = np.concatenate(pair_cells_Z, axis=0)
species_top3 = np.concatenate(pair_cells_species)
label_top3   = np.concatenate(pair_cells_label)
pairidx_top3 = np.concatenate(pair_cells_pairidx)

adata_top3 = ad.AnnData(X=Z_top3.astype(np.float32))
adata_top3.obs["species"]  = species_top3
adata_top3.obs["pair"]     = label_top3
sc.pp.neighbors(adata_top3, use_rep="X", n_neighbors=15)
sc.tl.umap(adata_top3)

# Overview: species + pair, same style as Step 8B but restricted to just
# these 3 pairs' cells (cleaner signal, no dilution from other regions)
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sc.pl.umap(adata_top3, color="species", ax=axes[0], show=False,
           title="Top 3 homolog pairs — by species")
sc.pl.umap(adata_top3, color="pair", ax=axes[1], show=False,
           title="Top 3 homolog pairs — by pair identity", legend_fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "umap_top3_pairs_overview.png"), dpi=150, bbox_inches="tight")
plt.show()

# Per-pair detail: 1x3 grid, each panel highlights ONLY that pair's cells
# (mouse vs human, colored) with everything else shown as light gray
# background for spatial context — this is the direct "do these two
# clouds overlap" check per pair.
umap_coords = adata_top3.obsm["X_umap"]
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for i, row in top3.iterrows():
    ax = axes[i]
    mr, hr = row["M_region"], row["h_region"]
    bg_mask = pairidx_top3 != i
    ax.scatter(umap_coords[bg_mask, 0], umap_coords[bg_mask, 1],
               s=3, color="lightgray", alpha=0.3, label="_nolegend_")
    this_mask = pairidx_top3 == i
    mouse_mask = this_mask & (species_top3 == "mouse")
    human_mask = this_mask & (species_top3 == "human")
    ax.scatter(umap_coords[mouse_mask, 0], umap_coords[mouse_mask, 1],
               s=6, color="#5A9E8C", alpha=0.7, label=f"{mr} (mouse)")
    ax.scatter(umap_coords[human_mask, 0], umap_coords[human_mask, 1],
               s=6, color="#E07B60", alpha=0.7, label=f"{hr} (human)")
    ax.set_title(f"{mr} vs {hr}\ncosine={row['cosine']:.3f}", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
plt.suptitle("Per-pair overlap check — mouse vs. matched human cells "
             "(gray = other pairs' cells, for spatial context)", fontsize=12, y=1.03)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "umap_top3_pairs_detail.png"), dpi=150, bbox_inches="tight")
plt.show()
print("Saved: umap_top3_pairs_overview.png, umap_top3_pairs_detail.png")
print("In the detail panels: good alignment looks like the teal (mouse) and")
print("coral (human) points genuinely intermixed within each panel's own")
print("cluster. If they form two adjacent-but-separate blobs even for these")
print("top-cosine pairs, that means pseudobulk similarity is being driven by")
print("the two clusters pointing in a similar overall direction, not by")
print("individual cells actually landing in the same neighborhood — a real")
print("distinction worth knowing before treating the cosine numbers as proof")
print("of fine-grained alignment.")

# ══════════════════════════════════════════════════════════════════════════════
# 9. SAVE MODEL
# ══════════════════════════════════════════════════════════════════════════════
torch.save({"model_state_dict": model.state_dict(), "input_dim": input_dim,
            "latent_dim": LATENT_DIM, "n_batches": n_batches,
            "best_epoch": best_epoch, "best_val_loss": best_val_loss},
           os.path.join(SAVE_DIR, "icebear_model.pt"))
with open(os.path.join(SAVE_DIR, "label_encoders.pkl"), "wb") as f:
    pickle.dump({"le_mouse": le_mouse, "le_human": le_human, "le_batch": le_batch}, f)

print(f"\nAll outputs saved to: {SAVE_DIR}")
print("\nNOTE: gene panel still uses case-insensitive name matching, not the")
print("paper's proper one-to-one ortholog graph (percent-identity + ")
print("transitivity). That's the next highest-value upgrade if this result")
print("still looks weak — a mismatched \"orthologous\" gene pair would add")
print("noise to every downstream step regardless of model architecture.")