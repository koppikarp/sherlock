import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, roc_curve
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import re

# ── 1. Load data ────────────────────────────────────────────────────────────

results = pd.read_csv("SubCellPortable/result_head.csv")
hpa = pd.read_csv("data/groundtruth/proteinatlas_filtered.csv")  # has Gene, is_multi_localized

# ── 2. Parse gene name from id ──────────────────────────────────────────────
# SubCellPortable sets id from your output_prefix.
# If you used f"{gene}_{img_id}_" as the prefix, extract gene name here.
# Adjust the regex to match whatever prefix you used.

def extract_gene(id_str):
    # Example: "SCYL3_10_A1_001_001_" → "SCYL3"
    # Modify this to match your actual prefix format
    return id_str.split("_")[0]

results["Gene"] = results["id"].apply(extract_gene)

# ── 3. Identify column groups ────────────────────────────────────────────────

feat_cols = [c for c in results.columns if c.startswith("feat")]
prob_cols = [c for c in results.columns if c.startswith("prob")]

print(f"Embedding dims: {len(feat_cols)}")   # expect 1536
print(f"Probability classes: {len(prob_cols)}")  # expect 31

# ── 4. Classifier-based multi-localization signals (per cell) ────────────────

probs = results[prob_cols].values.astype(np.float32)

# Clip to avoid log(0)
probs_safe = np.clip(probs, 1e-10, 1.0)

results["entropy"]    = -np.sum(probs_safe * np.log(probs_safe), axis=1)
results["one_minus_max"] = 1.0 - probs.max(axis=1)
results["n_classes_01"] = (probs > 0.10).sum(axis=1)  # threshold 0.10
results["n_classes_02"] = (probs > 0.20).sum(axis=1)  # threshold 0.20

# ── 5. Aggregate per protein ─────────────────────────────────────────────────

def twonn_lid(X):
    """TwoNN LID estimator. Returns NaN if fewer than 3 cells."""
    if len(X) < 3:
        return np.nan
    nbrs = NearestNeighbors(n_neighbors=3).fit(X)
    distances, _ = nbrs.kneighbors(X)
    r1 = distances[:, 1]
    r2 = distances[:, 2]
    valid = (r1 > 0) & (r2 > 0)
    if valid.sum() < 2:
        return np.nan
    return 1.0 / np.mean(np.log(r2[valid] / r1[valid]))

protein_records = []

for gene, group in results.groupby("Gene"):
    n_cells = len(group)
    if n_cells < 3:
        continue  # LID unreliable below 3 points

    embs = group[feat_cols].values.astype(np.float32)
    lid  = twonn_lid(embs)

    protein_records.append({
        "Gene":          gene,
        "LID":           lid,
        "n_cells":       n_cells,
        # Aggregate classifier signals as mean across cells for this protein
        "entropy_mean":      group["entropy"].mean(),
        "one_minus_max_mean": group["one_minus_max"].mean(),
        "n_classes_01_mean": group["n_classes_01"].mean(),
        "n_classes_02_mean": group["n_classes_02"].mean(),
        # Also keep max entropy (single-cell peak signal)
        "entropy_max":   group["entropy"].max(),
    })

protein_df = pd.DataFrame(protein_records)

# ── 6. Join with HPA ground truth ────────────────────────────────────────────

protein_df = protein_df.merge(hpa[["Gene", "is_multi_localized"]], 
                               on="Gene", how="inner")
protein_df = protein_df.dropna(subset=["LID"])

print(f"\nProteins with LID computed: {len(protein_df)}")
print(f"  Mono-localized: {(protein_df.is_multi_localized==0).sum()}")
print(f"  Multi-localized: {(protein_df.is_multi_localized==1).sum()}")

# ── 7. Statistics ─────────────────────────────────────────────────────────────

mono  = protein_df[protein_df.is_multi_localized == 0]["LID"]
multi = protein_df[protein_df.is_multi_localized == 1]["LID"]

stat, p = stats.mannwhitneyu(multi, mono, alternative="greater")
print(f"\nMann-Whitney U (LID multi > mono):")
print(f"  p = {p:.4f}  |  multi median = {multi.median():.3f}  |  mono median = {mono.median():.3f}")

y_true = protein_df["is_multi_localized"].values

def safe_auroc(y_true, scores):
    try:
        return roc_auc_score(y_true, scores)
    except Exception:
        return np.nan

aurocs = {
    "LID":            safe_auroc(y_true, protein_df["LID"]),
    "Entropy (mean)": safe_auroc(y_true, protein_df["entropy_mean"]),
    "1−max_prob":     safe_auroc(y_true, protein_df["one_minus_max_mean"]),
    "#classes>0.1":   safe_auroc(y_true, protein_df["n_classes_01_mean"]),
    "#classes>0.2":   safe_auroc(y_true, protein_df["n_classes_02_mean"]),
}

print("\nAUROC scores (predicting multi-localization):")
for name, auc in aurocs.items():
    print(f"  {name:<20}: {auc:.3f}")

# ── 8. Plots ──────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(15, 5))
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

# — Plot 1: LID distribution mono vs multi —
ax1 = fig.add_subplot(gs[0])
ax1.hist(mono,  bins=25, alpha=0.65, density=True, color="#4878CF", label=f"Mono (n={len(mono)})")
ax1.hist(multi, bins=25, alpha=0.65, density=True, color="#E87D3E", label=f"Multi (n={len(multi)})")
ax1.set_xlabel("LID (per-protein)", fontsize=11)
ax1.set_ylabel("Density", fontsize=11)
ax1.set_title(f"LID distribution\np = {p:.3g} (Mann-Whitney)", fontsize=11)
ax1.legend(fontsize=9)

# — Plot 2: ROC curves for all metrics —
ax2 = fig.add_subplot(gs[1])
colors = ["#E87D3E", "#4878CF", "#6ACC65", "#D65F5F", "#B47CC7"]
metrics = [
    ("LID",            protein_df["LID"]),
    ("Entropy (mean)", protein_df["entropy_mean"]),
    ("1−max_prob",     protein_df["one_minus_max_mean"]),
    ("#classes>0.1",   protein_df["n_classes_01_mean"]),
    ("#classes>0.2",   protein_df["n_classes_02_mean"]),
]
for (name, scores), color in zip(metrics, colors):
    fpr, tpr, _ = roc_curve(y_true, scores)
    auc = aurocs[name]
    ax2.plot(fpr, tpr, color=color, lw=2, label=f"{name} ({auc:.2f})")
ax2.plot([0,1],[0,1],"k--", lw=1)
ax2.set_xlabel("False Positive Rate", fontsize=11)
ax2.set_ylabel("True Positive Rate", fontsize=11)
ax2.set_title("ROC: predicting multi-localization", fontsize=11)
ax2.legend(fontsize=8, loc="lower right")

# — Plot 3: LID vs entropy scatter, coloured by ground truth —
ax3 = fig.add_subplot(gs[2])
colors_gt = protein_df["is_multi_localized"].map({0: "#4878CF", 1: "#E87D3E"})
ax3.scatter(protein_df["entropy_mean"], protein_df["LID"],
            c=colors_gt, alpha=0.6, s=20, edgecolors="none")
ax3.set_xlabel("Classifier entropy (mean over cells)", fontsize=11)
ax3.set_ylabel("LID", fontsize=11)
ax3.set_title("LID vs classifier entropy\n(orange=multi, blue=mono)", fontsize=11)

# Add a simple correlation note
r, rp = stats.spearmanr(protein_df["entropy_mean"], protein_df["LID"])
ax3.text(0.05, 0.95, f"Spearman ρ = {r:.2f}", transform=ax3.transAxes,
         fontsize=9, verticalalignment="top")

plt.suptitle("Preliminary validation: LID vs classifier confidence for multi-localization",
             fontsize=12, y=1.02)
plt.savefig("preliminary_validation.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nPlot saved to preliminary_validation.png")
protein_df[["Gene","n_cells","LID","entropy_mean","one_minus_max_mean",
            "n_classes_01_mean","is_multi_localized"]].to_csv("protein_lid_results.csv", index=False)