"""
=============================================================
UNSW-NB15 | Phase 3: Feature Selection
=============================================================
Input files (same directory as this script):
  - UNSW_Processed_X.csv   → full feature matrix from etl.py
  - UNSW_Processed_y.csv   → binary labels (0 = normal, 1 = intrusion)

Output files:
  - UNSW_Processed_X_selected.csv  → reduced feature matrix (top K features)
  - outputs/feature_scores.csv     → full ranking of all features
  - outputs/feature_selection.png  → bar chart of top 20 features

Run this AFTER etl.py and BEFORE unsw_models.py.
=============================================================
"""
 
# ── 0. Imports ───────────────────────────────────────────────
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── 1. Configuration ─────────────────────────────────────────
X_FILE    = "UNSW_Processed_X.csv"
Y_FILE    = "UNSW_Processed_y.csv"
OUT_DIR   = "outputs"
TOP_K     = 20        # number of features to keep — change if needed

os.makedirs(OUT_DIR, exist_ok=True)

# ── 2. Load Data ─────────────────────────────────────────────
print("=" * 60)
print("  UNSW-NB15 — Phase 3: Feature Selection")
print("=" * 60)

print("\n[1/5] Loading processed data...")
X = pd.read_csv(X_FILE)
y = pd.read_csv(Y_FILE)

if isinstance(y, pd.DataFrame):
    y = y.iloc[:, 0]
y = y.astype(int)

print(f"      X shape : {X.shape}")
print(f"      y shape : {y.shape}")
print(f"      Features : {list(X.columns)}")

# ── 3. Method 1 — Mutual Information (SelectKBest) ──────────
# Mutual information measures how much knowing a feature reduces
# uncertainty about the label. Works well for both linear and
# non-linear relationships. Best for classification tasks.
print("\n[2/5] Running Mutual Information (SelectKBest)...")

selector_mi = SelectKBest(score_func=mutual_info_classif, k="all")
selector_mi.fit(X, y)

mi_scores = pd.Series(selector_mi.scores_, index=X.columns, name="MI_Score")
mi_scores = mi_scores.sort_values(ascending=False)

print(f"\n      Top 10 features by Mutual Information:")
print(mi_scores.head(10).to_string())

# ── 4. Method 2 — Random Forest Feature Importance ───────────
# Tree-based importance measures how much each feature reduces
# impurity across all trees. Robust and handles non-linearity well.
# Good cross-check against Mutual Information.
print("\n[3/5] Running Random Forest Feature Importance...")

rf = RandomForestClassifier(
    n_estimators=100,   # lighter than the modeling phase — just for ranking
    max_depth=10,
    n_jobs=-1,
    random_state=42
)
rf.fit(X, y)

rf_scores = pd.Series(rf.feature_importances_, index=X.columns, name="RF_Importance")
rf_scores = rf_scores.sort_values(ascending=False)

print(f"\n      Top 10 features by Random Forest Importance:")
print(rf_scores.head(10).to_string())

# ── 5. Combine Both Rankings ─────────────────────────────────
# Normalise both scores to 0–1, then average them.
# Features that rank highly in BOTH methods are most reliable.
print("\n[4/5] Combining rankings...")

mi_norm = (mi_scores - mi_scores.min()) / (mi_scores.max() - mi_scores.min())
rf_norm = (rf_scores - rf_scores.min()) / (rf_scores.max() - rf_scores.min())

combined = pd.DataFrame({
    "MI_Score_Raw"    : mi_scores,
    "RF_Score_Raw"    : rf_scores,
    "MI_Normalised"   : mi_norm,
    "RF_Normalised"   : rf_norm,
})

combined["Combined_Score"] = (combined["MI_Normalised"] + combined["RF_Normalised"]) / 2
combined = combined.sort_values("Combined_Score", ascending=False)

print(f"\n  Full feature ranking (top 20):")
print(combined.head(20)[["MI_Score_Raw", "RF_Score_Raw", "Combined_Score"]].to_string())

# Save full ranking
combined.to_csv(os.path.join(OUT_DIR, "feature_scores.csv"))
print(f"\n  ✓ Full ranking saved → {OUT_DIR}/feature_scores.csv")

# ── 6. Select Top K Features ─────────────────────────────────
top_features = combined.head(TOP_K).index.tolist()

print(f"\n  Selected top {TOP_K} features:")
for i, feat in enumerate(top_features, 1):
    score = combined.loc[feat, "Combined_Score"]
    print(f"    {i:2}. {feat:<35} (score: {score:.4f})")

X_selected = X[top_features]
X_selected.to_csv("UNSW_Processed_X_selected.csv", index=False)
print(f"\n  ✓ Reduced feature matrix saved → UNSW_Processed_X_selected.csv")
print(f"    Shape : {X_selected.shape}  (was {X.shape})")

# ── 7. Plot — Top 20 Feature Comparison ─────────────────────
print("\n[5/5] Generating feature selection plot...")

top20 = combined.head(20).copy()

fig, axes = plt.subplots(1, 3, figsize=(20, 8))

# Plot 1 — Mutual Information
top20_mi = top20["MI_Normalised"].sort_values()
axes[0].barh(top20_mi.index, top20_mi.values, color="#4C72B0", edgecolor="black")
axes[0].set_title("Mutual Information\n(Normalised)", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Score")
axes[0].tick_params(axis="y", labelsize=9)

# Plot 2 — Random Forest Importance
top20_rf = top20["RF_Normalised"].sort_values()
axes[1].barh(top20_rf.index, top20_rf.values, color="#55A868", edgecolor="black")
axes[1].set_title("Random Forest Importance\n(Normalised)", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Score")
axes[1].tick_params(axis="y", labelsize=9)

# Plot 3 — Combined Score
top20_comb = top20["Combined_Score"].sort_values()
axes[2].barh(top20_comb.index, top20_comb.values, color="#C44E52", edgecolor="black")
axes[2].set_title(f"Combined Score\n(Top {TOP_K} Selected)", fontsize=13, fontweight="bold")
axes[2].set_xlabel("Score")
axes[2].tick_params(axis="y", labelsize=9)

plt.suptitle(
    "Phase 3 — Feature Selection: UNSW-NB15",
    fontsize=16, fontweight="bold", y=1.01
)
plt.tight_layout()
plt.savefig(
    os.path.join(OUT_DIR, "feature_selection.png"),
    dpi=150, bbox_inches="tight"
)
plt.close()
print(f"  ✓ Plot saved → {OUT_DIR}/feature_selection.png")

# ── 8. Summary ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Phase 3 Complete — Summary")
print("=" * 60)
print(f"  Total features in   : {X.shape[1]}")
print(f"  Total features out  : {TOP_K}")
print(f"  Reduction           : {X.shape[1] - TOP_K} features dropped "
      f"({(X.shape[1] - TOP_K) / X.shape[1] * 100:.1f}%)")
print(f"\n  Next step → run unsw_models.py")
print("  (it will use UNSW_Processed_X_selected.csv automatically)")
print("=" * 60)