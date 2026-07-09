"""
=============================================================
UNSW-NB15 | Phase 4: Comprehensive Modelling
             Phase 5: Evaluation & Comparison
=============================================================
Files expected (same directory as this script):
  - UNSW-Processed-X.csv   → feature matrix
  - UNSW-Processed-Y.csv   → binary labels (0 = normal, 1 = intrusion)
  - UNSW-cleaned-step1.csv → (used optionally for EDA reference)

Outputs written to ./outputs/:
  - evaluation_report.csv          → accuracy / precision / recall / F1 per model
  - confusion_matrices.png         → grid of confusion matrices
  - feature_importance.png         → top-20 features (tree-based models)
  - roc_curves.png                 → ROC curve comparison
  - <ModelName>_model.pkl          → saved trained model (joblib)
=============================================================
"""

# ── 0. Imports ──────────────────────────────────────────────
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # non-interactive backend (safe for all envs)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from collections import Counter

# Preprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline


# Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans

# Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── 1. Configuration ─────────────────────────────────────────
X_FILE = "UNSW_Processed_X_selected.csv"
Y_FILE   = "UNSW_Processed_y.csv"
OUT_DIR  = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

TEST_SIZE     = 0.20          # 80/20 split
SMOTE_RATIO   = 0.5           # minority : majority after SMOTE (0.5 = 50%)
RANDOM_STATE  = 42

# ── 2. Load Data ─────────────────────────────────────────────
print("=" * 60)
print("  UNSW-NB15 — Phase 4 & 5 Pipeline")
print("=" * 60)

print("\n[1/7] Loading data...")
X = pd.read_csv(X_FILE)
y = pd.read_csv(Y_FILE)

# Flatten label column if it's a DataFrame
if isinstance(y, pd.DataFrame):
    y = y.iloc[:, 0]

print(f"      X shape : {X.shape}")
print(f"      y shape : {y.shape}")
print(f"      Class distribution (raw):\n{y.value_counts().to_string()}")

# ── 3. Basic Cleaning ────────────────────────────────────────
print("\n[2/7] Cleaning...")

# Drop columns that are pure identifiers or have zero variance
id_cols = [c for c in X.columns if c.lower() in ("id", "srcip", "dstip", "sport", "dsport")]
if id_cols:
    print(f"      Dropping identifier columns: {id_cols}")
    X.drop(columns=id_cols, inplace=True)

# Encode any remaining categorical / object columns
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
if cat_cols:
    print(f"      Label-encoding categorical columns: {cat_cols}")
    le = LabelEncoder()
    for col in cat_cols:
        X[col] = le.fit_transform(X[col].astype(str))

# Fill missing values with column median
if X.isnull().any().any():
    print(f"      Filling {X.isnull().sum().sum()} missing values with column median.")
    X.fillna(X.median(numeric_only=True), inplace=True)

# Ensure label is integer
y = y.astype(int)

print(f"      Final feature count : {X.shape[1]}")

# ── 4. Train / Test Split ────────────────────────────────────
print("\n[3/7] Splitting (80 / 20 stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"      Train : {X_train.shape[0]} rows | Test : {X_test.shape[0]} rows")


# ── 6. Feature Scaling (shared scaler) ──────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
feature_names = X.columns.tolist()

# ── 7. Model Definitions ─────────────────────────────────────
print("\n[5/7] Training models...")

MODELS = {
    "Decision Tree": DecisionTreeClassifier(
        max_depth=15, min_samples_split=10, random_state=RANDOM_STATE, class_weight="balanced"
    ),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=20, n_jobs=-1, random_state=RANDOM_STATE, class_weight="balanced"
    ),
    # LinearSVC + calibration wrapper → same concept as SVM but O(n) — handles 100K+ rows fine
    "SVM (Linear)": CalibratedClassifierCV(
        LinearSVC(C=1.0, max_iter=2000, random_state=RANDOM_STATE, class_weight="balanced"), cv=3
    ),
    "Logistic Regression": LogisticRegression(
        max_iter=1000, solver="saga", n_jobs=-1, random_state=RANDOM_STATE, class_weight="balanced"
    ),
    "Gradient Boosted Trees": GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1, random_state=RANDOM_STATE
    ),
}

# Models that need scaled input
NEEDS_SCALING = {"SVM (Linear)", "Logistic Regression", "Naive Bayes"}

results   = []
trained   = {}
roc_data  = {}

for name, model in MODELS.items():
    print(f"\n  ▸ {name}")
    Xtr = X_train_sc if name in NEEDS_SCALING else X_train.values
    Xte = X_test_sc  if name in NEEDS_SCALING else X_test.values

    model.fit(Xtr, y_train)
    y_pred  = model.predict(Xte)
    y_prob  = model.predict_proba(Xte)[:, 1] if hasattr(model, "predict_proba") else None

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_prob) if y_prob is not None else None

    print(f"    Accuracy  : {acc:.4f}")
    print(f"    Precision : {prec:.4f}")
    print(f"    Recall    : {rec:.4f}")
    print(f"    F1-Score  : {f1:.4f}")
    if auc: print(f"    ROC-AUC   : {auc:.4f}")

    results.append({
        "Model": name, "Accuracy": acc, "Precision": prec,
        "Recall": rec, "F1-Score": f1, "ROC-AUC": auc if auc else "N/A"
    })
    trained[name] = (model, y_pred, Xte)
    if y_prob is not None:
        roc_data[name] = (y_prob,)

    # Save model
    safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
    joblib.dump(model, os.path.join(OUT_DIR, f"{safe_name}_model.pkl"))

# ── 8. Save Evaluation Report ────────────────────────────────
# ── 8. Save Evaluation Reports (Including Power BI Exports) ──
print("\n[6/7] Saving evaluation reports...")

# 1. Standard Wide Report
df_results = pd.DataFrame(results).sort_values("F1-Score", ascending=False)
df_results.to_csv(os.path.join(OUT_DIR, "evaluation_report.csv"), index=False)

# 2. Power BI 'Melted' Metrics (Long Format)
# This turns columns (Accuracy, Recall, etc.) into a single 'Metric' column
df_melted = df_results.melt(
    id_vars=["Model"], 
    value_vars=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
    var_name="Metric", 
    value_name="Score"
)
df_melted.to_csv(os.path.join(OUT_DIR, "powerbi_metrics_long.csv"), index=False)
print("      ✓ powerbi_metrics_long.csv (For Model Showdown Dashboard)")

# 3. Power BI Feature Importance Export (Tree Models Only)
tree_models = {
    k: v[0] for k, v in trained.items()
    if hasattr(v[0], "feature_importances_")
}

if tree_models:
    fi_data = []
    for name, model in tree_models.items():
        for feat, imp in zip(feature_names, model.feature_importances_):
            fi_data.append({"Model": name, "Feature": feat, "Importance": imp})
    
    df_fi = pd.DataFrame(fi_data).sort_values(by=["Model", "Importance"], ascending=[True, False])
    df_fi.to_csv(os.path.join(OUT_DIR, "powerbi_feature_importances.csv"), index=False)
    print("      ✓ powerbi_feature_importances.csv (For Threat Drivers Dashboard)")

print("\n  ╔══════════════════════════════════════════════════════════════╗")
print("  ║                  MODEL COMPARISON SUMMARY                    ║")
print("  ╚══════════════════════════════════════════════════════════════╝")
print(df_results.to_string(index=False))

# ── 9. Plots ─────────────────────────────────────────────────
print("\n[7/7] Generating plots...")

# ── 9a. Confusion Matrices ───────────────────────────────────
n_models = len(MODELS)
ncols = 3
nrows = (n_models + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
axes = axes.flatten()

for idx, (name, (model, y_pred, _)) in enumerate(trained.items()):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx],
        xticklabels=["Normal", "Intrusion"],
        yticklabels=["Normal", "Intrusion"]
    )
    axes[idx].set_title(name, fontsize=13, fontweight="bold")
    axes[idx].set_xlabel("Predicted")
    axes[idx].set_ylabel("Actual")

for j in range(idx + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Confusion Matrices — All Models", fontsize=16, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "confusion_matrices.png"), dpi=150, bbox_inches="tight")
plt.close()
print("      ✓ confusion_matrices.png")

# ── 9b. Feature Importance ───────────────────────────────────
tree_models = {
    k: v[0] for k, v in trained.items()
    if hasattr(v[0], "feature_importances_")
}

if tree_models:
    n_tree = len(tree_models)
    fig, axes = plt.subplots(1, n_tree, figsize=(8 * n_tree, 8))
    if n_tree == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, tree_models.items()):
        importances = pd.Series(model.feature_importances_, index=feature_names)
        top20 = importances.nlargest(20).sort_values()
        top20.plot(kind="barh", ax=ax, color="steelblue")
        ax.set_title(f"Top 20 Features — {name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Importance Score")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "feature_importance.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("      ✓ feature_importance.png")

# ── 9c. ROC Curves ───────────────────────────────────────────
plt.figure(figsize=(10, 7))
colors = plt.cm.tab10.colors

for i, (name, (y_prob,)) in enumerate(roc_data.items()):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, color=colors[i % 10], lw=2, label=f"{name} (AUC={auc_val:.3f})")

plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Classifier")
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("ROC Curves — All Models", fontsize=14, fontweight="bold")
plt.legend(loc="lower right", fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "roc_curves.png"), dpi=150, bbox_inches="tight")
plt.close()
print("      ✓ roc_curves.png")

# ── 9d. Metric Bar Chart ─────────────────────────────────────
metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
df_plot = df_results.set_index("Model")[metrics]

ax = df_plot.plot(kind="bar", figsize=(14, 6), colormap="Set2", edgecolor="black", width=0.7)
plt.title("Model Performance Comparison", fontsize=14, fontweight="bold")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=20, ha="right")
plt.legend(loc="lower right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "model_comparison_bar.png"), dpi=150, bbox_inches="tight")
plt.close()
print("      ✓ model_comparison_bar.png")

# ── 10. K-Means Clustering (Phase 4 — Unsupervised) ──────────
print("\n── K-Means Clustering (unlabelled pattern exploration) ──")
kmeans = KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init=10)
cluster_labels = kmeans.fit_predict(X_test_sc)

# Cross-tabulate clusters vs true labels
ct = pd.crosstab(
    pd.Series(cluster_labels, name="Cluster"),
    pd.Series(y_test.values, name="True Label")
)
print(f"\n  Cluster vs True Label cross-tab:\n{ct.to_string()}")
ct.to_csv(os.path.join(OUT_DIR, "kmeans_cluster_crosstab.csv"))
print("  ✓ kmeans_cluster_crosstab.csv saved")

# ── 11. Best Model Summary ───────────────────────────────────
best = df_results.iloc[0]
print("\n" + "=" * 60)
print("  🏆  BEST MODEL (by F1-Score)")
print("=" * 60)
print(f"  Model     : {best['Model']}")
print(f"  Accuracy  : {best['Accuracy']:.4f}")
print(f"  Precision : {best['Precision']:.4f}")
print(f"  Recall    : {best['Recall']:.4f}")
print(f"  F1-Score  : {best['F1-Score']:.4f}")
print(f"  ROC-AUC   : {best['ROC-AUC']}")
print("=" * 60)
print(f"\n  All outputs saved to → ../{OUT_DIR}/")
print("  Files:")
for f in sorted(os.listdir(OUT_DIR)):
    print(f"    • {f}")
print("\n  Pipeline complete ✓")