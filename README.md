# ML-Driven-Network-Intrusion-Detection

An end-to-end automated data processing and machine learning pipeline for network intrusion detection. Designed for research-driven evaluation of classification models on the UNSW-NB15 dataset, complete with feature engineering and Power BI dashboard integration. 

## Pipeline Architecture

* **Phase 1 & 2: ETL & Preprocessing (`etl.py`)**
  * Ingests and merges training and testing datasets.
  * Cleans structural anomalies, drops redundant identifiers, and handles missing values.
  * Applies Label Encoding for categorical variables and Min-Max scaling for numerical consistency.
* **Phase 3: Automated Feature Selection (`feature_selection.py`)**
  * Implements a dual-method ranking system: Mutual Information (SelectKBest) and Random Forest Feature Importance.
  * Normalizes and aggregates scores to extract the top 20 most reliable predictive features.
  * Auto-generates comparative visualizations for feature distributions.
* **Phase 4 & 5: Comprehensive Modeling & Evaluation (`unsw_models.py`)**
  * Trains a diverse suite of classifiers: Decision Tree, Naive Bayes, Random Forest, Linear SVC, Logistic Regression, and Gradient Boosted Trees.
  * Evaluates performance utilizing Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
  * Integrates an unsupervised K-Means clustering step for unlabelled pattern exploration.
  * Exports analytical metrics and "melted" data structures explicitly formatted for Business Intelligence ingestion.

## Repository Structure

* `etl.py`: Core data ingestion and standardization script.
* `feature_selection.py`: Dimensionality reduction and feature extraction script.
* `unsw_models.py`: Model training, evaluation, and artifact generation script.
* `DM_LAB_PROJ.pbix`: Power BI dashboard utilizing the exported metrics for interactive model showdowns and threat driver analysis.
* `outputs/`: Auto-generated directory containing performance reports (`.csv`), trained model artifacts (`.pkl`), and visualization grids (`.png`).

## Tech Stack

* **Systems & Automation:** Python, Pandas, NumPy
* **Technical Research & ML:** Scikit-Learn, Joblib
* **Data Visualization:** Matplotlib, Seaborn, Power BI

## Execution Steps

Ensure the standard UNSW-NB15 dataset CSV files (`UNSW_NB15_training-set.csv`, `UNSW_NB15_testing-set.csv`) are placed in your configured local data directory before running the pipeline.

**1. Run the ETL Pipeline:**
```bash
python etl.py
```

**2. Extract Key Features:**
```bash
python feature_selection.py
```

**3. Train Models and Generate BI Data:**
```bash
python unsw_models.py
```

**4. Visualize Insights:** Open `DM_LAB_PROJ.pbix` in Power BI Desktop. Refresh the data connections to point to the newly generated `outputs/powerbi_metrics_long.csv` and `outputs/powerbi_feature_importances.csv` files to interact with the visualizations.
