import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# ==========================================
# STEP 1: LOAD & MERGE DATASETS
# ==========================================
base_path = 'C:/Users/Aryan/OneDrive - Manipal Academy of Higher Education/Documents/Academics/SEM-6/DM/LAB/Intrusion_proj/Intrusion_proj/data/UNSW_NB15_Archive/'

try:
    print("⏳ Loading datasets...")
    # Load both the Training and Testing sets
    df_train = pd.read_csv(base_path + 'UNSW_NB15_training-set.csv')
    df_test = pd.read_csv(base_path + 'UNSW_NB15_testing-set.csv')
    
    print(f"Training Set Shape: {df_train.shape}")
    print(f"Testing Set Shape:  {df_test.shape}")

    # MERGE THEM into one large dataset
    df = pd.concat([df_train, df_test], ignore_index=True)
    print(f"✅ Merged Dataset Shape: {df.shape}")

except FileNotFoundError:
    print("❌ Error: One or both files not found. Check file names in your folder.")
    exit() # Stop execution if files are missing

# ==========================================
# STEP 2: DATA CLEANING
# ==========================================
print("\n--- Starting Data Cleaning ---")

# 1. DROP IRRELEVANT COLUMNS
# The 'id' column acts as an index and confuses the model.
if 'id' in df.columns:
    df = df.drop(columns=['id'])
    print("✅ Dropped 'id' column.")

# 2. REMOVE DUPLICATES
initial_rows = df.shape[0]
df = df.drop_duplicates()
print(f"✅ Removed {initial_rows - df.shape[0]} duplicate rows.")

# 3. HANDLE MISSING VALUES
# Drop rows with strict missing values (NaN)
df = df.dropna()
print(f"Final Cleaned Data Shape: {df.shape}")

# ==========================================
# STEP 3: PREPROCESSING (ENCODING & SCALING)
# ==========================================
print("\n--- Starting Preprocessing ---")

# 1. SEPARATE FEATURES (X) AND TARGET (y)
# We want to predict 'label' (0=Normal, 1=Attack).
# We REMOVE 'attack_cat' because it is also a target (it tells us the attack type).
y = df['label']
X = df.drop(columns=['label', 'attack_cat']) 

# 2. ENCODE CATEGORICAL COLUMNS (Text -> Numbers)
# Identify columns that are text (e.g., proto, service, state)
cat_cols = X.select_dtypes(include=['object']).columns
print(f"Encoding Categorical Columns: {list(cat_cols)}")

le = LabelEncoder()
for col in cat_cols:
    # Convert all to string just in case mixed types exist
    X[col] = X[col].astype(str) 
    X[col] = le.fit_transform(X[col])

# 3. NORMALIZE NUMERICAL COLUMNS (Scale 0 to 1)
# This prevents large numbers (like bytes) from dominating small ones (like duration)
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# ==========================================
# STEP 4: SAVE & VERIFY
# ==========================================
print("\n✅ Data Successfully Processed!")
print("First 5 rows of the final processed features:")
print(X_scaled.head())

# Save the processed X and y for the next steps
# We save them separately to make modeling easier later
X_scaled.to_csv("UNSW_Processed_X.csv", index=False)
y.to_csv("UNSW_Processed_y.csv", index=False)
print("\n✅ Saved processed data to 'UNSW_Processed_X.csv' and 'UNSW_Processed_y.csv'")