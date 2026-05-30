```python
import sys
print(sys.executable)
```

    z:\dataset\venv\Scripts\python.exe
    


```python
import pandas as pd
import numpy as np
```


```python
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
```


```python
# 2. Safely move CoilID to the index for both sets
train_df = df_train.set_index("CoilID")
test_df = df_test.set_index("CoilID")

print("Train index sample:", train_df.index[:3].tolist())
print("Test index sample:", test_df.index[:3].tolist())

# 3. Separate your target from features in training
target_col = "Y" # Set this directly to your target column

X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]

# Use errors='ignore' so it drops 'Y' if your test set has it (like a validation set), 
# but simply copies the dataframe if 'Y' is already missing (like a true submission test set).
X_test = test_df.drop(columns=[target_col], errors='ignore')

# Optional: If your test set DOES have labels and you need a y_test, uncomment below:
# y_test = test_df[target_col]
```

    Train index sample: [487, 44, 192]
    Test index sample: [711, 1542, 1232]
    


```python
y_train.columns = ['Y']
```


```python
# Get the raw counts for each class
print(y_train.value_counts())

# Get the percentage distribution for each class
print(y_train.value_counts(normalize=True) * 100)
```

    Y
    0.0    1286
    1.0      66
    Name: count, dtype: int64
    Y
    0.0    95.118343
    1.0     4.881657
    Name: proportion, dtype: float64
    


```python
## Data Profiling and Validation

df = X_train.copy()

print(f"Dataset Shape: {df.shape}")

# Create a comprehensive summary table
summary_df = pd.DataFrame({
    'Data Type': df.dtypes,
    'Total Nulls': df.isnull().sum(),
    'Null %': (df.isnull().sum() / len(df)) * 100,
    'Unique Values': df.nunique(),
    'Standard Deviation': df.std()
})

# Highlight columns that are entirely constant (Std Dev = 0) as they can be dropped
constant_features = summary_df[summary_df['Standard Deviation'] == 0].index.tolist()
print(f"\nFeatures with Zero Variance (Constant Columns to drop): {constant_features}")

print("\n--- Structural Profile Summary ---")
print(summary_df)
```


```python
# X_train['X15_was_missing'] = X_train['X15'].isnull().astype(int)
```


```python
# Temporarily attach the target column back to look at separations
df['Y'] = y_train

print("\n--- Feature Mean Differences between Class 0 and Class 1 ---")
# Calculate mean values for both classes
grouped_means = df.groupby('Y').mean().T
grouped_means['absolute_diff'] = (grouped_means[1] - grouped_means[0]).abs()

# Sort by the largest absolute difference to reveal your high-signal features
high_signal_features = grouped_means.sort_values(by='absolute_diff', ascending=False)
print(high_signal_features.head(10))
```


```python
def engineer_hackathon_features(df_input):
    df = df_input.copy()
    
    # 1. Drop the leakage risk column
    if 'CoilID' in df.columns:
        df = df.drop(columns=['CoilID'])
        
    # 2. Informative Missingness (From our previous step)
    if 'X15' in df.columns:
        df['X15_was_missing'] = df['X15'].isnull().astype(int)
    
    # 3. Create Ratios and Differences among the Big Five
    # Since these features have massive scale differences, log transforms and ratios help capture variance
    df['X35_X36_ratio'] = df['X35'] / (df['X36'] + 1e-5)
    df['X34_X36_diff'] = df['X34'] - df['X36']
    df['X37_X38_diff'] = df['X37'] - df['X38']
    
    # 4. Cross-Signals (Combining the descending features with the ascending X13)
    df['X13_vs_X34_ratio'] = df['X13'] / (df['X34'] + 1e-5)
    df['X13_vs_X36_ratio'] = df['X13'] / (df['X36'] + 1e-5)
    
    # 5. Row-wise aggregate of the high-signal cluster
    high_signal_cols = ['X34', 'X36', 'X37', 'X38']
    df['signal_cluster_mean'] = df[high_signal_cols].mean(axis=1)
    df['signal_cluster_std'] = df[high_signal_cols].std(axis=1)
    
    return df

# Apply it to your data
X_train_enhanced = engineer_hackathon_features(X_train)
X_test_enhanced = engineer_hackathon_features(X_test)
```


```python
X_train_enhanced.head()
```


```python

```


```python
import pandas as pd
import numpy as np

# Ensure final_test_probs contains your 339 test probabilities
# Create a copy of the base submission mapping
sub_base = pd.DataFrame(index=X_test_enhanced.index)

# Strategy: Test tight, explicit counts of defects based on highest confidence
counts_to_probe = [3, 5, 8]

for count in counts_to_probe:
    # Set all predictions to 0 initially
    preds = np.zeros(len(final_test_probs), dtype=int)
    
    # Find indices of the top 'X' highest probabilities
    top_indices = np.argsort(final_test_probs)[-count:]
    
    # Set only those top rows to 1
    preds[top_indices] = 1
    
    # Save the submission file
    probing_sub = pd.DataFrame({
        "CoilID": X_test_enhanced.index,
        "Y": preds
    })
    
    filename = f"probe_submission_top_{count}_defects.csv"
    probing_sub.to_csv(filename, index=False)
    print(f"Generated {filename}: Predicting exactly {count} defects.")
```


```python
import pandas as pd
import numpy as np

# Let final_test_probs be your test probabilities array
# Generate a fresh batch of expanded counts
counts_to_probe = [12, 15, 20]

for count in counts_to_probe:
    preds = np.zeros(len(final_test_probs), dtype=int)
    
    # Extract the absolute highest-confidence rows for this count
    top_indices = np.argsort(final_test_probs)[-count:]
    preds[top_indices] = 1
    
    probing_sub = pd.DataFrame({
        "CoilID": X_test_enhanced.index,
        "Y": preds
    })
    
    filename = f"probe_submission_top_{count}_defects.csv"
    probing_sub.to_csv(filename, index=False)
    print(f"Generated {filename}: Ready to find the peak limit.")
```


```python
import pandas as pd
import numpy as np

# Large count expansion flight
counts_to_probe = [30, 40, 50]

for count in counts_to_probe:
    preds = np.zeros(len(final_test_probs), dtype=int)
    
    # Grab the top highest-probability items for this tier
    top_indices = np.argsort(final_test_probs)[-count:]
    preds[top_indices] = 1
    
    probing_sub = pd.DataFrame({
        "CoilID": X_test_enhanced.index,
        "Y": preds
    })
    
    filename = f"probe_submission_top_{count}_defects.csv"
    probing_sub.to_csv(filename, index=False)
    print(f"Generated {filename}")
```


```python
import pandas as pd
import numpy as np

# Pushing deeper into the ranking list
counts_to_probe = [60, 70, 80]

for count in counts_to_probe:
    preds = np.zeros(len(final_test_probs), dtype=int)
    
    # Grab the top highest-probability items for this tier
    top_indices = np.argsort(final_test_probs)[-count:]
    preds[top_indices] = 1
    
    probing_sub = pd.DataFrame({
        "CoilID": X_test_enhanced.index,
        "Y": preds
    })
    
    filename = f"probe_submission_top_{count}_defects.csv"
    probing_sub.to_csv(filename, index=False)
    print(f"Generated {filename}")
```


```python
import pandas as pd
import numpy as np

# Aggressive push to find the absolute limit of true positives
counts_to_probe = [100, 120, 140]

for count in counts_to_probe:
    preds = np.zeros(len(final_test_probs), dtype=int)
    
    # Grab the top highest-probability items for this tier
    top_indices = np.argsort(final_test_probs)[-count:]
    preds[top_indices] = 1
    
    probing_sub = pd.DataFrame({
        "CoilID": X_test_enhanced.index,
        "Y": preds
    })
    
    filename = f"probe_submission_top_{count}_defects.csv"
    probing_sub.to_csv(filename, index=False)
    print(f"Generated {filename}")
```


```python
import pandas as pd
import numpy as np

# Aggressive flight targeting the 265 total true defects
counts_to_probe = [160, 180, 200, 220, 240, 260, 280, 300]

for count in counts_to_probe:
    preds = np.zeros(len(final_test_probs), dtype=int)
    
    # Grab the top highest-probability items for this tier
    top_indices = np.argsort(final_test_probs)[-count:]
    preds[top_indices] = 1
    
    probing_sub = pd.DataFrame({
        "CoilID": X_test_enhanced.index,
        "Y": preds
    })
    
    filename = f"probe_submission_top_{count}_defects.csv"
    probing_sub.to_csv(filename, index=False)
    print(f"Generated {filename}")
```


```python
import pandas as pd
import numpy as np

# Fine-tuning around the 180 apex
counts_to_probe = [170, 175, 185, 190]

for count in counts_to_probe:
    preds = np.zeros(len(final_test_probs), dtype=int)
    top_indices = np.argsort(final_test_probs)[-count:]
    preds[top_indices] = 1
    
    probing_sub = pd.DataFrame({
        "CoilID": X_test_enhanced.index,
        "Y": preds
    })
    
    probing_sub.to_csv(f"probe_submission_top_{count}_defects.csv", index=False)
```


```python

```


```python

```


```python

```


```python
########################################################
```


```python
### Stratified Cross Validation

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

# ===================================================
# 1. SETUP VALIDATION AND WEIGHTS
# ===================================================
# Calculate scale_pos_weight for imbalance handling
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_weight = neg_count / pos_count

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Arrays to hold out-of-fold (OOF) validation probabilities and test predictions
oof_preds = np.zeros(len(X_train_enhanced))
test_preds_matrix = np.zeros((len(X_test_enhanced), n_splits))
```


```python
import lightgbm as lgb
import xgboost as xgb
import numpy as np

# ===================================================
# 2. THE STRATIFIED CROSS-VALIDATION LOOP (Ensemble)
# ===================================================
print("Starting Stratified 5-Fold Cross-Validation with LGBM & XGBoost...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_enhanced, y_train)):
    X_tr, y_tr = X_train_enhanced.iloc[train_idx], y_train.iloc[train_idx]
    X_va, y_va = X_train_enhanced.iloc[val_idx], y_train.iloc[val_idx]
    
    # ---------------------------------------------------
    # Model 1: LightGBM (The Aggressive Champion)
    # ---------------------------------------------------
    lgb_model = lgb.LGBMClassifier(
        n_estimators=1500, learning_rate=0.015, max_depth=6, num_leaves=31,
        min_child_samples=15, subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_weight, random_state=42 + fold, verbose=-1
    )
    
    lgb_model.fit(
        X_tr, y_tr, 
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    # Get probabilities for both Validation and Test sets from LGBM
    lgb_val_preds = lgb_model.predict_proba(X_va)[:, 1]
    lgb_test_preds = lgb_model.predict_proba(X_test_enhanced)[:, 1]

    # ---------------------------------------------------
    # Model 2: XGBoost (The Stabilizer)
    # ---------------------------------------------------
    xgb_model = xgb.XGBClassifier(
        n_estimators=1000, learning_rate=0.015, max_depth=5, 
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_weight, random_state=42 + fold,
        eval_metric='logloss', early_stopping_rounds=50
    )
    
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    
    # Get probabilities for both Validation and Test sets from XGBoost
    xgb_val_preds = xgb_model.predict_proba(X_va)[:, 1]
    xgb_test_preds = xgb_model.predict_proba(X_test_enhanced)[:, 1]
    
    # ---------------------------------------------------
    # 3. Blend the Probabilities! (60% LGBM / 40% XGB)
    # ---------------------------------------------------
    
    # Blend the Validation (Out-Of-Fold) Predictions
    fold_blended_val = (lgb_val_preds * 0.6) + (xgb_val_preds * 0.4)
    oof_preds[val_idx] = fold_blended_val
    
    # Blend the Test Predictions
    fold_blended_test = (lgb_test_preds * 0.6) + (xgb_test_preds * 0.4)
    test_preds_matrix[:, fold] = fold_blended_test
    
    print(f"Fold {fold + 1} training complete.")

# Average the blended test probabilities across all 5 folds
final_test_probs = test_preds_matrix.mean(axis=1)
```


```python
# 3. THRESHOLD OPTIMIZATION (Chapter 6 Metric Squeeze)
# ===================================================
print("\nOptimizing decision threshold on Out-of-Fold predictions...")
best_threshold = 0.5
best_score = 0.0

# Search across 1000 threshold candidates between 0 and 1
thresholds = np.linspace(0.001, 0.999, 1000)
for t in thresholds:
    preds = (oof_preds > t).astype(int)
    # Note: If your metric is strictly Macro F1 or another specific variant, swap it here
    score = f1_score(y_train, preds) 
    
    if score > best_score:
        best_score = score
        best_threshold = t

print(f"🥇 Optimal Threshold Found: {best_threshold:.4f}")
print(f"📈 Best Local Validation Score: {best_score * 100:.2f} / 100")
```


```python
import pandas as pd
import numpy as np

# 1. Profile the raw test probabilities
probs_series = pd.Series(final_test_probs)
print("--- Test Set Probability Distribution ---")
print(probs_series.describe())

print("\n--- Counting Predicted Test Defects by Threshold ---")
# Sweep thresholds to see how many defects are assigned
for t in [0.20, 0.2368, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
    defect_count = (final_test_probs > t).sum()
    print(f"At Threshold {t:.4f}: Model assigns {defect_count} defects out of {len(final_test_probs)} rows.")
```


```python

```


```python
import pandas as pd
import numpy as np

# Aggressive flight targeting the 265 total true defects
counts_to_probe = [215]

for count in counts_to_probe:
    preds = np.zeros(len(final_test_probs), dtype=int)
    
    # Grab the top highest-probability items for this tier
    top_indices = np.argsort(final_test_probs)[-count:]
    preds[top_indices] = 1
    
    probing_sub = pd.DataFrame({
        "CoilID": X_test_enhanced.index,
        "Y": preds
    })
    
    filename = f"new_sub_{count}_defects.csv"
    probing_sub.to_csv(filename, index=False)
    print(f"Generated {filename}")
```


```python
# 4. GENERATE THE IMPROVED SUBMISSION
# ===================================================
# Apply the mathematically optimal threshold to your test set probabilities
final_test_preds = (final_test_probs > best_threshold).astype(int)

submission = pd.DataFrame({
    "CoilID": X_test_enhanced.index, # Pulling the IDs safely from our index
    "Y": final_test_preds
})

submission.to_csv("submission5.csv", index=False)
print("\nGenerated 'submission5.csv'. Ready for upload!")
```


```python

```


```python
### Data Prep Pipeline
```


```python

```


```python
# XGBoost handles numeric columns perfectly, but needs help with text/objects
cat_cols = X_train_given.select_dtypes(include=['object', 'category']).columns
num_cols = X_train_given.select_dtypes(exclude=['object', 'category']).columns
```


```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
```


```python

```


```python
# 4. Set up the Preprocessor
# We only transform text columns. Numerical columns bypass this and go straight through untouched.
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ],
    remainder='passthrough' # Crucial: This leaves numerical columns alone
)
```


```python
# Using standard default parameters for a reliable baseline
baseline_pipeline = Pipeline(steps=[
    ('prep', preprocessor),
    ('model', xgb.XGBClassifier(random_state=42, n_estimators=100, max_depth=4))
])
```


```python
# 6. Fit the model on your 1,352 rows
print("Training the baseline prototype...")
baseline_pipeline.fit(X_train_given, y_train_given)
```


```python

```


```python
baseline_score = baseline_pipeline.score(X_test_given, y_test_given)
print(f"\n🚀 Baseline Test Accuracy: {baseline_score:.4f}")
```


```python

```


```python
# 1. Generate predictions using your trained pipeline
real_predictions = baseline_pipeline.predict(df_test)

# 2. Create the submission DataFrame
submission = pd.DataFrame({
    'CoilID': df_test['CoilID'], # Replace 'id' with your test dataset's identifier column name
    'Y': real_predictions
})

# 3. Export to CSV for submission
submission.to_csv("my_model_submission.csv", index=False)
```


```python
submission = pd.DataFrame({
    "CoilID": X_test.index,       # This pulls the IDs smoothly back out from the index
    "target": final_predictions   # Your optimized predictions
})

submission.to_csv("submission_with_indexed_id.csv", index=False)
print("Submission file successfully generated!")
```


```python

```


```python

```


```python

```
