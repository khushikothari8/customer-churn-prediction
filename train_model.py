# ============================================================
# CUSTOMER CHURN PREDICTION — COMPLETE TRAINING SCRIPT
# ============================================================

# --- Import Libraries ---
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 50)
print("STEP 1 — Loading and Cleaning Data")
print("=" * 50)

# Load dataset
df = pd.read_csv('data/telecom_churn.csv')

# Drop customerID — just a unique ID, no pattern to learn
df = df.drop('customerID', axis=1)

# Fix TotalCharges — it's stored as text, convert to number
# errors='coerce' turns invalid values into NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Drop the ~11 rows where TotalCharges became NaN
df = df.dropna()

# Convert target column: Yes → 1, No → 0
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# One-hot encode all text columns into numbers
# drop_first=True avoids duplicate info columns
df = pd.get_dummies(df, drop_first=True)

print("Data cleaned. Shape:", df.shape)

# ============================================================
print("\n" + "=" * 50)
print("STEP 2 — Splitting Features and Target")
print("=" * 50)
# ============================================================

# X = all columns EXCEPT Churn (these are inputs/features)
# y = only the Churn column (this is what we want to predict)
X = df.drop('Churn', axis=1)
y = df['Churn']

print("Features shape:", X.shape)
print("Target shape:", y.shape)
print("Churn distribution BEFORE SMOTE:")
print(y.value_counts())

# ============================================================
print("\n" + "=" * 50)
print("STEP 3 — Train/Test Split")
print("=" * 50)
# ============================================================

# Split data into training set (80%) and testing set (20%)
# Training set — model learns from this
# Testing set  — we test how well it learned on UNSEEN data
# random_state=42 means the split is always the same (reproducible)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,        # 20% goes to testing
    random_state=42,      # fixes the randomness so results are same every run
    stratify=y            # keeps same churn ratio in both train and test
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples:  {X_test.shape[0]}")

# ============================================================
print("\n" + "=" * 50)
print("STEP 4 — Feature Scaling")
print("=" * 50)
# ============================================================

# Some columns have very different ranges:
# tenure: 0-72, MonthlyCharges: 18-118, SeniorCitizen: 0-1
# Logistic Regression gets confused by this difference in scale
# StandardScaler brings everything to same scale (mean=0, std=1)

scaler = StandardScaler()

# IMPORTANT RULE:
# fit_transform on TRAIN — scaler learns the mean/std from training data
# transform only on TEST — apply same scale, don't re-learn
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Scaling done!")

# ============================================================
print("\n" + "=" * 50)
print("STEP 5 — Applying SMOTE on Training Data")
print("=" * 50)
# ============================================================

# SMOTE creates synthetic examples of minority class (churners)
# IMPORTANT: Apply SMOTE ONLY on training data, NEVER on test data
# (test data must stay real to give honest evaluation)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

print("Churn distribution AFTER SMOTE:")
unique, counts = np.unique(y_train_resampled, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  Class {u}: {c} samples")

# ============================================================
print("\n" + "=" * 50)
print("STEP 6 — Training Models")
print("=" * 50)
# ============================================================

# --- MODEL 1: Logistic Regression ---
# Simple, fast, draws a straight line to separate classes
# max_iter=1000 gives it enough attempts to find the best line
print("\nTraining Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_resampled, y_train_resampled)
# .fit() = "train the model on this data"

# --- MODEL 2: Random Forest ---
# Builds 100 decision trees and takes majority vote
# n_estimators=100 means 100 trees
print("Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_resampled, y_train_resampled)

print("Both models trained!")

# ============================================================
print("\n" + "=" * 50)
print("STEP 7 — Evaluating and Comparing Models")
print("=" * 50)
# ============================================================

# Make predictions on TEST data (data model has never seen)
lr_predictions = lr_model.predict(X_test_scaled)
rf_predictions = rf_model.predict(X_test_scaled)

# --- F1 Scores ---
# average='weighted' accounts for class imbalance in scoring
lr_f1 = f1_score(y_test, lr_predictions, average='weighted')
rf_f1 = f1_score(y_test, rf_predictions, average='weighted')

print(f"\nLogistic Regression F1 Score: {lr_f1:.4f}")
print(f"Random Forest F1 Score:       {rf_f1:.4f}")

# --- Detailed Report ---
print("\n--- Logistic Regression Report ---")
print(classification_report(y_test, lr_predictions, target_names=['No Churn', 'Churn']))

print("\n--- Random Forest Report ---")
print(classification_report(y_test, rf_predictions, target_names=['No Churn', 'Churn']))

# ============================================================
print("\n" + "=" * 50)
print("STEP 8 — Picking Best Model and Saving")
print("=" * 50)
# ============================================================

# Pick whichever model has higher F1 score
if rf_f1 >= lr_f1:
    best_model = rf_model
    best_name = "Random Forest"
    best_f1 = rf_f1
else:
    best_model = lr_model
    best_name = "Logistic Regression"
    best_f1 = lr_f1

print(f"\n✅ Best Model: {best_name} (F1: {best_f1:.4f})")

# Save the best model to a file so Streamlit app can load it
# joblib.dump = saves Python object to disk
joblib.dump(best_model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(list(X.columns), 'columns.pkl')  # save column names too

print("✅ model.pkl saved!")
print("✅ scaler.pkl saved!")
print("✅ columns.pkl saved!")

# ============================================================
print("\n" + "=" * 50)
print("STEP 9 — Confusion Matrix Chart")
print("=" * 50)
# ============================================================

# Confusion Matrix shows:
# True Negatives  | False Positives
# False Negatives | True Positives
cm = confusion_matrix(y_test, rf_predictions if best_name == "Random Forest" else lr_predictions)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted No Churn', 'Predicted Churn'],
            yticklabels=['Actual No Churn', 'Actual Churn'])
plt.title(f'Confusion Matrix — {best_name}')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("✅ confusion_matrix.png saved!")

print("\n" + "=" * 50)
print("🎉 TRAINING COMPLETE!")
print("=" * 50)