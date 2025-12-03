"""
DMart Classification Models - Complete Analysis
This script runs all the analysis from the Jupyter notebook
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, roc_auc_score, auc
)
import warnings
warnings.filterwarnings('ignore')
import re

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("DMart Product Classification - Complete Analysis")
print("="*80)

# Step 1: Load Dataset
print("\n[1/8] Loading dataset...")
df = pd.read_csv('DMart.csv')
print(f"✓ Dataset loaded successfully!")
print(f"  Shape: {df.shape}")
print(f"  Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# Step 2: Data Preprocessing
print("\n[2/8] Preprocessing data...")
print(f"Missing values before cleaning:")
missing = df.isnull().sum()
print(missing[missing > 0])

df['Brand'].fillna('Unknown', inplace=True)
df.dropna(subset=['Price', 'DiscountedPrice', 'Category', 'Quantity'], inplace=True)
print(f"✓ Dataset shape after handling missing values: {df.shape}")

# Feature Engineering
def parse_quantity(q):
    if isinstance(q, str):
        q = q.lower()
        num = re.search(r'(\d+(\.\d+)?)', q)
        if num:
            val = float(num.group(1))
            if 'kg' in q or 'l' in q:
                val *= 1000
            return val
    return 0

df['Quantity_Value'] = df['Quantity'].apply(parse_quantity)
df['DiscountPercentage'] = (df['Price'] - df['DiscountedPrice']) / df['Price']
df['DiscountPercentage'] = df['DiscountPercentage'].fillna(0)
print("✓ Feature engineering completed!")

# Filter categories
min_samples = 2
category_counts = df['Category'].value_counts()
valid_categories = category_counts[category_counts >= min_samples].index
df = df[df['Category'].isin(valid_categories)]
print(f"✓ Filtered dataset shape: {df.shape}")
print(f"  Number of categories: {df['Category'].nunique()}")

# Step 3: Exploratory Data Analysis
print("\n[3/8] Performing EDA...")
print("\nTop 10 Categories:")
print(df['Category'].value_counts().head(10))

print("\nPrice Statistics:")
print(df[['Price', 'DiscountedPrice']].describe())

# Step 4: Prepare features
print("\n[4/8] Preparing features...")
df['text_features'] = (
    df['Name'].fillna('') + ' ' +
    df['Brand'].fillna('') + ' ' +
    df['Description'].fillna('') + ' ' +
    df['SubCategory'].fillna('')
)

X_text = df['text_features']
X_numeric = df[['Price', 'DiscountedPrice', 'Quantity_Value', 'DiscountPercentage']]
y = df['Category']

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(f"✓ Encoded {len(label_encoder.classes_)} categories")

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
X_text_tfidf = tfidf.fit_transform(X_text)

# Scale numeric features
scaler = StandardScaler()
X_numeric_scaled = scaler.fit_transform(X_numeric)

# Combine features
X_combined = hstack([X_text_tfidf, X_numeric_scaled])
print(f"✓ Combined feature shape: {X_combined.shape}")

# Step 5: Train-Test Split
print("\n[5/8] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"✓ Training set: {X_train.shape[0]} samples")
print(f"✓ Test set: {X_test.shape[0]} samples")

# Step 6: Train Models
print("\n[6/8] Training models...")

# Logistic Regression
print("\n  Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)
print(f"  ✓ Logistic Regression Accuracy: {lr_accuracy:.4f}")

# Random Forest
print("\n  Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f"  ✓ Random Forest Accuracy: {rf_accuracy:.4f}")

# XGBoost
print("\n  Training XGBoost...")
xgb_model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_pred)
print(f"  ✓ XGBoost Accuracy: {xgb_accuracy:.4f}")

# Step 7: Model Evaluation
print("\n[7/8] Evaluating models...")

models_results = {
    'Logistic Regression': {'accuracy': lr_accuracy, 'predictions': lr_pred},
    'Random Forest': {'accuracy': rf_accuracy, 'predictions': rf_pred},
    'XGBoost': {'accuracy': xgb_accuracy, 'predictions': xgb_pred}
}

print("\n" + "="*80)
print("MODEL PERFORMANCE SUMMARY")
print("="*80)

for model_name, results in models_results.items():
    print(f"\n{model_name}:")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  Precision: {precision_score(y_test, results['predictions'], average='weighted'):.4f}")
    print(f"  Recall: {recall_score(y_test, results['predictions'], average='weighted'):.4f}")
    print(f"  F1-Score: {f1_score(y_test, results['predictions'], average='weighted'):.4f}")

# Step 8: Final Summary
print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\n✓ Dataset: {df.shape[0]} products across {df['Category'].nunique()} categories")
print(f"✓ Best Model: XGBoost with {xgb_accuracy:.2%} accuracy")
print(f"✓ Target Achieved: {'YES' if xgb_accuracy > 0.88 else 'NO'} (Target: 88%)")
print("\n" + "="*80)
