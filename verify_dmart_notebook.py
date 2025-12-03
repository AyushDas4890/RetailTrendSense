# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Data preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

# Machine Learning Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier

# Model evaluation metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, roc_auc_score, auc
)
from sklearn.multiclass import OneVsRestClassifier
from itertools import cycle

# Warnings
import warnings
warnings.filterwarnings('ignore')

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("✅ All libraries imported successfully!")

# Load the dataset
try:
    df = pd.read_csv('DMart.csv')
    print("✅ Dataset loaded successfully!")
    print(f"Dataset Shape: {df.shape}")
except FileNotFoundError:
    print("❌ Error: 'DMart.csv' not found. Please ensure the file is in the same directory.")
    exit()

# Handle Missing Values
df['Brand'].fillna('Unknown', inplace=True)
df.dropna(subset=['Price', 'DiscountedPrice', 'Category', 'Quantity'], inplace=True)

# Feature Engineering: Parse Quantity
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

# Feature Engineering: Discount Percentage
df['DiscountPercentage'] = (df['Price'] - df['DiscountedPrice']) / df['Price']
df['DiscountPercentage'] = df['DiscountPercentage'].fillna(0)

# Filter out categories with less than 2 samples
min_samples = 2
category_counts = df['Category'].value_counts()
valid_categories = category_counts[category_counts >= min_samples].index
df = df[df['Category'].isin(valid_categories)]

print(f"Filtered Dataset Shape: {df.shape}")
print(f"Number of Categories: {df['Category'].nunique()}")

# Label Encoding
le_brand = LabelEncoder()
df['Brand_Encoded'] = le_brand.fit_transform(df['Brand'].astype(str))

le_category = LabelEncoder()
df['Category_Encoded'] = le_category.fit_transform(df['Category'])
class_names = le_category.classes_

# TF-IDF on Name (Text Feature)
print("Vectorizing Product Names...")
tfidf = TfidfVectorizer(max_features=2000, stop_words='english')
name_tfidf = tfidf.fit_transform(df['Name'].fillna(''))

# Numerical Features
X_numerical = df[['Brand_Encoded', 'Price', 'DiscountedPrice', 'Quantity_Value', 'DiscountPercentage']]

# Scaling Numerical Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numerical)

# Combine Numerical and Text Features
print("Combining Features...")
X = hstack([X_scaled, name_tfidf])
y = df['Category_Encoded']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"✅ Train-Test Split completed!")
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# Define Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=30, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_estimators=300, learning_rate=0.1, random_state=42)
}

# Add Voting Classifier
voting_clf = VotingClassifier(
    estimators=[
        ('lr', models["Logistic Regression"]),
        ('rf', models["Random Forest"]),
        ('xgb', models["XGBoost"])
    ],
    voting='soft'  # Soft voting for probability-based ROC/AUC
)
models["Ensemble Voting Classifier"] = voting_clf

# Function to plot Confusion Matrix (Modified to not show plot during script run, just print)
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    print(f"Confusion Matrix for {title} generated.")
    # In script, we skip plt.show() to avoid blocking

# Function to plot ROC Curve (Modified to not show plot during script run)
def plot_roc_curve(y_test, y_prob, title, n_classes):
    # Binarize the output
    y_test_bin = label_binarize(y_test, classes=range(n_classes))
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    print(f"ROC Curve for {title} calculated. Micro-average AUC: {roc_auc['micro']:.4f}")

# Training and Evaluation Loop
results = {}

for name, model in models.items():
    print(f"\\n{'='*60}")
    print(f"Training {name}...")
    print(f"{'='*60}")
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"Accuracy: {acc:.4f}")
    # print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0)) # Suppress long output
    
    # Visualizations
    plot_confusion_matrix(y_test, y_pred, name)
    plot_roc_curve(y_test, y_prob, name, len(class_names))

print("\\n✅ All models trained and evaluated!")

results_df = pd.DataFrame(list(results.items()), columns=['Model', 'Accuracy'])
results_df = results_df.sort_values(by='Accuracy', ascending=False)

print("\\nModel Comparison:")
print(results_df)
