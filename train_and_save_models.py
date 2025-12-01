import pandas as pd
import numpy as np
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# 1. Load Data
print("Loading dataset...")
df = pd.read_csv('DMart.csv')

# 2. Preprocessing
print("Preprocessing data...")
# Fill missing values
df['Brand'].fillna('Unknown', inplace=True)
df.dropna(subset=['Price', 'DiscountedPrice', 'Category', 'Quantity'], inplace=True)

# Parse Quantity
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

# Filter categories
min_samples = 2
category_counts = df['Category'].value_counts()
valid_categories = category_counts[category_counts >= min_samples].index
df = df[df['Category'].isin(valid_categories)]

# 3. Feature Selection & Encoding
print("Encoding features...")
# Text Vectorization
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
name_tfidf = tfidf.fit_transform(df['Name'].fillna(''))

# Label Encoding for Categorical Columns
le_brand = LabelEncoder()
df['Brand_Encoded'] = le_brand.fit_transform(df['Brand'].astype(str))

le_subcategory = LabelEncoder()
df['SubCategory_Encoded'] = le_subcategory.fit_transform(df['SubCategory'].astype(str))

# Target Encoding
le_category = LabelEncoder()
y = le_category.fit_transform(df['Category'])

# Numerical Features
scaler = StandardScaler()
numerical_features = scaler.fit_transform(df[['Price', 'DiscountedPrice', 'Quantity_Value', 'DiscountPercentage', 'Brand_Encoded', 'SubCategory_Encoded']])

# Combine Features
X = hstack([name_tfidf, numerical_features])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Model Training
print("Training models...")

# Logistic Regression
print("Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, lr_model.predict(X_test)):.4f}")

# Random Forest
print("Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=300, max_depth=30, random_state=42)
rf_model.fit(X_train, y_train)
print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_model.predict(X_test)):.4f}")

# XGBoost
print("Training XGBoost...")
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_estimators=300, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)
print(f"XGBoost Accuracy: {accuracy_score(y_test, xgb_model.predict(X_test)):.4f}")

# 5. Save Artifacts
print("Saving models and encoders...")
artifacts = {
    'tfidf': tfidf,
    'le_brand': le_brand,
    'le_subcategory': le_subcategory,
    'le_category': le_category,
    'scaler': scaler,
    'lr_model': lr_model,
    'rf_model': rf_model,
    'xgb_model': xgb_model
}

joblib.dump(artifacts, 'dmart_models.joblib')
print("All models and encoders saved to 'dmart_models.joblib'")
