import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from scipy.sparse import hstack
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(page_title="DMart Product Classifier", layout="wide")

# Load models and encoders
@st.cache_resource
def load_artifacts():
    return joblib.load('dmart_models.joblib')

artifacts = load_artifacts()
tfidf = artifacts['tfidf']
le_brand = artifacts['le_brand']
le_subcategory = artifacts['le_subcategory']
le_category = artifacts['le_category']
scaler = artifacts['scaler']
lr_model = artifacts['lr_model']
rf_model = artifacts['rf_model']
xgb_model = artifacts['xgb_model']

# Helper function to parse quantity
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

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Prediction", "EDA"])

if page == "Prediction":
    st.title("DMart Product Classification")
    st.markdown("Enter product details to predict its category.")

    # Input Form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Product Name", "Maggi 2-Minute Masala Noodles")
            brand = st.selectbox("Brand", le_brand.classes_)
            price = st.number_input("Price", min_value=0.0, value=10.0)
        
        with col2:
            discounted_price = st.number_input("Discounted Price", min_value=0.0, value=9.0)
            quantity = st.text_input("Quantity (e.g., 500 gm, 1 kg)", "70 gm")
            subcategory = st.selectbox("SubCategory", le_subcategory.classes_)
            
        model_choice = st.selectbox("Choose Model", ["XGBoost", "Random Forest", "Logistic Regression"])
        
        submit = st.form_submit_button("Predict")

    if submit:
        # Preprocess Input
        # 1. Quantity Value
        qty_val = parse_quantity(quantity)
        
        # 2. Discount Percentage
        if price > 0:
            discount_pct = (price - discounted_price) / price
        else:
            discount_pct = 0.0
            
        # 3. Encode Categorical
        try:
            brand_enc = le_brand.transform([brand])[0]
        except:
            brand_enc = le_brand.transform(['Unknown'])[0] # Handle unseen labels if 'Unknown' exists, else might error. Ideally handle better.
            
        try:
            subcat_enc = le_subcategory.transform([subcategory])[0]
        except:
             # Fallback or error handling
             st.error("Unknown Subcategory selected")
             st.stop()

        # 4. Scale Numerical
        # Order: 'Price', 'DiscountedPrice', 'Quantity_Value', 'DiscountPercentage', 'Brand_Encoded', 'SubCategory_Encoded'
        num_features = np.array([[price, discounted_price, qty_val, discount_pct, brand_enc, subcat_enc]])
        num_scaled = scaler.transform(num_features)
        
        # 5. TF-IDF
        name_vec = tfidf.transform([name])
        
        # 6. Combine
        input_data = hstack([name_vec, num_scaled])
        
        # Predict
        if model_choice == "Logistic Regression":
            model = lr_model
        elif model_choice == "Random Forest":
            model = rf_model
        else:
            model = xgb_model
            
        prediction = model.predict(input_data)
        predicted_category = le_category.inverse_transform(prediction)[0]
        
        st.success(f"Predicted Category: **{predicted_category}**")
        
elif page == "EDA":
    st.title("Exploratory Data Analysis")
    
    # Load data for EDA
    @st.cache_data
    def load_data():
        df = pd.read_csv('DMart.csv')
        # Basic cleaning for visualization
        df['Brand'].fillna('Unknown', inplace=True)
        df.dropna(subset=['Price', 'DiscountedPrice', 'Category', 'Quantity'], inplace=True)
        return df

    df = load_data()
    
    st.subheader("Dataset Overview")
    st.write(df.head())
    st.write(f"Total Records: {df.shape[0]}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Category Distribution")
        fig, ax = plt.subplots()
        df['Category'].value_counts().plot(kind='bar', ax=ax)
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)
        
    with col2:
        st.subheader("Price Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df['Price'], bins=30, kde=True, ax=ax)
        st.pyplot(fig)

    st.subheader("Top 10 Brands")
    fig, ax = plt.subplots()
    df['Brand'].value_counts().head(10).plot(kind='bar', ax=ax)
    st.pyplot(fig)
