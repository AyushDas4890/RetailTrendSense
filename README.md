# DMart Product Classification

## Project Overview
This project aims to classify DMart products into their respective categories based on various features such as product name, brand, price, discounted price, and quantity. It utilizes Machine Learning algorithms to predict the category of a product and provides a user-friendly Streamlit web application for real-time predictions and Exploratory Data Analysis (EDA).

## Dataset
The project uses the `DMart.csv` dataset, which contains information about various products available at DMart.
Key features include:
- **Name**: Name of the product.
- **Brand**: Brand of the product.
- **Price**: Original price of the product.
- **DiscountedPrice**: Discounted price of the product.
- **Quantity**: Quantity of the product (e.g., 500 gm, 1 kg).
- **Category**: The target variable representing the product category.
- **SubCategory**: Sub-category of the product.

## Features
- **Data Preprocessing**: Handling missing values, parsing quantity units (gm, kg, l, ml), and encoding categorical variables.
- **Feature Engineering**: Calculating discount percentage and scaling numerical features.
- **Text Processing**: Using TF-IDF Vectorizer to process product names.
- **Model Training**: Training multiple models to find the best performer:
    - Logistic Regression
    - Random Forest Classifier
    - XGBoost Classifier
- **Web Application**: A Streamlit app with two main sections:
    - **Prediction**: Interactive form to input product details and get category predictions using the trained models.
    - **EDA**: Visualizations of dataset distribution, price analysis, and top brands.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create and activate a virtual environment (optional but recommended):**
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install pandas numpy scikit-learn xgboost matplotlib seaborn streamlit joblib scipy
    ```

## Usage

### Training Models
To retrain the models and save the artifacts:
```bash
python train_and_save_models.py
```
This will generate `dmart_models.joblib` containing the trained models and preprocessors.

### Running the Web App
To start the Streamlit application:
```bash
streamlit run app.py
```
The app will open in your default web browser.

## Project Structure
- `app.py`: Main Streamlit application file.
- `train_and_save_models.py`: Script for training models and saving artifacts.
- `DMart_Classification_Models.ipynb`: Jupyter Notebook for experimentation and analysis.
- `DMart.csv`: Dataset file.
- `dmart_models.joblib`: Serialized model artifacts (generated after training).

## Future Improvements
- Integrate more advanced NLP techniques for product name processing.
- Expand the dataset to include more diverse products.
- Deploy the application to a cloud platform.
