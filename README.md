# Credit Card Fraud Detection Dashboard

This project is an interactive Streamlit dashboard that demonstrates how machine learning can be used to detect fraudulent credit card transactions. Built as a final project for DS 2010 by Naushin Uddin, the app focuses on both technical accuracy and user-friendly design, including a pastel-inspired visual theme.

## Project Overview

Credit card fraud is a real-world problem with significant financial implications. This dashboard presents a simplified end-to-end solution using a well-known Kaggle dataset. It includes data preprocessing, model training, evaluation, and interpretation—all accessible through a web-based interface.

## What the App Does

- Loads the Kaggle credit card fraud dataset (~285,000 transactions), or allows CSV upload
- Applies data scaling and SMOTE to address class imbalance
- Lets the user choose between Logistic Regression and Random Forest
- Trains and evaluates models on resampled data
- Displays:
  - A classification report
  - Confusion matrix
  - ROC curve
  - Feature importance chart (for Random Forest)

## Features

- Intuitive Streamlit interface
- Custom CSS for a clean, pastel-inspired design
- Fast-mode toggle for reduced training time
- Supports user-provided CSVs with the same structure
- Organized metrics for clear comparison

## Installation

1. Clone the repository or download the files
2. Install required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset from Kaggle:
   https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

4. Place `creditcard.csv` in:

   ```
   ~/.cache/kagglehub/datasets/mlg-ulb/creditcardfraud/versions/3/
   ```

5. Run the app:

   ```bash
   streamlit run credit_card_fraud_app.py
   ```

## Repository Structure

```
credit_card_fraud_app.py   # Main Streamlit app
README.md                  # Project overview
requirements.txt           # List of dependencies
```

## Example Metrics

Example output (Random Forest):

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| 0     | 0.93      | 0.97   | 0.95     |
| 1     | 0.97      | 0.92   | 0.95     |

## Tools Used

- Python 3.x
- Pandas, NumPy, Scikit-learn
- SMOTE from imbalanced-learn
- Matplotlib, Seaborn
- Streamlit

## About

Developed by Naushin Uddin  

This project was created as a way to practice applied machine learning, explainable evaluation, and visual communication of results in a business-relevant context.