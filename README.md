# 🛍️ Shopper Intelligence
### Customer Segmentation & Purchase Prediction Using Machine Learning

Shopper Intelligence is a Machine Learning-based project that analyzes online shopping behaviour, identifies patterns in browsing sessions, and predicts purchase intention using classification and clustering techniques.The project combines Logistic Regression and K-Means Clustering with an interactive Streamlit web application to explore shopping behaviour and generate meaningful insights.

## 📌 Project Overview

Online shopping platforms generate large amounts of browsing data. Understanding this data can help identify purchasing patterns and distinguish between different types of shopping sessions.

Shopper Intelligence focuses on two major Machine Learning tasks:

- **Purchase Prediction:** Predict whether an online shopping session will result in a purchase.
- **Customer Segmentation:** Group similar shopping sessions based on browsing behaviour and session characteristics.

The project also provides an interactive dashboard to explore shopping statistics, customer segments, purchase predictions, and model performance.

## 🎯 Problem Statement

E-commerce platforms need to understand visitor behaviour to improve customer engagement and purchasing outcomes.

This project addresses the following questions:

1. Can we predict whether a visitor will make a purchase based on their browsing session?
2. Can we identify groups of similar shopping sessions using unsupervised learning?
3. What insights can be extracted from online shopping behaviour?

## 📊 Dataset

**Dataset:** Online Shoppers Intention Dataset

The dataset contains 12,330 browsing sessions and 18 features describing visitor activities and session characteristics.

### Important Features

| Feature | Description |
|---|---|
| Administrative | Administrative pages visited |
| Informational | Informational pages visited |
| ProductRelated | Product-related pages visited |
| BounceRates | Bounce rate of the session |
| ExitRates | Exit rate of the session |
| PageValues | Value of pages visited |
| SpecialDay | Closeness to a special shopping day |
| Month | Month of the session |
| VisitorType | Type of website visitor |
| Weekend | Whether the session occurred on a weekend |
| Revenue | Purchase outcome |

**Target Variable:** `Revenue`

- `1` – Purchase
- `0` – No Purchase

The dataset is cleaned by removing duplicate rows, and exploratory data analysis is performed to understand the underlying patterns.

## 🧠 Machine Learning Algorithms

### 1. Logistic Regression – Purchase Prediction

Logistic Regression is used as a supervised classification algorithm to predict whether an online shopping session will result in a purchase.

**Implementation:**
- Numerical feature preprocessing using median imputation and StandardScaler.
- Categorical feature preprocessing using most-frequent imputation and OneHotEncoder.
- Train-test split with 80% training data and 20% testing data.
- Class balancing using `class_weight='balanced'`.

### 2. K-Means Clustering – Customer Segmentation

K-Means is used as an unsupervised learning algorithm to group similar online shopping sessions.

**Implementation:**
- Revenue is excluded from clustering.
- Numerical and categorical features are preprocessed separately.
- The Elbow Method and Silhouette Score are used to examine cluster configurations.
- The final model uses 2 clusters.

The resulting clusters represent groups of similar shopping sessions rather than individually identified customers.

## ✨ Key Features

- **Interactive Dashboard:** Displays total sessions, purchases, non-purchases, and conversion rate.
- **Customer Segmentation:** Visualizes shopping-session clusters and their behavioural profiles.
- **Purchase Prediction:** Estimates purchase intention using visitor session details.
- **Model Performance:** Presents classification evaluation results.
- **Data Visualization:** Displays purchase distribution, session statistics, and cluster comparisons.
- **Streamlit Interface:** Provides an interactive interface with multiple navigation sections.

## 📸 Application Screenshots

The following screenshots showcase the different sections of the Shopper Intelligence application.

### 1. Dashboard

### 2. Customer Segmentation

### 3. Purchase Prediction

### 4. Model Performance

## 📈 Model Evaluation

The Logistic Regression model is evaluated using the following metrics on the test dataset:

| Evaluation Metric | Score |
|---|---:|
| Accuracy | 85.13% |
| Precision | 51.64% |
| Recall | 78.53% |
| F1 Score | 62.31% |
| ROC-AUC | 91.09% |

These results indicate the model's classification performance on the test dataset.

## 🔄 Project Workflow

1. Dataset Collection
2. Data Understanding and Exploration
3. Data Cleaning and Preprocessing
4. Exploratory Data Analysis
5. Feature Engineering and Preprocessing
6. Train-Test Split
7. Logistic Regression Model Training
8. K-Means Clustering
9. Model Evaluation
10. Model Saving and Integration
11. Streamlit Web Application Development

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Analysis | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-learn |
| Model Serialization | Joblib |
| Web Application | Streamlit |
| Development Environment | Jupyter Notebook / Google Colab |

## 📁 Project Structure

```text
SML-Project/
│
├── .streamlit/
│   └── config.toml
│
├── screenshots/
│   ├── dashboard.png
│   ├── customer-segmentation.png
│   ├── purchase-prediction.png
│   └── model-performance.png
│
├── Shopper Intelligence.ipynb
├── Online Shopper Dataset.csv
├── app.py
│
├── purchase_model.pkl
├── clustering_model.pkl
├── cluster_preprocessor.pkl
│
├── customer_data.csv
├── test_data.csv
├── model_metrics.json
│
├── requirements.txt
└── README.md

## 🚀 Future Scope

- Experiment with additional classification algorithms.
- Improve predictive performance through hyperparameter tuning.
- Explore alternative clustering techniques.
- Add more detailed behavioural analytics.
- Enhance the dashboard with additional interactive visualizations.
- Deploy the application for online accessibility.

## 👥 Team Members

- Drishti Bansal [S25CSEU1174]
- Yashvi Parashar [S25CSEU1168]

## 🎓 Academic Context

Developed as part of the Statistical Machine Learning (SML) coursework.

## Acknowledgements

- Online Shoppers Intention Dataset.
- Scikit-learn documentation.
- Streamlit documentation.
- Python data science community.


