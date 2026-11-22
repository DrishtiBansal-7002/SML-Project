# Shopper Intelligence – Customer Segmentation & Purchase Prediction

## Project Overview

Shopper Intelligence is a Machine Learning-based project designed to analyze online shopping behavior, segment shopping sessions, and predict whether a visitor is likely to make a purchase.

The project uses the Online Shoppers Purchasing Intention dataset and combines classification and clustering techniques to understand customer behavior and support data-driven decisions.

A Streamlit web application is developed to present the analysis, predictions, customer segments, and model performance through an interactive dashboard.

## Objectives

- Analyze online shopping behavior using Exploratory Data Analysis (EDA).
- Predict whether a shopping session will result in a purchase.
- Group shopping sessions into different segments using clustering.
- Visualize important patterns and model performance.
- Present the results through an interactive Streamlit application.

## Dataset

**Dataset:** Online Shoppers Purchasing Intention Dataset

- 12,330 sessions
- 18 features
- Contains numerical and categorical attributes related to online shopping behavior.
- Target variable: `Revenue`

The `Revenue` column indicates whether a shopping session resulted in a purchase.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Plotly
- Streamlit
- Joblib
- Jupyter Notebook

## Machine Learning Models

### 1. Logistic Regression – Purchase Prediction

Logistic Regression is used as a classification algorithm to predict whether a visitor will make a purchase.

The model uses preprocessing techniques such as:
- Missing value imputation
- Feature scaling
- One-hot encoding
- Train-test split

The model is evaluated using Accuracy, Precision, Recall, F1 Score, and ROC-AUC.

### 2. K-Means Clustering – Customer Segmentation

K-Means is used to group shopping sessions based on their behavioral patterns.

- The `Revenue` column is excluded during clustering.
- The Elbow Method and Silhouette Score are used to help determine the number of clusters.
- The final model uses 2 clusters.

**Note:** These clusters represent groups of shopping sessions based on behavior, not individually identified customers.

## Model Performance

The Logistic Regression model achieved the following results:

| Evaluation Metric | Score |
|---|---:|
| Accuracy | 85.13% |
| Precision | 51.64% |
| Recall | 78.53% |
| F1 Score | 62.31% |
| ROC-AUC | 91.09% |

## Project Structure

```text
Shopper-Intelligence/
│
├── app.py
├── Shopper Intelligence.ipynb
├── Online Shopper Dataset.csv
│
├── purchase_model.pkl
├── clustering_model.pkl
├── cluster_preprocessor.pkl
│
├── customer_data.csv
├── test_data.csv
├── model_metrics.json
│
└── .streamlit/
    └── config.toml
```
## Application Features

The Streamlit application includes the following sections:

- **Dashboard:** Overview of shopping behavior and key statistics.
- **Customer Segmentation:** Visualization and analysis of shopping session clusters.
- **Purchase Prediction:** Predicts the likelihood of a shopping session resulting in a purchase.
- **Model Performance:** Displays evaluation metrics of the trained model.
- **About Project:** Provides information about the project and its objectives.

## Future Scope

- Experiment with additional Machine Learning algorithms to improve prediction performance.
- Explore more advanced customer segmentation techniques.
- Incorporate additional shopping behavior features.
- Improve the user interface and dashboard experience.
- Deploy the application online for wider accessibility.

## Team Members

Drishti Bansal – S25CSEU1174
Yashvi Parashar - S25CSEU1168

## Conclusion

Shopper Intelligence demonstrates how Machine Learning techniques can be used to understand online shopping behavior, identify behavioral patterns, and predict purchase intentions.

By combining Logistic Regression, K-Means Clustering, and an interactive Streamlit application, the project provides an accessible way to explore and interpret online shopping data.




