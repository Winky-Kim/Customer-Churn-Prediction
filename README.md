
# Customer Churn Prediction

This project focuses on predicting customer churn using supervised and unsupervised learning techniques. The workflow includes data preprocessing, model building, evaluation, and performance comparison across different algorithms.

## Project Structure

```
.
├── data/
│   ├── customer_churn_dataset-training-master.xls
│   └── customer_churn_dataset-testing-master.xls
├── 1.data_preprocessing.ipynb
├── 2.model.ipynb
├── X_scaled_df.csv
├── y.csv
├── scaler.pkl
└── label_encoders.pkl
```

## Workflow Overview

### 1. Data Preprocessing
- Standardization using `StandardScaler`
- Label encoding of categorical features
- Output: `X_scaled_df.csv`, `y.csv`, and saved encoders

### 2. Modeling Techniques
- **Unsupervised**: K-Means Clustering with silhouette score analysis and Sankey Diagram visualization
- **Supervised**: KNN, Logistic Regression, XGBoost, StackingClassifier (LogReg + XGBoost)
- Hyperparameter tuning using `RandomizedSearchCV`
- Addressed class imbalance with `class_weight='balanced'` and `scale_pos_weight`

### 3. Evaluation Metrics
- Accuracy
- Precision, Recall, F1-Score
- ROC AUC
- PR AUC
- Confusion Matrix

## Results Summary

| Model               | Accuracy | AUC    | Precision (Churn) | Recall (Churn) | F1 (Churn) |
|--------------------|----------|--------|--------------------|----------------|------------|
| KNN                | 0.64     | 0.5785 | 0.66               | 0.88           | 0.75       |
| Logistic Regression| 0.66     | 0.7748 | 0.68               | 0.86           | 0.76       |
| XGBoost            | 0.63     | 0.6069 | 0.65               | 0.91           | 0.75       |
| StackingClassifier | 0.63     | 0.6089 | 0.65               | 0.91           | 0.75       |

## Observations

- **Logistic Regression** performed best in terms of AUC and overall balance.
- **XGBoost** and **Stacking** had strong recall but lower precision.
- **Class imbalance** was a major challenge, handled with weighting and SMOTE.
- **K-Means Clustering** was not effective for meaningful churn separation.

## Future Work

- Incorporate advanced feature engineering and automated interaction terms
- Use GPU-accelerated training for XGBoost and deep learning models
- Implement cross-validation for all models

