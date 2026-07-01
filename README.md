# Insurance Claim Prediction

An end-to-end machine learning project for predicting insurance claim outcomes. Built as an actuarial data science portfolio project.

## Project Overview

**Goal:** Predict whether a policyholder will file an insurance claim (`Claim = 0 or 1`) using machine learning.

**Dataset:** 50,000 synthetic policyholders with 30 features covering customer demographics, vehicle information, driving history, policy details, and environmental factors.

**Claim Rate:** ~17.6% (realistic for auto insurance portfolios)

## Dataset

The dataset (`data/insurance_claim_dataset.csv`) contains 50,000 rows and 30 columns:

| Category                  | Columns                                                                                     |
| ------------------------- | ------------------------------------------------------------------------------------------- |
| **Customer Demographics** | Customer_ID, Age, Gender, Marital_Status, Occupation, Annual_Income, Credit_Score, Region   |
| **Vehicle Information**   | Vehicle_Type, Vehicle_Age, Vehicle_Value, Annual_Mileage, Engine_Size, Safety_Rating        |
| **Driving History**       | Driving_Experience, Years_Licensed, Previous_Claims, Previous_Accidents, Traffic_Violations |
| **Policy Information**    | Policy_Type, Coverage_Amount, Deductible, Annual_Premium, Policy_Duration                   |
| **Environmental Factors** | Urban_Rural, Parking_Type, Flood_Risk, Crime_Index                                          |
| **Target**                | Claim, Claim_Amount                                                                         |

## Project Structure

```
insurance_claim_project/
├── data/
│   └── insurance_claim_dataset.csv    # Dataset (50,000 rows)
├── notebooks/
│   ├── 01_EDA.ipynb                   # Exploratory Data Analysis
│   ├── 02_Machine_Learning.ipynb      # Model Training & Tuning
│   └── 03_Model_Evaluation.ipynb      # Model Evaluation & Comparison
├── models/
│   ├── powerbi_dashboard.xlsx         # Power BI data (all sheets)
│   ├── model_comparison.csv           # Metrics summary
│   ├── test_predictions.csv           # Predictions + probabilities
│   ├── feature_importance.csv         # Feature importance by model
│   ├── roc_curve_data.csv             # ROC curve coordinates
│   ├── pr_curve_data.csv              # PR curve coordinates
│   ├── threshold_analysis.csv         # Threshold optimization
│   ├── business_impact.csv            # Cost savings analysis
│   ├── confusion_matrices.csv         # Confusion matrix data
│   ├── *.pkl                          # Trained model files
│   └── *.csv                          # Additional exports
├── reports/figures/                   # Generated visualizations
└── README.md                          # This file
```

## Notebooks

### 1. Exploratory Data Analysis (`01_EDA.ipynb`)

- Data overview, missing values, summary statistics
- Univariate & bivariate analysis of all features vs claim rate
- Correlation heatmaps, violin plots, pairplots
- Business insights after every visualization

### 2. Machine Learning (`02_Machine_Learning.ipynb`)

- Data preprocessing (encoding, scaling, train/test split)
- Three models with hyperparameter tuning:
  - **Logistic Regression** (GridSearchCV)
  - **Random Forest** (RandomizedSearchCV)
  - **XGBoost** (RandomizedSearchCV)
- Feature importance analysis
- Step-by-step explanations

### 3. Model Evaluation (`03_Model_Evaluation.ipynb`)

- Comprehensive evaluation metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- ROC curves, PR curves, confusion matrices
- Threshold optimization analysis
- Business impact analysis (cost savings estimation)
- Final model recommendation

## Models Used

| Model                   | Why?                                                                              |
| ----------------------- | --------------------------------------------------------------------------------- |
| **Logistic Regression** | Interpretable, industry standard in insurance, well-calibrated probabilities      |
| **Random Forest**       | Captures non-linear relationships, robust to outliers, feature importance         |
| **XGBoost**             | State-of-the-art gradient boosting, built-in regularization, handles missing data |

## Key Findings

From the EDA:

- Young drivers (18-25) have the highest claim rate (~38%)
- Poor credit scores correlate with significantly higher claims
- Sports cars and high-mileage drivers show elevated risk
- Previous claims history is a strong predictor of future claims

## Requirements

- Python 3.8+
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- xgboost

## Usage

```bash
# Run notebooks in order
jupyter notebook notebooks/01_EDA.ipynb
jupyter notebook notebooks/02_Machine_Learning.ipynb
jupyter notebook notebooks/03_Model_Evaluation.ipynb
```

## Author

Actuarial Data Science Portfolio Project
