# 🏦 Bank Customer Churn Prediction using Machine Learning

This project focuses on predicting whether a customer will leave the bank (churn) based on various features using different machine learning models and data balancing techniques. Multiple versions of the notebook were used to explore and compare the impact of different preprocessing, sampling, and validation strategies. This project was implemented across separate notebooks, each with slight variations in preprocessing, sampling techniques, or model tuning to compare results. 

[Click here to view the dataset: `Churn_Modelling.csv`](https://www.kaggle.com/datasets/adammaus/predicting-churn-for-bank-customers)

---

## Common Workflow in All Notebooks

- **Data Cleaning**
- **Dropped Unwanted Features**: `'RowNumber'`, `'CustomerId'`, `'Surname'` (in most files)
- **Train-Test Split** (mostly 70:30 or 80:20)
- **Class Imbalance Handling**: either using `resample()` or `SMOTE()`
- **Label Encoding & Feature Scaling**
- **Trained Models**:
  - Logistic Regression
  - KNN
  - Naive Bayes
  - SVM
  - Decision Tree
  - Random Forest
  - XGBoost
  - AdaBoost
- **Applied Cross Validation (mostly 5-fold)**
- **Hyperparameter Tuning** using `GridSearchCV` (for all models except Naive Bayes)

---

## Notebook-wise Differences

### `churn-1.ipynb`
- Dropped: `'RowNumber'`, `'CustomerId'`, `'Surname'`, `'Geography'`
- Data split: 70:30
- Class imbalance handled using `resample()`
- Cross-validation: **5-fold**
- **Highest Accuracy**: 
  - Before Cross Validation: `0.9416` (Random Forest)
  - After Cross Validation: `0.9208` (Random Forest)
  - After Hyperparameter Tuning: `0.94` (Random Forest)

---

### `churn-2.ipynb`
- Same as `churn-1.ipynb`, but:
  - Cross-validation: **10-fold**
- **Highest Accuracy**: 
  - Before Cross Validation: `0.9450` (Random Forest)
  - After Cross Validation: `0.9302` (Random Forest)
  - After Hyperparameter Tuning: `0.94` (Random Forest)
    
---

### `churn-3.ipynb`
- Dropped: `'RowNumber'`, `'CustomerId'`, `'Surname'`, `'Geography'`
- Class imbalance handled using **SMOTE**
- Used `StratifiedKFold` for cross-validation
- Slight drop in accuracy observed after applying SMOTE
- **Highest Accuracy**: 
  - Before Cross Validation: `0.8517` (XGBoost)
  - After Cross Validation: `0.8904` (Random Forest)
  - After Hyperparameter Tuning: `0.85` (AdaBoost)

---

### `churn-4.ipynb`
- Dropped: `'RowNumber'`, `'CustomerId'`, `'Surname'`
- Retained: `'Geography'` as it added valuable information
- Used `resample()` for class balancing
- Data split: 70:30
- Cross-validation: 5-fold
- **Highest Accuracy**: 
  - Before Cross Validation: `0.9481` (Random Forest)
  - After Cross Validation: `0.9276` (Random Forest)
  - After Hyperparameter Tuning: `0.95` (Random Forest)

---

### `churn-5.ipynb`
- Same steps as `churn-4.ipynb`, but:
  - Train-test split changed to **80:20**
  - Minor improvements observed in model performances
- **Highest Accuracy**: 
  - Before Cross Validation: `0.9570` (Random Forest)
  - After Cross Validation: `0.9367` (Random Forest)
  - After Hyperparameter Tuning: `0.96` (Random Forest)

---

## Techniques Used

- StandardScaler for Feature Scaling  
- LabelEncoder for Categorical Conversion  
- Resample and SMOTE for Class Imbalance  
- GridSearchCV for Hyperparameter Tuning  
- StratifiedKFold and Cross-Validation  
- Performance Metrics: Accuracy, Precision, Recall, F1-Score

---

## Technologies

- Python, Jupyter Notebook  
- Pandas, NumPy, Scikit-learn, imbalanced-learn  
- Matplotlib, Seaborn

---

> Each notebook serves as an experiment to evaluate how changes in preprocessing, sampling, and validation impact model performance.
