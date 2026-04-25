# 🏠 House Price Prediction - End-to-End ML Pipeline

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Scikit-Learn](https://img.shields.io/badge/ML-ScikitLearn-orange)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)

---

## 📌 Problem Statement

The goal of this project is to predict house prices based on various features such as location, size, quality, and other housing attributes.

This is a supervised machine learning regression problem where the target variable is `SalePrice`.

---

## 📊 Dataset

* Source: Kaggle - House Prices: Advanced Regression Techniques
* Number of samples: ~1460
* Features: 70+ (numerical + categorical)

Key feature types:

* Numerical (e.g., `GrLivArea`, `LotArea`)
* Categorical (e.g., `Neighborhood`, `HouseStyle`)
* Ordinal (e.g., `ExterQual`, `KitchenQual`)

---

## ⚙️ Project Structure

```
house-price-ml/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│
├── models/
├── notebooks/
├── requirements.txt
└── README.md
```

---

## 🛠️ Installation

```bash
git clone https://github.com/ahmedsherif999/house-price-ml.git
cd house-price-ml

python -m venv venv
source venv/bin/activate  # Mac/Linux

pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Preprocess data

```bash
python src/preprocess.py
```

### 2. Train model

```bash
python src/train.py
```

### 3. Predict

```bash
python src/predict.py
```

---

## 🧠 Approach

* Data Cleaning & Missing Value Handling
* Feature Engineering (TotalSF, etc.)
* Skewness correction using log transformation
* Hybrid Encoding:

  * Ordinal Encoding for ordered features
  * One-Hot Encoding for nominal features
* Pipeline using `ColumnTransformer`

---

## 🤖 Models Used

* Random Forest Regressor
* XGBoost Regressor

---

## 📈 Results

| Model        | RMSE |
| ------------ | ---- |
| RandomForest | 0.02 |
| XGBoost      | 0.13 |

> Best model: XGBoost

---

## 📊 Key Insights

* `OverallQual` and `GrLivArea` are among the most important features
* Log transformation significantly improved model performance
* Feature engineering improved generalization

---

## 🧰 Technologies Used

* Python
* Pandas / NumPy
* Scikit-learn
* XGBoost
* Matplotlib / Seaborn

---

## 👨‍💻 Author

Ahmed Sherif
GitHub: https://github.com/ahmedsherif999
