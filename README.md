# 🏨 Hotel Booking Cancellation Prediction

## 📌 Project Overview

This project predicts whether a hotel booking will be **canceled (1)** or **not canceled (0)** using machine learning.
Cancellations cause significant revenue losses and inefficient resource allocation for hotels.
By analyzing booking patterns, we can identify high-risk cancellations and provide actionable insights for revenue management.

---

## 🎯 Objectives

* Build a machine learning model for **binary classification** (canceled vs not canceled).
* Handle **class imbalance** in cancellations.
* Perform **feature engineering & exploratory data analysis (EDA)**.
* Compare multiple models and tune hyperparameters.
* Use **SHAP explainability** to understand feature importance.
* Provide **business insights** for hotels.

---

## 📂 Dataset

* **Source**: Hotel Booking Cancellation Dataset from kaggle
* **Target Variable**: `is_canceled`
---

## 🔍 Exploratory Data Analysis (EDA)

✔️ Distribution of canceled vs non-canceled bookings
✔️ Correlation between features and cancellations
✔️ Trends: Longer lead times and higher prices → higher cancellation rates

---

## ⚙️ Approach & Pipeline

1. **Data Preprocessing**

   * Missing value imputation
   * Encoding categorical variables
   * Feature scaling
   * Train-test split

2. **Feature Engineering**

   * Extracted temporal features (month, season)
   * Interaction features (lead\_time × adr)
   * Balanced dataset with **SMOTE**

3. **Modeling**

   * Baseline: Logistic Regression, KNN
   * Advanced: Decision Trees, Random Forest, XGBoost
   * Hyperparameter tuning with GridSearchCV

4. **Evaluation Metrics**

   * Accuracy, ROC-AUC, Precision, Recall, F1
   * Confusion Matrix
   * Precision-Recall tradeoff

---

## 📊 Results

* **Best Model**: **XGBoost (tuned)**
* **Accuracy**: \~88–89%
* **ROC-AUC**: \~0.94
* **Insights**:

  * Long lead time bookings → higher chance of cancellation.
  * No-deposit bookings cancel more frequently.
  * Resort hotels show higher cancellation rates compared to city hotels.

---

## 🧠 Explainability (SHAP)

* Used SHAP values to interpret predictions.

---

## 📦 Installation & Usage

Clone the repository:

```bash
git clone https://github.com/fn12/booking_cancellation_prediction.git
cd booking_cancellation_prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Jupyter Notebook for training & evaluation:

```bash
jupyter notebook booking.ipynb
```

---

## 📌 Future Improvements

* Try **LightGBM / CatBoost** for faster training.
* Optimize with **cost-sensitive learning** (penalize false negatives).
* Deploy as a **REST API (FastAPI/Flask)**.
* Add **real-time dashboard** for hotel managers.

---

## 📝 License

This project is licensed under the **MIT License** – feel free to use and modify.

---

## 🙌 Acknowledgments

* Dataset: Kaggle
* Libraries: `scikit-learn`, `xgboost`, `shap`, `matplotlib`, `seaborn`, `pandas`

