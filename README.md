# 🧠 Fintech Firm AI Transition Strategy - End-to-End Data Science Project

This project demonstrates how a fintech firm can leverage data science to transition into an **AI-first strategy**, using classification, clustering, and feature engineering to predict bankruptcy, identify risk, and derive actionable insights.

---

## 📌 Objective

To analyze financial data of a fintech firm and:

1. Preprocess and clean noisy financial records
2. Segment companies based on financial health using clustering
3. Engineer business-specific features to identify risk
4. Build ML models to predict bankruptcy risk
5. Evaluate strategies and insights for business decisions

---

## 📁 Dataset

- Source: Provided CSV file (`data.csv`)
- Shape: `6819 rows × 96 columns`
- Target Variable: `Bankrupt?`

---

## 🧹 Step 1–3: Data Preprocessing & Cleaning

- Handled missing values using `SimpleImputer`
- Normalized numerical features using `StandardScaler`
- Checked for nulls, outliers, and structure using `info()`, `describe()`

---

## 📊 Step 4: Business Strategy Formation (Clustering)

- Used **KMeans Clustering** on scaled data (`n_clusters=3`)
- Created **Cluster Profiles** with mean financial indicators
- Evaluated clusters using **Silhouette Score**
- Used **PCA** to improve cluster separation and interpretability

---

## 🏗️ Step 5: Feature Engineering

- Created key business-driven features:
  - `High_Debt_Equity`
  - `Low_Profit_Margin`
  - `High_Risk_Flag` (a combination indicator)

---

## 🤖 Step 6: Bankruptcy Classification (ML Models)

- Used **SMOTE** for handling class imbalance
- Models Used:
  - ✅ Random Forest (GridSearchCV tuned)
  - ✅ Logistic Regression (baseline)
- Evaluation:
  - F1 Score (Random Forest): `0.50` (for class 1 - bankrupt)
  - Accuracy: `96%`
- ✅ Final model saved as `bankruptcy_rf_model.pkl` using `joblib`

---

## 📉 Step 7: Regression Task

- ❌ Skipped due to lack of a continuous financial target (e.g., revenue)
- Can be added later with extended data

---

## 📈 Step 8: Clustering Evaluation & Insights

- Compared cluster counts (`k=2 to 5`) for Silhouette Score
- Used **PCA-based clustering** for improved separation
- Best Silhouette Score: `0.1824`

---

## 💾 Step 9: Final Output

- ✔ Cleaned & engineered dataset saved as: `cleaned_fintech_data.csv`
- ✔ Random Forest Model saved as: `bankruptcy_rf_model.pkl`
- ✔ End-to-end pipeline executed with logging & modular structure

---

## 📌 Technologies Used

- Python (Pandas, Scikit-learn, Seaborn, Matplotlib)
- SMOTE (from imblearn)
- GridSearchCV, KMeans, PCA
- Jupyter Notebook / Google Colab
- Joblib (for model saving)

---

## ✅ Final Outcome

This pipeline can help fintech firms:
- Predict bankruptcy risk early
- Identify high-risk company profiles
- Improve decision-making with AI insights

---

## 📂 Folder Structure

