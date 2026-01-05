# 🚦 Accident Severity Prediction (ML Engineer Assessment)
📌 Overview

This project analyzes a large-scale U.S. road accidents dataset and builds machine learning models to predict accident severity.
The goal is to understand the factors contributing to accident severity and develop a reliable predictive model while handling real-world challenges such as class imbalance, large data volume, and feature heterogeneity.

The final solution follows a full ML lifecycle: data cleaning, feature engineering, model training, evaluation, and model serialization for future use.

---
# 📊 Dataset

Source: US Accidents Dataset (March 2023)
Size: ~7.7 million records
Target Variable: Severity (1–4)

| Value | Description                 |
| ----- | --------------------------- |
| 1     | Low impact                  |
| 2     | Moderate impact             |
| 3     | High impact                 |
| 4     | Severe / long traffic delay |

The dataset contains spatial, temporal, weather, and road-related features.

---
# 🧹 Data Preprocessing

* Key preprocessing steps include:

* Dropped non-informative identifiers (ID, descriptions, etc.)

* Converted timestamps and engineered temporal features:

      * Hour

      * Day of week

      * Month

* Created Accident Duration (minutes) from start and end times

* Handled outliers using percentile clipping

* Missing value treatment:

      * Numerical → median imputation

      * Categorical → "Unknown"

* Encoded categorical features using OrdinalEncoder

* Converted boolean features to numeric format

All preprocessing steps were designed to be reproducible and reusable.

---
# 📈 Exploratory Insights

Accident severity is highly imbalanced:

    * Severity 2 ≈ 80%

     * Severity 1 & 4 together < 4%

Accuracy alone is misleading → Macro F1 and recall are more meaningful

Temporal and weather-related features show strong influence on severity

---
# 🤖 Models Trained
1️⃣ Logistic Regression (Baseline)

* Used as a baseline for comparison

* Applied class_weight="balanced"

* Strength: Simple, interpretable

* Limitation: Linear model struggles with non-linear relationships

Result:
Low accuracy but reasonable recall for minority classes, highlighting class imbalance issues.
---
2️⃣ Random Forest (Final Model)

* Trained on a stratified sample for computational efficiency

* Handles non-linear interactions effectively

* Robust to noise and imbalance

* Key Results (Validation):

      * Accuracy ≈ 0.67

      * Macro F1 ≈ 0.47

* High recall for critical severity levels (1 & 4)

* Stable train / validation / test performance → no overfitting

✅ Selected as the final model

---

🏆 Model Comparison

| Model               | Validation Accuracy | Validation Macro F1 |
| ------------------- | ------------------- | ------------------- |
| Logistic Regression | ~0.35               | ~0.25               |
| Random Forest       | ~0.67               | ~0.47               |

---
# 💾 Model Serialization

* Trained models and preprocessing artifacts were saved using joblib for reuse:
  
      * random_forest_severity_model.pkl

      * categorical_encoder.pkl

This ensures consistent predictions on future data without retraining

---
# 🔮 Future Predictions

* The saved Random Forest model can be loaded and used to predict accident severity for new incoming data, provided the same preprocessing pipeline is applied.

This enables:

    * Batch predictions

    * Real-time inference via APIs

    * Integration into traffic monitoring systems

---
# 🛠 Tech Stack

* Python

* Pandas, NumPy

* Scikit-learn

* Google Colab

* Joblib

---
# 🧾 Requirements & Setup
🔧 Environment

* Python: 3.9 or higher

* Platform: Tested on Google Colab and local Python environments

📦 Required Libraries

All dependencies are listed in requirements.txt.
Install them using
```
pip install -r requirements.txt
```
---
 ## 👨‍💻 Author

Abhijith Babu
Passionate about ML & AI 🚀

📌 GitHub: [https://github.com/AbhijithBabu12]

📌 LinkedIn: [https://www.linkedin.com/in/abhijith-babu-856170201/]
