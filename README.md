# 🍷 Wine Quality Prediction

---

## 📌 Overview

This project applies machine learning to predict the quality of red wine based on its chemical characteristics. Three classification models — **Random Forest**, **Stochastic Gradient Descent (SGD)**, and **Support Vector Classifier (SVC)** — are trained, evaluated, and compared on the WineQT dataset.

The study demonstrates how physicochemical features such as alcohol content, volatile acidity, sulphates, and density can serve as reliable predictors of perceived wine quality.

---

## 📁 Project Structure

```
wine-quality-prediction/
│
├── WineQT.csv                    # Dataset
├── wine_quality_prediction.py    # Main script
│
├── outputs/
│   ├── eda_plots.png             # Exploratory Data Analysis plots
│   ├── confusion_matrices.png    # Model confusion matrices
│   ├── feature_importance.png    # Random Forest feature importances
│   └── accuracy_comparison.png   # Model accuracy comparison
│
└── README.md
```

---

## 📊 Dataset

| Property        | Details                          |
|-----------------|----------------------------------|
| **Source**      | WineQT (Red Wine subset)         |
| **Samples**     | 1,143                            |
| **Features**    | 11 physicochemical attributes    |
| **Target**      | Quality score (3 – 8)            |
| **Missing Data**| None                             |

### Features

| Feature                | Description                              |
|------------------------|------------------------------------------|
| `fixed acidity`        | Tartaric acid content                    |
| `volatile acidity`     | Acetic acid — excess causes vinegar taste|
| `citric acid`          | Adds freshness and flavour               |
| `residual sugar`       | Sugar remaining after fermentation       |
| `chlorides`            | Salt content                             |
| `free sulfur dioxide`  | Prevents microbial growth                |
| `total sulfur dioxide` | Free + bound SO₂                         |
| `density`              | Mass per unit volume                     |
| `pH`                   | Measure of acidity/alkalinity            |
| `sulphates`            | Antimicrobial and antioxidant additive   |
| `alcohol`              | Alcohol percentage by volume             |
| `quality` *(target)*   | Expert quality rating (3–8)              |

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/your-username/wine-quality-prediction.git
cd wine-quality-prediction

# Install dependencies
pip install pandas numpy scikit-learn seaborn matplotlib
```

---

## 🚀 Usage

```bash
python wine_quality_prediction.py
```

Ensure `WineQT.csv` is in the same directory. All output plots will be saved automatically.

---

## 🧠 Methodology

### 1. Exploratory Data Analysis
- Quality score distribution
- Feature correlation heatmap
- Boxplots of density and volatile acidity vs quality

### 2. Preprocessing
- Dropped non-predictive `Id` column
- Stratified 80/20 train-test split
- Applied `StandardScaler` normalization for SGD and SVC

### 3. Models Trained

| Model | Key Configuration |
|-------|-------------------|
| **Random Forest** | `n_estimators=150`, `random_state=42` |
| **SGD Classifier** | `max_iter=1000`, `random_state=42` |
| **Support Vector Classifier** | `kernel='rbf'`, `C=10`, `gamma='scale'` |

> Random Forest was trained on raw features; SGD and SVC on scaled features.

---

## 📈 Results

| Model | Accuracy |
|-------|----------|
| 🥇 **Random Forest** | **71.18%** |
| 🥈 Support Vector Classifier | 64.19% |
| 🥉 Stochastic Gradient Descent | 57.64% |

### Key Findings
- **Random Forest** achieved the highest accuracy, demonstrating the advantage of ensemble methods for non-linear relationships.
- **Top predictive features**: `alcohol`, `sulphates`, `volatile acidity`, `total sulfur dioxide`, `density`.
- All models showed lower performance on rare quality classes (3 and 8) due to severe class imbalance.

---

## 📉 Visualisations

| Plot | Description |
|------|-------------|
| `eda_plots.png` | Quality distribution, correlation heatmap, feature boxplots |
| `confusion_matrices.png` | Prediction accuracy per class for all 3 models |
| `feature_importance.png` | Ranked feature importances from Random Forest |
| `accuracy_comparison.png` | Side-by-side model accuracy bar chart |

---

## 🔮 Future Improvements

- Apply **SMOTE** oversampling to handle class imbalance
- Reframe as a **3-class problem** (Low / Medium / High quality)
- Experiment with **XGBoost** and **LightGBM**
- Perform **hyperparameter tuning** via `GridSearchCV`
- Extend analysis to **white wine** data for cross-varietal comparison

---

## 🛠️ Tech Stack

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | Model training and evaluation |
| `seaborn` | Statistical visualisation |
| `matplotlib` | Plot rendering and export |

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙋‍♂️ Author

**Sania Aftab**
---
