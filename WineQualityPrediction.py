import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load Data ──────────────────────────────────────────────────────────────
df = pd.read_csv('WineQT.csv')
df.drop(columns=['Id'], inplace=True, errors='ignore')

print("Shape:", df.shape)
print("\nQuality distribution:\n", df['quality'].value_counts().sort_index())
print("\nMissing values:", df.isnull().sum().sum())

# ── 2. EDA Visualisations ─────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Wine Quality – Exploratory Data Analysis', fontsize=16, fontweight='bold')

# 2a. Quality distribution
sns.countplot(x='quality', data=df, palette='viridis', ax=axes[0, 0])
axes[0, 0].set_title('Quality Distribution')
axes[0, 0].set_xlabel('Quality Score')
axes[0, 0].set_ylabel('Count')

# 2b. Correlation heatmap
corr = df.corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5,
            ax=axes[0, 1], annot_kws={'size': 7})
axes[0, 1].set_title('Feature Correlation Heatmap')
axes[0, 1].tick_params(axis='x', rotation=45, labelsize=8)
axes[0, 1].tick_params(axis='y', rotation=0, labelsize=8)

# 2c. Density vs Quality
sns.boxplot(x='quality', y='density', data=df, palette='Set2', ax=axes[1, 0])
axes[1, 0].set_title('Density vs Quality')

# 2d. Volatile Acidity vs Quality
sns.boxplot(x='quality', y='volatile acidity', data=df, palette='Set3', ax=axes[1, 1])
axes[1, 1].set_title('Volatile Acidity vs Quality')

plt.tight_layout()
plt.savefig('eda_plots.png', dpi=150, bbox_inches='tight')
plt.show()
print("EDA plots saved.")

# ── 3. Preprocessing ──────────────────────────────────────────────────────────
X = df.drop('quality', axis=1)
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── 4. Train Three Classifiers ────────────────────────────────────────────────
models = {
    'Random Forest':               RandomForestClassifier(n_estimators=150, random_state=42),
    'Stochastic Gradient Descent': SGDClassifier(max_iter=1000, random_state=42),
    'Support Vector Classifier':   SVC(kernel='rbf', C=10, gamma='scale', random_state=42),
}

results = {}
for name, model in models.items():
    # Random Forest uses raw features; SGD & SVC use scaled features
    X_tr = X_train if name == 'Random Forest' else X_train_sc
    X_te = X_test  if name == 'Random Forest' else X_test_sc

    model.fit(X_tr, y_train)
    preds = model.predict(X_te)
    acc   = accuracy_score(y_test, preds)
    results[name] = {'model': model, 'preds': preds, 'accuracy': acc}

    print(f"\n{'='*50}")
    print(f"{name}  —  Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))

# ── 5. Confusion Matrices ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Confusion Matrices – Model Comparison', fontsize=15, fontweight='bold')

for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res['preds'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f"{name}\nAcc: {res['accuracy']:.4f}")
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()

# ── 6. Feature Importance (Random Forest) ─────────────────────────────────────
rf = results['Random Forest']['model']
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(x=importances.values, y=importances.index, palette='viridis')
plt.title('Random Forest – Feature Importances', fontsize=14, fontweight='bold')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

# ── 7. Accuracy Comparison ────────────────────────────────────────────────────
plt.figure(figsize=(8, 4))
short_names = ['Random\nForest', 'SGD', 'SVC']
accs = [results[n]['accuracy'] for n in models]
bars = plt.barh(short_names, accs, color=['#2ecc71', '#3498db', '#e74c3c'])
for bar, acc in zip(bars, accs):
    plt.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
             f'{acc:.4f}', va='center', fontweight='bold')
plt.xlim(0, 1.1)
plt.xlabel('Accuracy')
plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('accuracy_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

best = max(results, key=lambda k: results[k]['accuracy'])
print(f"\n✅ Best model: {best}  ({results[best]['accuracy']:.4f})")
