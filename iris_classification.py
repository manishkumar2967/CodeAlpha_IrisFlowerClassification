"""
CodeAlpha Data Science Internship - Task 1: Iris Flower Classification
Author: <Manish Kumar>

Pipeline:
1. Load data and run exploratory data analysis (EDA)
2. Train/test split (stratified)
3. Compare several classifiers with cross-validation
4. Tune the best candidates with GridSearchCV
5. Evaluate the final model on held-out test data
6. Save plots and a metrics report into ./outputs
"""
import os
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)
sns.set_theme(style="whitegrid")

# ---------------------------------------------------------------- 1. Load data
df = pd.read_csv("Iris.csv")          # dataset supplied by CodeAlpha
df = df.drop(columns="Id")            # Id is just a row counter, not a feature
df = df.rename(columns={"Species": "species"})
df["species"] = df["species"].str.replace("Iris-", "", regex=False)
class_names = sorted(df["species"].unique())
df["target"] = df["species"].map({c: i for i, c in enumerate(class_names)})
features = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
print("Duplicate rows:", df.duplicated().sum())

print("Shape:", df.shape)
print("Missing values:\n", df.isna().sum().to_string())
print("Class balance:\n", df["species"].value_counts().to_string())
print(df[features].describe().round(2).to_string())

# ---------------------------------------------------------------- 2. EDA plots
pair = sns.pairplot(df.drop(columns="target"), hue="species", corner=True)
pair.fig.suptitle("Iris pairplot by species", y=1.02)
pair.savefig(f"{OUT}/01_pairplot.png", dpi=150, bbox_inches="tight")
plt.close("all")

plt.figure(figsize=(6, 5))
sns.heatmap(df[features].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature correlation")
plt.tight_layout()
plt.savefig(f"{OUT}/02_correlation.png", dpi=150)
plt.close()

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, col in zip(axes.ravel(), features):
    sns.boxplot(data=df, x="species", y=col, ax=ax)
    ax.set_title(col)
plt.tight_layout()
plt.savefig(f"{OUT}/03_boxplots.png", dpi=150)
plt.close()

# ---------------------------------------------------------------- 3. Split
X, y = df[features], df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

# ---------------------------------------------------------------- 4. Compare models
models = {
    "Logistic Regression": Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=500))]),
    "KNN": Pipeline([("sc", StandardScaler()), ("m", KNeighborsClassifier())]),
    "SVM (RBF)": Pipeline([("sc", StandardScaler()), ("m", SVC(random_state=RANDOM_STATE))]),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_rows = []
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    cv_rows.append({"model": name, "cv_mean": scores.mean(), "cv_std": scores.std()})
cv_df = pd.DataFrame(cv_rows).sort_values("cv_mean", ascending=False)
print("\nCross-validation results:\n", cv_df.round(4).to_string(index=False))

plt.figure(figsize=(8, 4))
plt.barh(cv_df["model"], cv_df["cv_mean"], xerr=cv_df["cv_std"], color="steelblue")
plt.xlim(0.8, 1.02)
plt.xlabel("5-fold CV accuracy")
plt.title("Model comparison")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f"{OUT}/04_model_comparison.png", dpi=150)
plt.close()

# ---------------------------------------------------------------- 5. Tune
param_grids = {
    "SVM (RBF)": {"m__C": [0.1, 1, 10, 100], "m__gamma": ["scale", 0.01, 0.1, 1]},
    "KNN": {"m__n_neighbors": list(range(3, 16, 2)), "m__weights": ["uniform", "distance"]},
    "Random Forest": {"n_estimators": [100, 300], "max_depth": [None, 3, 5]},
}
tuned = {}
for name, grid in param_grids.items():
    gs = GridSearchCV(models[name], grid, cv=cv, scoring="accuracy")
    gs.fit(X_train, y_train)
    tuned[name] = gs
    print(f"Tuned {name}: best CV={gs.best_score_:.4f} params={gs.best_params_}")

best_name = max(tuned, key=lambda n: tuned[n].best_score_)
best_model = tuned[best_name].best_estimator_
print("\nSelected model:", best_name)

# ---------------------------------------------------------------- 6. Final evaluation
y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
print(f"\nTest accuracy: {acc:.4f}")
print(classification_report(y_test, y_pred, target_names=class_names))

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=class_names).plot(cmap="Blues")
plt.title(f"Confusion matrix - {best_name}")
plt.tight_layout()
plt.savefig(f"{OUT}/05_confusion_matrix.png", dpi=150)
plt.close()

# Feature importance via Random Forest (interpretability)
rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE).fit(X_train, y_train)
imp = pd.Series(rf.feature_importances_, index=features).sort_values()
plt.figure(figsize=(7, 4))
imp.plot.barh(color="seagreen")
plt.title("Random Forest feature importance")
plt.tight_layout()
plt.savefig(f"{OUT}/06_feature_importance.png", dpi=150)
plt.close()

with open(f"{OUT}/metrics.json", "w") as f:
    json.dump({
        "selected_model": best_name,
        "best_params": tuned[best_name].best_params_,
        "cv_accuracy": round(tuned[best_name].best_score_, 4),
        "test_accuracy": round(acc, 4),
        "classification_report": report,
        "feature_importance": imp.round(4).to_dict(),
    }, f, indent=2, default=str)

# ---------------------------------------------------------------- 7. Predict a new sample
sample = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=features)
print("\nExample prediction for", sample.values.tolist()[0], "->",
      class_names[best_model.predict(sample)[0]])
