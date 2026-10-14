# CodeAlpha_IrisFlowerClassification

Task 1 of the **CodeAlpha Data Science Internship**: classify Iris flowers (*setosa*, *versicolor*, *virginica*) from their sepal and petal measurements.

## Approach
1. **Data**: the classic Iris dataset (150 samples, 4 features, perfectly balanced, no missing values), provided by CodeAlpha as `Iris.csv` (the `Id` column is dropped because it is only a row counter).
2. **EDA**: pairplot, correlation heatmap and per-species boxplots (`outputs/01-03`).
3. **Split**: 80/20 stratified train/test split (`random_state=42`).
4. **Model comparison**: Logistic Regression, KNN, SVM (RBF), Decision Tree and Random Forest, scored with 5-fold stratified cross-validation on the training set.
5. **Tuning**: `GridSearchCV` for SVM, KNN and Random Forest.
6. **Evaluation**: the best model is scored once on the untouched test set (accuracy, precision/recall/F1, confusion matrix).

## Results
| Stage | Result |
|---|---|
| Best CV model | SVM (RBF), C=1, gamma=0.1, CV accuracy 0.975 |
| Test accuracy | **0.967** (29 of 30 correct) |
| Weak spot | one versicolor flower predicted as virginica |

Petal length and petal width are the most informative features. Setosa is linearly separable from the other two species, while versicolor and virginica overlap slightly, which is where the remaining error comes from.

Note: with only 30 test samples, one misclassification moves accuracy by about 3 points, so differences between the top models are within noise.

## Run it
```bash
pip install -r requirements.txt
python iris_classification.py
```
All charts and `metrics.json` are written to `outputs/`.

## Files
- `Iris.csv`: dataset
- `iris_classification.py`: full pipeline
- `outputs/`: plots and metrics
- `requirements.txt`: dependencies
