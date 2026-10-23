# Income Prediction (Adult / Census Dataset)

Predicts whether an individual's income is **above or below $50K/year**, using the UCI Adult Census dataset, with a Keras deep learning model.

## Files

| File | Description |
|---|---|
| `Income.ipynb` | End-to-end notebook — cleaning, EDA, preprocessing pipeline, neural network training, evaluation, threshold tuning, feature importance, inference function |
| `adult.csv` | Input dataset (not included — place alongside the notebook to re-run) |

## Dataset

32,561 records with demographic and employment features: `age`, `workclass`, `fnlwgt`, `education`, `education.num`, `marital.status`, `occupation`, `relationship`, `race`, `sex`, `capital.gain`, `capital.loss`, `hours.per.week`, `native.country`. Target: `income` (`<=50K` / `>50K`).

**Class balance:** imbalanced — about 76% `<=50K` vs 24% `>50K`.

## Pipeline

1. **Missing values** — `'?'` placeholders replaced with `NaN`; missing `workclass`, `occupation`, and `native.country` imputed with the column mode.
2. **EDA** — target distribution, descriptive statistics for numeric features.
3. **Target encoding** — `income` mapped to `1` (`>50K`) / `0` (`<=50K`).
4. **Split** — stratified 70/15/15 train/validation/test split.
5. **Preprocessing pipeline** — `ColumnTransformer` with `StandardScaler` on numeric features and `OneHotEncoder` on categorical features, fit on training data only.
6. **Class imbalance** — class weights computed and passed to the loss function during training.
7. **Model** — feed-forward network (64 → 32 → 16 → 1 units) with Batch Normalization and Dropout, `binary_crossentropy` loss, Adam optimizer.
8. **Training** — `EarlyStopping`, `ReduceLROnPlateau`, and `ModelCheckpoint` callbacks; tracked accuracy, precision, recall, and AUC.
9. **Evaluation** — classification report, ROC-AUC, confusion matrix, ROC curve, precision-recall curve on the held-out test set.
10. **Threshold tuning** — scanned thresholds to maximize F1 on the test set.
11. **Interpretability** — permutation feature importance for the neural network.
12. **Inference function** — `predict_income()` wraps the fitted preprocessor and model for scoring new raw records.

## Results (test set)

| | Default threshold (0.5) | Tuned threshold (0.65) |
|---|---|---|
| Precision (>50K) | 0.56 | 0.67 |
| Recall (>50K) | 0.86 | 0.73 |
| F1 (>50K) | 0.67 | 0.70 |
| Accuracy | 0.80 | 0.85 |
| ROC-AUC | 0.90 | — |

The tuned threshold gives a better balance of precision and recall for the minority (`>50K`) class.

## Usage

Open `Income.ipynb` with `adult.csv` in the same directory and run all cells top to bottom.

```python
prediction_output = predict_income(raw_customer_df, model, preprocessor)
```

## Limitations

- `capital.gain` and `capital.loss` are highly skewed (mostly zero) and are only standard-scaled, not transformed — a log transform could help a linear-ish model separate the non-zero cases better.
- `fnlwgt` (a census sampling weight, not a demographic attribute) is included as a numeric feature as-is; it's often dropped or treated specially in analyses of this dataset.
- The train/val/test split is a single random stratified split, not cross-validated, so metrics carry some sampling variance.
