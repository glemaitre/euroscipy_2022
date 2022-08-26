# %%
import seaborn as sns

sns.set_context("poster")

# %%
import pandas as pd

training_data = pd.read_csv("./train.csv")
# testing_data = pd.read_csv("./test.csv")

y = training_data[["id", "target"]].set_index("id")["target"]
X = training_data.drop(["target"], axis=1).set_index("id")
# X_test = testing_data.set_index("id")

# %%
X.info()

# %%
y.value_counts()

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# %%
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.preprocessing import OrdinalEncoder

preprocessor = make_column_transformer(
    (OrdinalEncoder(), make_column_selector(dtype_include=int)),
    remainder="passthrough",
)
preprocessor.fit(X_train)

# %%
categorical_features = [
    "ordinalencoder" in name for name in preprocessor.get_feature_names_out()
]

# %%
print("Training Vanilla HGBDT...")
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

model = make_pipeline(
    preprocessor,
    HistGradientBoostingClassifier(
        categorical_features=categorical_features,
        max_iter=10_000,
        early_stopping=True,
        random_state=0,
    ),
).fit(X_train, y_train)

# %%
print("Training Bagged HGBDT...")
from imblearn.ensemble import BalancedBaggingClassifier

model_bag = make_pipeline(
    preprocessor,
    BalancedBaggingClassifier(
        base_estimator=HistGradientBoostingClassifier(
            categorical_features=categorical_features,
            max_iter=10_000,
            early_stopping=True,
            random_state=0,
        ),
        n_estimators=50,
        n_jobs=-1,
        random_state=0,
    ),
).fit(X_train, y_train)

# %%
print("Training Calibrated Bagged HGBDT...")
from sklearn.calibration import CalibratedClassifierCV

model_bag_calibrated = make_pipeline(
    preprocessor,
    CalibratedClassifierCV(
        BalancedBaggingClassifier(
            base_estimator=HistGradientBoostingClassifier(
                categorical_features=categorical_features,
                max_iter=10_000,
                early_stopping=True,
                random_state=0,
            ),
            n_estimators=50,
            n_jobs=-1,
            random_state=0,
        ),
        method="sigmoid",
        cv=3,
    ),
).fit(X_train, y_train)

# %%
from sklearn.metrics import classification_report

print(classification_report(y_test, model.predict(X_test)))
print(classification_report(y_test, model_bag.predict(X_test)))
print(classification_report(y_test, model_bag_calibrated.predict(X_test)))

# %%
from imblearn.metrics import classification_report_imbalanced

print(classification_report_imbalanced(y_test, model.predict(X_test)))
print(classification_report_imbalanced(y_test, model_bag.predict(X_test)))
print(classification_report_imbalanced(y_test, model_bag_calibrated.predict(X_test)))

# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import RocCurveDisplay, roc_curve

fig, ax = plt.subplots(figsize=(8, 8))
for name, est in {
    "Vanilla": model,
    "Bag": model_bag,
    "Calibrated bag": model_bag_calibrated,
}.items():
    disp = RocCurveDisplay.from_estimator(
        est,
        X_test,
        y_test,
        response_method="predict_proba",
        ax=ax,
        name=name,
    )
    fpr, tpr, thresholds = roc_curve(y_test, est.predict_proba(X_test)[:, 1])
    prediction_threshold_idx = (
        len(thresholds) - np.searchsorted(thresholds[::-1], 0.5) - 1
    )
    disp.ax_.scatter(
        fpr[prediction_threshold_idx],
        tpr[prediction_threshold_idx],
        s=200,
        edgecolor="black",
    )
handles, _ = disp.ax_.get_legend_handles_labels()
disp.ax_.legend(
    handles=handles,
    labels=["Vanilla", "Bag", "Calibrated bag"],
    bbox_to_anchor=(-0.1, 1.02, 1.2, 0.102),
    loc="lower left",
    ncol=3,
    mode="expand",
    borderaxespad=0.0,
)
_ = fig.suptitle("ROC Curve for HGBDT", y=1.05)

# %%
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve

fig, ax = plt.subplots(figsize=(8, 8))
for name, est in {
    "Vanilla": model,
    "Bag": model_bag,
    "Calibrated bag": model_bag_calibrated,
}.items():
    disp = PrecisionRecallDisplay.from_estimator(
        est,
        X_test,
        y_test,
        response_method="predict_proba",
        ax=ax,
        name=name,
    )
    precision, recall, thresholds = precision_recall_curve(
        y_test, est.predict_proba(X_test)[:, 1]
    )
    prediction_threshold_idx = np.searchsorted(thresholds, 0.5)
    disp.ax_.scatter(
        recall[prediction_threshold_idx],
        precision[prediction_threshold_idx],
        s=200,
        edgecolor="black",
    )
handles, _ = disp.ax_.get_legend_handles_labels()
disp.ax_.legend(
    handles=handles,
    labels=["Vanilla", "Bag", "Calibrated bag"],
    bbox_to_anchor=(-0.1, 1.02, 1.2, 0.102),
    loc="lower left",
    ncol=3,
    mode="expand",
    borderaxespad=0.0,
)
_ = fig.suptitle("PR Curve for HGBDT", y=1.05)

# %%
from sklearn.calibration import CalibrationDisplay

_, ax = plt.subplots(figsize=(8, 8))

for name, est in {
    "Vanilla HGBDT": model,
    "Bag of HGBDT": model_bag,
    "Calibrated bag of HGBDT": model_bag_calibrated,
}.items():
    disp = CalibrationDisplay.from_estimator(
        est,
        X_test,
        y_test,
        strategy="quantile",
        n_bins=10,
        ax=ax,
        name=name,
    )
    disp.ax_.set_ylabel(disp.ax_.get_ylabel().replace("(Positive class: 1)", ""))
    disp.ax_.set_xlabel("Mean confidence score")
    disp.ax_.legend(loc="upper left")
_ = disp.ax_.set_title("Reliability Diagram")

# %%


def gini(actual, pred):
    assert len(actual) == len(pred)
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float64)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.0
    return giniSum / len(actual)


def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)


# %%
print(f"Vanilla normalized Gini: {gini_normalized(y_test, model.predict(X_test)):.3f}")
print(f"Bag normalized Gini: {gini_normalized(y_test, model_bag.predict(X_test)):.3f}")
print(
    "Calibrated bag normalized Gini: "
    f"{gini_normalized(y_test, model_bag_calibrated.predict(X_test)):.3f}"
)

# %%
