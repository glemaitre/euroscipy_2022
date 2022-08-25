# %%
import seaborn as sns

sns.set_context("poster")

# %%
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_classes=2,
    n_informative=2,
    n_redundant=0,
    weights=[0.9, 0.1],
    random_state=0,
)

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 8))
scatter = plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.axis("square")
plt.legend(
    handles=scatter.legend_elements()[0], labels=["Majority", "Minority"], title="Class"
)
_ = plt.title("A toy imbalanced dataset ")

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# %%
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression().fit(X_train, y_train)

# %%
def fmt(x):
    s = f"{x*100:.1f}"
    return rf"{s} \%" if plt.rcParams["text.usetex"] else f"{s} %"


# %%
from sklearn.inspection import DecisionBoundaryDisplay

_, ax = plt.subplots(figsize=(8, 8))
disp = DecisionBoundaryDisplay.from_estimator(
    lr,
    X,
    plot_method="contour",
    response_method="predict_proba",
    levels=10,
    ax=ax,
)
scatter = disp.ax_.scatter(X[:, 0], X[:, 1], c=y)
disp.ax_.clabel(disp.surface_, disp.surface_.levels, inline=True, fmt=fmt, fontsize=10)
disp.ax_.axis("square")
disp.ax_.set_title("Vanilla Logistic Regression")
disp.ax_.set_xlabel("$x_1$")
disp.ax_.set_ylabel("$x_2$")
_ = disp.ax_.legend(
    handles=scatter.legend_elements()[0], labels=["Majority", "Minority"], title="Class"
)

# %%
lr.set_params(class_weight="balanced").fit(X_train, y_train)

_, ax = plt.subplots(figsize=(8, 8))
disp = DecisionBoundaryDisplay.from_estimator(
    lr,
    X,
    plot_method="contour",
    response_method="predict_proba",
    levels=10,
    ax=ax,
)
disp.ax_.scatter(X[:, 0], X[:, 1], c=y)
disp.ax_.clabel(disp.surface_, disp.surface_.levels, inline=True, fmt=fmt, fontsize=10)
disp.ax_.axis("square")
disp.ax_.set_title("Balanced Logistic Regression")
disp.ax_.set_xlabel("$x_1$")
disp.ax_.set_ylabel("$x_2$")
_ = disp.ax_.legend(
    handles=scatter.legend_elements()[0], labels=["Majority", "Minority"], title="Class"
)

# %%
from imblearn.pipeline import make_pipeline as imblearn_make_pipeline
from imblearn.under_sampling import RandomUnderSampler

weighted_lr = imblearn_make_pipeline(
    RandomUnderSampler(random_state=0), LogisticRegression()
).fit(X_train, y_train)

# %%
from sklearn.inspection import DecisionBoundaryDisplay

_, ax = plt.subplots(figsize=(8, 8))
disp = DecisionBoundaryDisplay.from_estimator(
    weighted_lr,
    X,
    plot_method="contour",
    response_method="predict_proba",
    levels=10,
    ax=ax,
)
disp.ax_.scatter(X[:, 0], X[:, 1], c=y)
disp.ax_.clabel(disp.surface_, disp.surface_.levels, inline=True, fmt=fmt, fontsize=10)
disp.ax_.axis("square")
disp.ax_.set_title("Resampled Logistic Regression")
disp.ax_.set_xlabel("$x_1$")
disp.ax_.set_ylabel("$x_2$")
_ = disp.ax_.legend(
    handles=scatter.legend_elements()[0], labels=["Majority", "Minority"], title="Class"
)

# %%
X, y = make_classification(
    n_samples=1_000,
    n_features=2,
    n_classes=2,
    n_informative=2,
    n_redundant=0,
    weights=[0.99, 0.01],
    random_state=0,
)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# %%
lr = LogisticRegression().fit(X_train, y_train)
balanced_lr = LogisticRegression(class_weight="balanced").fit(X_train, y_train)
weighted_lr = imblearn_make_pipeline(
    RandomUnderSampler(random_state=0), LogisticRegression()
).fit(X_train, y_train)

# %%
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(8 * 3, 8))

for ax, (name, est) in zip(
    axs,
    {
        "Vanilla \nLogistic Regression": lr,
        "Balanced \nLogistic Regression": balanced_lr,
        "Resampled \nLogistic Regrssion": weighted_lr,
    }.items(),
):
    disp = DecisionBoundaryDisplay.from_estimator(
        est,
        X,
        plot_method="contour",
        response_method="predict_proba",
        levels=10,
        ax=ax,
    )
    scatter = disp.ax_.scatter(X[:, 0], X[:, 1], c=y)
    disp.ax_.clabel(
        disp.surface_, disp.surface_.levels, inline=True, fmt=fmt, fontsize=10
    )
    _ = disp.ax_.legend(
        handles=scatter.legend_elements()[0],
        labels=["Majority", "Minority"],
        title="Class",
    )
    disp.ax_.set_title(name)

# %%
import warnings
import sklearn
from imblearn.metrics import classification_report_imbalanced

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", category=sklearn.exceptions.UndefinedMetricWarning
    )
    print("Vanilla Logistic Regression")
    print(classification_report_imbalanced(y_test, lr.predict(X_test)))
    print("Balanced Logistic Regression")
    print(classification_report_imbalanced(y_test, balanced_lr.predict(X_test)))
    print("Resampled Logistic Regrssion")
    print(classification_report_imbalanced(y_test, weighted_lr.predict(X_test)))

# %%
from sklearn.metrics import class_likelihood_ratios

clr = class_likelihood_ratios(lr.predict(X_test), y_test)
print(
    f"""Vanilla Logistic Regression
    LR+ : {clr[0]:.2f} - LR- : {clr[1]:.2f}
    """
)
clr = class_likelihood_ratios(balanced_lr.predict(X_test), y_test)
print(
    f"""Balanced Logistic Regression
    LR+ : {clr[0]:.2f} - LR- : {clr[1]:.2f}
    """
)
clr = class_likelihood_ratios(weighted_lr.predict(X_test), y_test)
print(
    f"""Resampled Logistic Regression
    LR+ : {clr[0]:.2f} - LR- : {clr[1]:.2f}
    """
)

# %%
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(8 * 3, 8))

for ax, (name, est) in zip(
    axs,
    {
        "Vanilla \nLogistic Regression": lr,
        "Balanced \nLogistic Regression": balanced_lr,
        "Resampled \nLogistic Regrssion": weighted_lr,
    }.items(),
):
    disp = DecisionBoundaryDisplay.from_estimator(
        est,
        X_train,
        plot_method="contour",
        response_method="predict_proba",
        levels=10,
        ax=ax,
    )
    scatter = disp.ax_.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
    disp.ax_.clabel(
        disp.surface_, disp.surface_.levels, inline=True, fmt=fmt, fontsize=10
    )
    _ = disp.ax_.legend(
        handles=scatter.legend_elements()[0],
        labels=["Majority", "Minority"],
        title="Class",
    )
    disp.ax_.set_title(name)
    fig.suptitle("Training Data", y=1.05)

# %%
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(8 * 3, 8))

for ax, (name, est) in zip(
    axs,
    {
        "Vanilla \nLogistic Regression": lr,
        "Balanced \nLogistic Regression": balanced_lr,
        "Resampled \nLogistic Regrssion": weighted_lr,
    }.items(),
):
    disp = DecisionBoundaryDisplay.from_estimator(
        est,
        X_test,
        plot_method="contour",
        response_method="predict_proba",
        levels=10,
        ax=ax,
    )
    scatter = disp.ax_.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
    disp.ax_.clabel(
        disp.surface_, disp.surface_.levels, inline=True, fmt=fmt, fontsize=10
    )
    _ = disp.ax_.legend(
        handles=scatter.legend_elements()[0],
        labels=["Majority", "Minority"],
        title="Class",
        loc="lower right",
    )
    disp.ax_.set_title(name)
    fig.suptitle("Testing Data", y=1.05)

# %%
from sklearn.calibration import CalibrationDisplay

fig, ax = plt.subplots(figsize=(8, 8))

for name, est in {
        "Vanilla Logistic Regression": lr,
        "Balanced Logistic Regression": balanced_lr,
        "Resampled Logistic Regrssion": weighted_lr,
    }.items():
    disp = CalibrationDisplay.from_estimator(
        est,
        X_test,
        y_test,
        strategy="quantile",
        ax=ax,
        name=name,
    )
    disp.ax_.set_ylabel(disp.ax_.get_ylabel().replace("(Positive class: 1)", ""))
    disp.ax_.set_xlabel(disp.ax_.get_xlabel().replace("(Positive class: 1)", ""))
    disp.ax_.legend(loc="upper left")
ax.set_title("")

# %%
from imblearn.ensemble import BalancedBaggingClassifier

bag_lr = BalancedBaggingClassifier(
    n_estimators=10,
    base_estimator=LogisticRegression(),
    random_state=0,
).fit(X_train, y_train)

# %%
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(8 * 3, 8))

for ax, (name, est) in zip(
    axs,
    {
        "Vanilla \nLogistic Regression": lr,
        "Resampled \nLogistic Regrssion": weighted_lr,
        "Bag of \nLogistic Regression": bag_lr,
    }.items(),
):
    disp = DecisionBoundaryDisplay.from_estimator(
        est,
        X_test,
        plot_method="contour",
        response_method="predict_proba",
        levels=10,
        ax=ax,
    )
    scatter = disp.ax_.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
    disp.ax_.clabel(
        disp.surface_, disp.surface_.levels, inline=True, fmt=fmt, fontsize=10
    )
    _ = disp.ax_.legend(
        handles=scatter.legend_elements()[0],
        labels=["Majority", "Minority"],
        title="Class",
        loc="lower right",
    )
    disp.ax_.set_title(name)
    fig.suptitle("Testing Data", y=1.05)

# %%
from sklearn.calibration import CalibratedClassifierCV

calibrated_bag_lr = CalibratedClassifierCV(bag_lr, method="sigmoid").fit(X_train, y_train)

# %%
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(8 * 3, 8))

for ax, (name, est) in zip(
    axs,
    {
        "Vanilla \nLogistic Regression": lr,
        "Bag of \nLogistic Regression": bag_lr,
        "Calibrated bag of \nLogistic Regression": calibrated_bag_lr,
    }.items(),
):
    disp = DecisionBoundaryDisplay.from_estimator(
        est,
        X_test,
        plot_method="contour",
        response_method="predict_proba",
        levels=10,
        ax=ax,
    )
    scatter = disp.ax_.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
    disp.ax_.clabel(
        disp.surface_, disp.surface_.levels, inline=True, fmt=fmt, fontsize=10
    )
    _ = disp.ax_.legend(
        handles=scatter.legend_elements()[0],
        labels=["Majority", "Minority"],
        title="Class",
        loc="lower right",
    )
    disp.ax_.set_title(name)
    fig.suptitle("Testing Data", y=1.05)

# %%
X, y = make_classification(
    n_samples=5_000,
    n_features=2,
    n_classes=2,
    n_informative=2,
    n_redundant=0,
    weights=[0.95, 0.05],
    class_sep=1.5,
    random_state=10,
)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# %%
lr.fit(X_train, y_train)
bag_lr.fit(X_train, y_train)
calibrated_bag_lr.fit(X_train, y_train)

# %%
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(8 * 3, 8))

for ax, (name, est) in zip(
    axs,
    {
        "Vanilla \nLogistic Regression": lr,
        "Bag of \nLogistic Regression": bag_lr,
        "Calibrated bag of \nLogistic Regression": calibrated_bag_lr,
    }.items(),
):
    disp = DecisionBoundaryDisplay.from_estimator(
        est,
        X_test,
        plot_method="contour",
        response_method="predict_proba",
        levels=10,
        ax=ax,
    )
    scatter = disp.ax_.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
    disp.ax_.clabel(
        disp.surface_, disp.surface_.levels, inline=True, fmt=fmt, fontsize=10
    )
    _ = disp.ax_.legend(
        handles=scatter.legend_elements()[0],
        labels=["Majority", "Minority"],
        title="Class",
        loc="lower right",
    )
    disp.ax_.set_title(name)
    fig.suptitle("Testing Data", y=1.05)

# %%
fig, ax = plt.subplots(figsize=(8, 8))

for name, est in {
        "Vanilla Logistic Regression": lr,
        "Bag of Logistic Regression": bag_lr,
        "Calibrated bag of Logistic Regression": calibrated_bag_lr,
    }.items():
    disp = CalibrationDisplay.from_estimator(
        est,
        X_test,
        y_test,
        strategy="uniform",
        ax=ax,
        name=name,
    )
    disp.ax_.set_ylabel(disp.ax_.get_ylabel().replace("(Positive class: 1)", ""))
    disp.ax_.set_xlabel(disp.ax_.get_xlabel().replace("(Positive class: 1)", ""))
    disp.ax_.legend(loc="upper left")
ax.set_title("")

# %%
