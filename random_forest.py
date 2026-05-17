import json
import pickle
import sys

import pandas as pd
import numpy as np
from collections import Counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

from standard_preprocessing import get_standard_dataset
from feature_preprocessing import get_feature_dataset

def test_model_basic(x_train, x_test, y_train, y_test):
    print("\nTesting basic Random Forest model...")
    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)

    y_pred = rf.predict(x_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1-score: {f1 * 100:.2f}%")

    # record results
    model_type = "RF - TEST"
    model_data = [acc, prec, recall, f1]
    save_results(model_type, "base-test", model_data)

    # generate confusion matrix
    generate_cm(y_test, y_pred, "TEST-RF")


def fine_tuning(x_train, y_train):
    print("\nFine-tuning Random Forest model...")

    # create rf model with no params
    rf = RandomForestClassifier(random_state=42)

    param_grid = [{
        "n_estimators": [50, 100],
        "max_depth": [5, 10, None],
        "min_samples_leaf": [1, 5, 10],
        "min_samples_split": [2, 6, 10]
    }]

    # five-fold - use StratifiedKFold to avoid imbalanced class distribution
    ff = StratifiedKFold(n_splits=5, shuffle=True, random_state=41)

    grid_search = GridSearchCV(rf, param_grid, cv=ff, scoring='accuracy', refit=True, n_jobs=-1, verbose=3)
    # n_jobs = -1 : run on all available cores
    # verbose = 2 : gives update each time fold finishes

    grid_search.fit(x_train, y_train)

    print(f"Best params: {grid_search.best_params_}")
    print(f"Best Object: {grid_search.best_estimator_}")
    print(f"Best Accuracy Score: {grid_search.best_score_}")

    cv_res = grid_search.cv_results_
    print(cv_res.keys())

    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df = results_df[
        ['param_n_estimators', 'param_max_depth', 'param_min_samples_leaf', 'param_min_samples_split', 'mean_test_score',
         'std_test_score', 'rank_test_score']].sort_values(by='rank_test_score')
    print(results_df.head())

    results_df.to_csv(f'results\\rf_fine_tune_results.csv', index=False)


def assess_model(x_train, x_test, y_train, y_test):
    """
    1. get best model params
    2. feed them into model
    3. train model
    4. test model
    5. record performance
    """
    print("\nAssessing Random Forest model...")

    # get best params
    dir = f"results\\rf_fine_tune_results.csv"
    df = pd.read_csv(dir)
    optimal_params = df[['param_n_estimators', 'param_max_depth', 'param_min_samples_leaf', 'param_min_samples_split']].iloc[0]
    params = optimal_params.to_list()
    if pd.isna(params[1]):
        params[1] = None
    print(params)

    rf = RandomForestClassifier(n_estimators=int(params[0]), max_depth=params[1], min_samples_leaf=int(params[2]), min_samples_split=int(params[3]), random_state=42)

    rf.fit(x_train, y_train)

    y_pred = rf.predict(x_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1-score: {f1 * 100:.2f}%")

    # record results
    model_type = "RF"
    model_data = [acc, prec, recall, f1]

    save_results(model_type, "assess", model_data)

    # # create confusion matrix
    # generate_cm(y_test, y_pred, model_type)
    #
    # # metric comparison graph
    metric_comparison(model_data, model_type)

    report = classification_report(y_test, y_pred, target_names=["light", "moderate-vigorous", "sedentary", "sleep"], output_dict=True)
    print(report)

    # class comparison graph
    class_comparison(report, model_type)

    # predicted class distribution graph
    # predict_class_distribution()

    # save model
    with open(f'models\\rf.pkl', 'wb') as file:
        pickle.dump(rf, file)


def save_results(model_type, results_type, metrics):
    model_data = {
        "accuracy": metrics[0],
        "precision": metrics[1],
        "recall": metrics[2],
        "f1-score": metrics[3]
    }

    if results_type == "assess":
        path = r"results/model_results.json"
        with open(path, "r") as file:
            model_rs = json.load(file)

    elif results_type == "base-test":
        path = r"results/model_base_test_results.json"
        with open(path, "r") as file:
            model_rs = json.load(file)

    model_rs['model_results'][model_type] = model_data

    with open(path, "w") as file:
        json.dump(model_rs, file, indent=4)


def generate_cm(y_test, y_pred, model_type):
    labels = ["light", "moderate-vigorous", "sedentary", "sleep"] # 0, 1, 2, 3

    cm = confusion_matrix(y_test, y_pred)
    cm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt=".1%", cmap="viridis", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{model_type} Confusion Matrix")
    plt.savefig(f"graphs\\{model_type}_cm.png")
    plt.show()


def metric_comparison(model_data, model_type):
    # bar chart of all metrics
    print(model_data)
    metrics = ["Accuracy", "Precision", "Recall", "F1-score"]
    # colours = ["#6ac1cc", "#6e28a1", "#82d622", "#d6a622"]
    colours = ["#b3cde3", "#6497b1", "#005b96", "#03396c"]

    bars = plt.bar(metrics, model_data, color=colours)
    plt.bar_label(bars, fmt="%.3f")
    plt.title(f"{model_type} Metric Comparison")
    plt.xlabel("Metrics")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.savefig(f"graphs\\{model_type}_mc")
    plt.show()


def class_comparison(cr, model_type):
    # per-class precision, recall, f1-score bar chart
    # 2x2 grid

    labels = ["light", "moderate-vigorous", "sedentary", "sleep"]
    metrics = ["Precision", "Recall", "F1-score"]
    colours = ["#6497b1", "#005b96", "#03396c"]
    # colours = ["#6e28a1", "#82d622", "#d6a622"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for idx, l in enumerate(labels):
        i = idx // 2
        j = idx % 2

        p = cr[l]['precision']
        r = cr[l]['recall']
        f = cr[l]['f1-score']

        data = [p, r, f]
        bars = axes[i, j].bar(metrics, data, color=colours)
        axes[i, j].bar_label(bars, fmt="%.3f")
        axes[i, j].set_title(l)
        axes[i, j].set_ylabel("Score")
        axes[i, j].set_xlabel("Metric")
        axes[i, j].set_ylim(0, 1)

    fig.suptitle(f"{model_type} Class Comparison")
    plt.tight_layout()
    plt.savefig(f"graphs\\{model_type}_cc")
    plt.show()



def predict_class_distribution(y_pred, model_type):
    # bar chart of predicted label distribution

    label_counts = Counter(y_pred)

    pass

if __name__ == "__main__":

    x_train, x_test, y_train, y_test = get_feature_dataset()

    # test_model_basic(x_train, x_test, y_train, y_test)

    # fine_tuning(x_train, y_train)

    assess_model(x_train, x_test, y_train, y_test)