import json
import pickle
import sys

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

from standard_preprocessing import get_standard_dataset
from feature_preprocessing import get_feature_dataset

def test_model_basic(x_train, x_test, y_train, y_test, dataset_type):
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
    model_type = "Random Forest - TEST"
    model_data = [dataset_type, acc, prec, recall, f1]
    save_results(model_type, model_data)

    # generate confusion matrix
    generate_cm(y_test, y_pred, "TEST-RF", dataset_type)


def fine_tuning(x_train, y_train, dataset_type):
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

    results_df.to_csv(f'results\\rf_fine_tune_results_{dataset_type}.csv', index=False)


def assess_model(x_train, x_test, y_train, y_test, dataset_type):
    """
    1. get best model params
    2. feed them into model
    3. train model
    4. test model
    5. record performance
    """

    # get best params
    dir = f"results\\rf_fine_tune_results_{dataset_type}.csv"
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
    model_type = "Random Forest"
    model_data = [dataset_type, acc, prec, recall, f1]

    save_results(model_type, model_data)

    # create confusion matrix
    generate_cm(y_test, y_pred, "rf", dataset_type)

    # save model
    with open(f'models\\rf.pkl', 'wb') as file:
        pickle.dump(rf, file)


def save_results(model_type, metrics):
    model_data = {
        "dataset": metrics[0],
        "accuracy": metrics[1],
        "precision": metrics[2],
        "recall": metrics[3],
        "f1-score": metrics[4]
    }

    if metrics[0] == "feature":
        path = r"results/model_feature_results.json"
        with open(path, "r") as file:
            model_rs = json.load(file)

    elif metrics[0] == "standard":
        path = r"results/model_standard_results.json"
        with open(path, "r") as file:
            model_rs = json.load(file)

    model_rs['model_results'][model_type] = model_data

    with open(path, "w") as file:
        json.dump(model_rs, file, indent=4)


def generate_cm(y_test, y_pred, model_type, dataset_type):
    # remap labels
    labels = ["sleep", "sitting", "walking", "bicycling", "mixed-activity", "standing", "manual-work", "sports"]

    # cm = confusion_matrix(y_test, y_pred)
    #
    # plt.figure(figsize=(12, 10))
    # sns.heatmap(cm, annot=True, fmt="d", cmap="viridis", xticklabels=labels, yticklabels=labels)
    # plt.xlabel("Predicted")
    # plt.ylabel("True")
    # plt.title(f"{model_type} Confusion Matrix ({dataset_type})")
    # plt.savefig(f"graphs\\{model_type}_{dataset_type}_cm.png")
    # plt.show()

    cm = confusion_matrix(y_test, y_pred)
    cm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt=".1%", cmap="viridis", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{model_type} Confusion Matrix ({dataset_type})")
    plt.savefig(f"graphs\\{model_type}_{dataset_type}_cm.png")
    plt.show()


if __name__ == "__main__":
    dataset_type = "standard"
    # dataset_type = "feature"

    if dataset_type == "standard":
        x_train, x_test, y_train, y_test = get_standard_dataset()

        # test_model_basic(x_train, x_test, y_train, y_test, dataset_type)

        # fine_tuning(x_train, y_train, dataset_type)

        assess_model(x_train, x_test, y_train, y_test, dataset_type)

    elif dataset_type == "feature":
        x_train, x_test, y_train, y_test = get_feature_dataset()

        test_model_basic(x_train, x_test, y_train, y_test, dataset_type)

        # fine_tuning(x_train, y_train, dataset_type)

        assess_model(x_train, x_test, y_train, y_test, dataset_type)