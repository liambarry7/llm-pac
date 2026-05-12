import pickle

from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

from random_forest import save_results, generate_cm

from standard_preprocessing import get_standard_dataset
from feature_preprocessing import get_feature_dataset


def test_model_basic(x_train, x_test, y_train, y_test, dataset_type):
    knn = KNeighborsClassifier()
    knn.fit(x_train, y_train)

    y_pred = knn.predict(x_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1-score: {f1 * 100:.2f}%")

    # record results
    model_type = "kNN - TEST"
    model_data = [dataset_type, acc, prec, recall, f1]
    save_results(model_type, model_data)

    # generate confusion matrix
    generate_cm(y_test, y_pred, "TEST-kNN", dataset_type)


def fine_tuning(x_train, y_train, dataset_type):
    # create new mlp
    knn = KNeighborsClassifier()

    param_grid = [{
        'n_neighbors': [1, 3, 5, 7],
        'metric': ['euclidean', 'manhattan', 'minkowski'],
        'leaf_size': [20, 30, 40] # does not affect accuracy, only speed and memory usage
    }]

    # five-fold - use StratifiedKFold to avoid imbalanced class distribution
    ff = StratifiedKFold(n_splits=5, shuffle=True, random_state=41)

    grid_search = GridSearchCV(knn, param_grid, cv=ff, scoring='accuracy', refit=True, n_jobs=-1, verbose=3)
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
        ['param_n_neighbors', 'param_metric', 'param_leaf_size', 'mean_test_score',
         'std_test_score', 'rank_test_score']].sort_values(by='rank_test_score')

    print(results_df.head())

    results_df.to_csv(f'results\\knn_fine_tune_results_{dataset_type}.csv', index=False)

def assess_model(x_train, x_test, y_train, y_test, dataset_type):
    """
    1. get best model params
    2. feed them into model
    3. train model
    4. test model
    5. record performance
    """

    # get best params
    dir = f"results\\knn_fine_tune_results_{dataset_type}.csv"
    df = pd.read_csv(dir)
    optimal_params = df[['param_n_neighbors', 'param_metric', 'param_leaf_size']].iloc[0]
    params = optimal_params.to_list()

    knn = KNeighborsClassifier(n_neighbors=params[0], metric=params[1], leaf_size=params[2])
    knn.fit(x_train, y_train)

    y_pred = knn.predict(x_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1-score: {f1 * 100:.2f}%")

    # record results
    model_type = "kNN"
    model_data = [dataset_type, acc, prec, recall, f1]

    save_results(model_type, model_data)

    # create confusion matrix
    generate_cm(y_test, y_pred, "knn", dataset_type)

    # save model
    with open(f'models\\knn.pkl', 'wb') as file:
        pickle.dump(knn, file)


if __name__ == "__main__":
    # dataset_type = "standard"
    dataset_type = "feature"

    if dataset_type == "standard":
        x_train, x_test, y_train, y_test = get_standard_dataset()

        test_model_basic(x_train, x_test, y_train, y_test, dataset_type)

        # fine_tuning(x_train, y_train, dataset_type)

        # assess_model(x_train, x_test, y_train, y_test)

    elif dataset_type == "feature":
        x_train, x_test, y_train, y_test = get_feature_dataset()

        test_model_basic(x_train, x_test, y_train, y_test, dataset_type)

        # fine_tuning(x_train, y_train, dataset_type)

        # assess_model(x_train, x_test, y_train, y_test, dataset_type)