import pickle
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from feature_preprocessing import get_feature_dataset, get_llm_dataset
from model_analysis import save_results, generate_cm, metric_comparison, class_comparison, predict_class_distribution


def test_model_basic(x_train, x_test, y_train, y_test):
    """
        Test the basic setup of a Random Forest Classifier to ensure library is imported correctly.

        Args:
            x_train (pandas.DataFrame): Dataframe containing training data.
            x_test (pandas.Dataframe): Dataframe containing testing data.
            y_train (pandas.Series): Series containing training labels.
            y_test (pandas.Series): Series containing testing labels.

        Returns:
            None
        """
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


def fine_tuning(x_train, y_train, dataset_type="standard"):
    """
        Fine-tune a Random Forest Classifier using a GridSearchCV object on the standardised feature dataset,
        saving the results of each configuration to a CSV file.

        Args:
            x_train (pandas.DataFrame): Dataframe containing training data.
            y_train (pandas.Series): Series containing training labels.
            dataset_type (String): String containing dataset type for saving file

        Returns:
            None
        """
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

    if dataset_type == "llm":
        results_df.to_csv(f'results\\rf_llm_fine_tune_results.csv', index=False)
    elif dataset_type == "standard":
        results_df.to_csv(f'results\\rf_fine_tune_results.csv', index=False)


def assess_model(x_train, x_test, y_train, y_test):
    """
        Assess a Random Forest Classifier using the best performaing hyperparameter configuration
        discovered during fine-tuning. Record performance metrics on test set, and create
        supporting graphs for analysis.

        Args:
            x_train (pandas.DataFrame): Dataframe containing training data.
            x_test (pandas.Dataframe): Dataframe containing testing data.
            y_train (pandas.Series): Series containing training labels.
            y_test (pandas.Series): Series containing testing labels.

        Returns:
            None
        """
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

    # Create graphs for analysis
    generate_cm(y_test, y_pred, model_type)
    metric_comparison(model_data, model_type)
    class_report = classification_report(y_test, y_pred, target_names=["light", "moderate-vigorous", "sedentary", "sleep"], output_dict=True)
    class_comparison(class_report, model_type)
    predict_class_distribution(y_pred, model_type)


    # save model
    with open(f'models\\rf.pkl', 'wb') as file:
        pickle.dump(rf, file)


def rf_llm_train(x_train, y_train):
    """
       Fine-tune a Random Forest Classifier using a GridSearchCV  on the LLM feature dataset, saving the results of each
        configuration to a CSV file.

        Args:
            x_train (pandas.DataFrame): Dataframe containing training data.
            y_train (pandas.Series): Series containing training labels.
            dataset_type (String): String containing dataset type for saving file

        Returns:
            None
       """
    # get best params
    dir = f"results\\rf_llm_fine_tune_results.csv"
    df = pd.read_csv(dir)
    optimal_params = \
    df[['param_n_estimators', 'param_max_depth', 'param_min_samples_leaf', 'param_min_samples_split']].iloc[0]
    params = optimal_params.to_list()
    if pd.isna(params[1]):
        params[1] = None
    print(params)

    rf = RandomForestClassifier(n_estimators=int(params[0]), max_depth=params[1], min_samples_leaf=int(params[2]),
                                min_samples_split=int(params[3]), random_state=42)

    rf.fit(x_train, y_train)

    # save model
    with open('models\\rf_llm.pkl', 'wb') as file:
        pickle.dump(rf, file)

if __name__ == "__main__":

    # x_train, x_test, y_train, y_test = get_feature_dataset()

    # test_model_basic(x_train, x_test, y_train, y_test)

    # fine_tuning(x_train, y_train)

    # assess_model(x_train, x_test, y_train, y_test)

    x_train, x_test, y_train, y_test = get_llm_dataset()
    fine_tuning(x_train, y_train, "llm")
    rf_llm_train(x_train, y_train)