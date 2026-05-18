import pickle
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from model_analysis import save_results, generate_cm, metric_comparison, class_comparison, predict_class_distribution
from feature_preprocessing import get_feature_dataset


def test_model_basic(x_train, x_test, y_train, y_test):
    """
        Test the basic setup of a XGBoost Classifier to ensure library is imported correctly.

        Args:
            x_train (pandas.DataFrame): Dataframe containing training data.
            x_test (pandas.Dataframe): Dataframe containing testing data.
            y_train (pandas.Series): Series containing training labels.
            y_test (pandas.Series): Series containing testing labels.

        Returns:
            None
            """
    print("\nTesting basic XGBoost model...")
    xgb = XGBClassifier()
    xgb.fit(x_train, y_train)

    y_pred = xgb.predict(x_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1-score: {f1 * 100:.2f}%")

    # record results
    model_type = "XGB - TEST"
    model_data = [acc, prec, recall, f1]
    save_results(model_type, "base-test", model_data)

    # generate confusion matrix
    generate_cm(y_test, y_pred, "TEST-XGB")


def fine_tuning(x_train, y_train):
    """
        Fine-tune a XGBoost Classifier using a GridSearchCV object, saving the results of each
        configuration to a CSV file.

        Args:
            x_train (pandas.DataFrame): Dataframe containing training data.
            y_train (pandas.Series): Series containing training labels.

        Returns:
            None
        """
    print("\nFine-tuning XGBoost model...")

    # create rf model with no params
    xgb = XGBClassifier()

    param_grid = [{
        'max_depth': [3,6,10],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 500, 1000],
        'colsample_bytree': [0.3, 0.7]
    }]

    # five-fold - use StratifiedKFold to avoid imbalanced class distribution
    ff = StratifiedKFold(n_splits=5, shuffle=True, random_state=41)

    grid_search = GridSearchCV(xgb, param_grid, cv=ff, scoring='accuracy', refit=True, n_jobs=-1, verbose=3)
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
        ['param_max_depth', 'param_learning_rate', 'param_n_estimators', 'param_colsample_bytree', 'mean_test_score',
         'std_test_score', 'rank_test_score']].sort_values(by='rank_test_score')
    print(results_df.head())

    results_df.to_csv(f'results\\xgb_fine_tune_results.csv', index=False)

def assess_model(x_train, x_test, y_train, y_test):
    """
        Assess a XGBoost Classifier using the best performaing hyperparameter configuration
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
    print("\nAssessing XGBoost model...")


    # get best params
    dir = f"results\\xgb_fine_tune_results.csv"
    df = pd.read_csv(dir)
    optimal_params = df[['param_max_depth', 'param_learning_rate', 'param_n_estimators', 'param_colsample_bytree']].iloc[0]
    params = optimal_params.to_list()


    xgb = XGBClassifier(max_depth=int(params[0]), learning_rate=params[1], n_estimators=int(params[2]), colsample_bytree=params[3])
    xgb.fit(x_train, y_train)

    y_pred = xgb.predict(x_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1-score: {f1 * 100:.2f}%")

    # record results
    model_type = "XGB"
    model_data = [acc, prec, recall, f1]

    save_results(model_type, "assess", model_data)

    # Create graphs for analysis
    generate_cm(y_test, y_pred, model_type)
    metric_comparison(model_data, model_type)
    class_report = classification_report(y_test, y_pred, target_names=["light", "moderate-vigorous", "sedentary", "sleep"], output_dict=True)
    class_comparison(class_report, model_type)
    predict_class_distribution(y_pred, model_type)

    # save model
    with open(f'models\\xgb.pkl', 'wb') as file:
        pickle.dump(xgb, file)


if __name__ == "__main__":
    x_train, x_test, y_train, y_test = get_feature_dataset()

    # test_model_basic(x_train, x_test, y_train, y_test)

    # fine_tuning(x_train, y_train)

    assess_model(x_train, x_test, y_train, y_test)