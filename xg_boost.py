from xgboost import XGBClassifier
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

from llm import save_results

dir = r"D:\kimia\Documents\University\UEA\Yr3 Project\Dataset\data\MASTER-DATA.csv"
df = pd.read_csv(dir)
x = df.drop(columns=['time', 'annotation', 'label']).to_numpy()
y = df['label'].to_numpy()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

def fine_tuning():
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

    results_df.to_csv('results\\xgb_fine_tune_results.csv', index=False)

def assess_model():
    """
    1. get best model params
    2. feed them into model
    3. train model
    4. test model
    5. record performance
    """

    # get best params
    dir = r"results\xgb_fine_tune_results.csv"
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
    model_type = "XGBoost"
    model_data = [acc, prec, recall, f1]

    save_results(model_type, model_data)

def test():
    # https://www.geeksforgeeks.org/machine-learning/xgbclassifier/

    # dir = r"D:\kimia\Documents\University\UEA\Yr3 Project\Dataset\P001-S.csv"
    dir = r"D:\kimia\Documents\University\UEA\Yr3 Project\Dataset\data\MASTER-DATA.csv"

    df = pd.read_csv(dir)
    # df = df.drop(columns=['time', 'annotation'])
    print(df.columns)
    print(df.head(15))

    x = df.drop(columns=['time', 'annotation', 'label'])
    y = df['label']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    xgb = XGBClassifier()
    xgb.fit(x_train, y_train)

    y_pred = xgb.predict(x_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc}")

def test_features():
    dir1 = r"D:\kimia\Documents\University\UEA\Yr3 Project\Dataset\data\TRAIN-DATA-SS.csv"
    df_train = pd.read_csv(dir1)

    dir2 = r"D:\kimia\Documents\University\UEA\Yr3 Project\Dataset\data\TEST-DATA-SS.csv"
    df_test = pd.read_csv(dir2)

    x_train = df_train.drop(columns=['label'])
    y_train = df_train['label']

    x_test = df_test.drop(columns=['label'])
    y_test = df_test['label']

    xgb = XGBClassifier()
    xgb.fit(x_train, y_train)

    y_pred = xgb.predict(x_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc}")

if __name__ == "__main__":
    print("Hello World!")
    # test()
    # https://www.geeksforgeeks.org/machine-learning/xgbclassifier/

    # NOTE -> XGBoost requires continuous labels (i.e. 0,1,2,3 not 0,1,3)
    # fine_tuning()
    assess_model()