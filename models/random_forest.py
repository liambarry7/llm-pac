from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

dir = r"D:\kimia\Documents\University\UEA\Yr3 Project\Dataset\data\MASTER-DATA.csv"
df = pd.read_csv(dir)
x = df.drop(columns=['time', 'annotation', 'label']).to_numpy()
y = df['label'].to_numpy()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#
def fine_tuning():
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

    print(f"Best DT params: {grid_search.best_params_}")
    # print(dt_gridsearch.scoring)
    print(f"Best DT Object: {grid_search.best_estimator_}")
    print(f"Best DT Accuracy Score: {grid_search.best_score_}")
    # print(dt_gridsearch.cv_results_)

    cv_res = grid_search.cv_results_
    print(cv_res.keys())

    results_df = pd.DataFrame(grid_search.cv_results_)
    # params = params used, mean_test_score = avg score over 5 folds, std_test_score =
    # results_df = results_df[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']].sort_values(by='rank_test_score')
    results_df = results_df[
        ['param_n_estimators', 'param_max_depth', 'param_min_samples_leaf', 'param_min_samples_split', 'mean_test_score',
         'std_test_score', 'rank_test_score']].sort_values(by='rank_test_score')
    print(results_df.head())

    results_df.to_csv('rf_fine_tune_results.csv', index=False)


def test():
    # dir = r"D:\kimia\Documents\University\UEA\Yr3 Project\Dataset\P001-S.csv"
    dir = r"D:\kimia\Documents\University\UEA\Yr3 Project\Dataset\data\MASTER-DATA.csv"
    df = pd.read_csv(dir)
    # df = df.drop(columns=['time', 'annotation'])
    print(df.columns)
    print(df.head(15))

    x = df.drop(columns=['time', 'annotation', 'label'])
    y = df['label']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    import time
    start = time.time()
    rf = RandomForestClassifier(n_estimators=100,
    random_state=42)

    rf.fit(x_train, y_train)
    print(time.time() - start)

    y_pred = rf.predict(x_test)


    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc}")

    print(classification_report(y_test, y_pred))

"""
    NOTES
    - turn x/y into numpy arrays
    - fine tune ->  n_estimators, max_depth, min_samples_leaf, min_samples_split, criterion
    - use gridsearch for fine tuning
    - save each result in a csv file
"""

if __name__ == "__main__":
    # test()
    fine_tuning()