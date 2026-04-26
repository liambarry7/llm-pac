from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

dir = r"D:\kimia\Documents\University\UEA\Yr3 Project\Dataset\data\MASTER-DATA.csv"
df = pd.read_csv(dir)
x = df.drop(columns=['time', 'annotation', 'label']).to_numpy()
y = df['label'].to_numpy()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


def fine_tuning():
    # create new mlp
    knn = KNeighborsClassifier(max_iter=1000, random_state=41)

    param_grid = [{

    }]

    # five-fold - use StratifiedKFold to avoid imbalanced class distribution
    ff = StratifiedKFold(n_splits=5, shuffle=True, random_state=41)

    grid_search = GridSearchCV(knn, param_grid, cv=ff, scoring='accuracy', refit=True, n_jobs=-1, verbose=3)
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

    # change these *****************
    results_df = results_df[
        ['param_activation', 'param_hidden_layer_sizes', 'param_learning_rate', 'param_solver', 'mean_test_score',
         'std_test_score', 'rank_test_score']].sort_values(by='rank_test_score')

    print(results_df.head())

    results_df.to_csv('knn_fine_tune_results.csv', index=False)


def test():
    # dir = r"D:\kimia\Documents\University\UEA\Yr3 Project\Dataset\P001-S.csv"
    dir = r"D:\kimia\Documents\University\UEA\Yr3 Project\Dataset\data\MASTER-DATA.csv"
    df = pd.read_csv(dir)
    # df = df.drop(columns=['time', 'annotation'])
    print(df.columns)
    print(df.head(15))

    x = df.drop(columns=['time', 'annotation', 'label']).to_numpy()
    y = df['label'].to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    knn = KNeighborsClassifier()
    knn.fit(x_train, y_train)

    y_pred = knn.predict(x_test)
    knn_acc = accuracy_score(y_test, y_pred)
    print(f"accuracy knn: {knn_acc}")


if __name__ == "__main__":
    print("Hello World!")
    test()