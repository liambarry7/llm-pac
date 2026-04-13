from xgboost import XGBClassifier
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

def test():
    dir = r"D:\kimia\Documents\University\UEA\Yr3 Project\Dataset\P001-S.csv"
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

if __name__ == "__main__":
    print("Hello World!")
    test()
    # https://www.geeksforgeeks.org/machine-learning/xgbclassifier/

    # NOTE -> XGBoost requires continuous labels (i.e. 0,1,2,3 not 0,1,3)