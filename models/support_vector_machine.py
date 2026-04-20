from sklearn.svm import SVC
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


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

    # svm_ovo = SVC(decision_function_shape="ovo") #one vs one
    # svm_ovo.fit(x_train, y_train)
    #
    # y_pred_ovo = svm_ovo.predict(x_test)
    #
    # ovo_acc = accuracy_score(y_test, y_pred_ovo)
    # print(f"OVO Accuracy: {ovo_acc}")

    svm_ova = SVC(decision_function_shape="ovr") #one vs all
    svm_ova.fit(x_train, y_train)

    y_pred_ova = svm_ova.predict(x_test)

    ova_acc = accuracy_score(y_test, y_pred_ova)
    print(f"OVA Accuracy: {ova_acc}")

"""
    NOTES
    - turn x/y into numpy arrays
    - fine tune ->  
    - use gridsearch for fine tuning
    - save each result in a csv file
"""


if __name__ == "__main__":
    print("Hello world!")
    test()
    # https://www.geeksforgeeks.org/machine-learning/multi-class-classification-using-support-vector-machines-svm/