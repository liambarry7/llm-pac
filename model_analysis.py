import json
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from utils import remap_labels


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
    label_remap = remap_labels(y_pred, "encode_to_label")

    label_counts = Counter(label_remap) # Counter({'sleep': 7111, 'sedentary': 6263, 'light': 5103, 'moderate-vigorous': 1523})

    print(label_counts)

    labels = list(label_counts.keys())
    values = list(label_counts.values())

    plt.figure(figsize=(7,5))
    bars = plt.bar(labels, values)
    plt.bar_label(bars)
    plt.title(f"{model_type} Class Distribution")
    plt.xlabel("Activity")
    plt.ylabel("Counts")
    plt.savefig(f"graphs\\{model_type}_pcd")
    plt.show()

def model_results_analysis():
    # create scatter graph of model results
    pass

def fine_tune_results(file):
    # create graph of fine tune results/comparison
    pass


if __name__ == "__main__":
    model_results_analysis()
    fine_tune_results()