import json
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from utils import remap_labels


def save_results(model_type, results_type, metrics):
    """
        Save the performance metrics of a given model to model_results.json

        Args:
            model_type (string): Name of the model
            results_type (string): result type ("assess" or "base-test")
            metrics (list): list of metrics (accuracy, precision, recall, f1_score)

        Returns:
            None
        """
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
    """
        Create and save a confusion matrix for a given model using predicted y labels and actual y labels.

        Args:
            y_test (list): list of predicted y labels
            y_pred (list): list of actual y labels
            model_type (string): name of the model

        Returns:
            None
        """
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
    """
        Create and save a bar graph of performance metrics for a given model.

        Args:
            model_data (dict): model metric results
            model_type (string): name of the model

        Returns:
            None
        """
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
    """
        Create and save a bar graph comparison the performance metrics for each class for a given model

        Args:
            cr (dictionary): sklearn classification report as a dict
            model_type (string): name of the model

        Returns:
            None
        """
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
    """
        Create and save a bar graph of the distribution of predict labels from a given model

        Args:
            y_pred (list): list of actual y labels
            model_type (string): name of the model

        Returns:
            None
        """
    label_remap = remap_labels(y_pred, "encode_to_label")

    label_counts = Counter(label_remap)

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

def model_accuracy_analysis():
    """
        Create and save a bar graph comparing all accuracies of assessed models.

        Args:
            None

        Returns:
            None
        """
    path = r"results/model_results.json"
    with open(path, "r") as file:
        model_rs = json.load(file)

    models = []
    accuracies = []

    for model_name in model_rs['model_results']:
        accuracy = model_rs['model_results'][model_name]['accuracy']
        models.append(model_name)
        accuracies.append(accuracy)

    # compare accuracy
    plt.figure(figsize=(10,5))
    plt.bar(models, accuracies)
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)

    for i, acc in enumerate(accuracies):
        plt.text(i, acc, f"{acc:.3f}", ha="center")
    plt.title("Model Accuracy Comparison")
    plt.tight_layout()
    plt.savefig(f"graphs\\models_accuracy_comparison")
    plt.show()


if __name__ == "__main__":
    model_accuracy_analysis()