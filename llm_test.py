# pip install -U "huggingface_hub"
import json

# hf auth login
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
import pandas as pd
import re

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay, \
    confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig # pip install torch transformers accelerate bitsandbytes

from model_analysis import save_results, generate_cm, metric_comparison, class_comparison, predict_class_distribution
from feature_preprocessing import get_feature_dataset, get_llm_dataset
from utils import remap_labels
from standard_preprocessing import get_standard_dataset

model_id = "google/gemma-2b-it"

config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=config, device_map="cuda")

# tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
# # model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it", device_map="auto", load_in_8bit=True) # pip install accelerate
# model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it").to("cuda") # pip install accelerate

# activities = ["sleep", "sitting", "walking", "bicycling", "mixed-activity", "standing", "manual-work", "sports"]

run_lim = 10


def test_llm():
    input_text = "Write me a poem about Machine Learning."
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

    outputs = model.generate(**input_ids,
                             max_new_tokens=100,
                             do_sample=False,
                             eos_token_id=tokenizer.eos_token_id,
                             pad_token_id=tokenizer.eos_token_id)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(response)

def zero_shot_prompting(x_test, y_test):

    y_unencoded = remap_labels(y_test, "encode_to_label")
    pred_labels = []

    for i in range(len(x_test)):

        zero_shot_template = f"""
                    You are a physical activity classifier. Predict the activity label from wrist accelerometer features.
                    
                    Feature interpretation guide:
                    - High std/rms/jerk = movement present. Low values = stillness.
                    - sleep: near-zero std, jerk, and rms on all axes. Low dominant frequency.
                    - sedentary: very low movement, some postural shifts. Low jerk.
                    - light: moderate std and rms. Some rhythmic pattern in dominant frequency.
                    - moderate-vigorous: high std, rms, jerk across axes. Elevated dominant frequency.

                    Possible labels:
                    sleep, sedentary, moderate-vigorous, light
                    
                    Features:
                    x-mean = {x_test.iloc[i]['x_mean']:.2f}, y-mean = {x_test.iloc[i]['y_mean']:.2f}, z-mean = {x_test.iloc[i]['z_mean']:.2f}
                    x-std = {x_test.iloc[i]['x_std']:.2f}, y-std = {x_test.iloc[i]['y_std']:.2f}, z-std = {x_test.iloc[i]['z_std']:.2f}
                    x-min = {x_test.iloc[i]['x_min']:.2f}, y-min = {x_test.iloc[i]['y_min']:.2f}, z-min = {x_test.iloc[i]['z_min']:.2f}
                    x-max = {x_test.iloc[i]['x_max']:.2f}, y-max = {x_test.iloc[i]['y_max']:.2f}, z-max = {x_test.iloc[i]['z_max']:.2f}
                    magnitude-mean = {x_test.iloc[i]['magnitude_mean']:.2f}, magnitude-std = {x_test.iloc[i]['magnitude_std']:.2f}
                    rms = {x_test.iloc[i]['rms']:.2f}, jerk = {x_test.iloc[i]['jerk']:.2f}, dominant frequency = {x_test.iloc[i]['dominant_freq']:.2f}
                    
                    Return only the single best activity and confidence score (between 0-1).
                    
                    Answer:
                """

        input_ids = tokenizer(zero_shot_template, return_tensors="pt").to("cuda")

        outputs = model.generate(**input_ids,
                                 max_new_tokens=10,
                                 temperature=0.1,
                                 do_sample=False)

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(response)
        r = extract_label(response)
        print(r)
        pred_labels.append(r)
        print(f"Actual label: {y_test[i]}, Activity: {y_unencoded[i]}")

        if i == run_lim:
            break

    pred_labels_encoded = remap_labels(pred_labels, "label_to_encode")

    acc = accuracy_score(y_test[:len(pred_labels_encoded)], pred_labels_encoded)
    prec = precision_score(y_test[:len(pred_labels_encoded)], pred_labels_encoded, average="weighted")
    recall = recall_score(y_test[:len(pred_labels_encoded)], pred_labels_encoded, average="weighted")
    f1 = f1_score(y_test[:len(pred_labels_encoded)], pred_labels_encoded, average="weighted")

    # acc = accuracy_score(y_test, pred_labels_encoded)
    # prec = precision_score(y_test, pred_labels_encoded, average="weighted")
    # recall = recall_score(y_test, pred_labels_encoded, average="weighted")
    # f1 = f1_score(y_test, pred_labels_encoded, average="weighted")

    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1-score: {f1 * 100:.2f}%")

    # record results
    model_type = "ZERO-SHOT"
    model_data = [acc, prec, recall, f1]

    save_results(model_type, "assess", model_data)

    # Create graphs for analysis
    generate_cm(y_test[:len(pred_labels_encoded)], pred_labels_encoded, model_type)
    metric_comparison(model_data, model_type)
    class_report = classification_report(y_test[:len(pred_labels_encoded)], pred_labels_encoded, target_names=["light", "moderate-vigorous", "sedentary", "sleep"], output_dict=True)
    class_comparison(class_report, model_type)
    predict_class_distribution(pred_labels_encoded, model_type)

    # generate_cm(y_test, pred_labels_encoded, model_type)
    # metric_comparison(model_data, model_type)
    # class_report = classification_report(y_test, pred_labels_encoded, target_names=["light", "moderate-vigorous", "sedentary", "sleep"], output_dict=True)
    # class_comparison(class_report, model_type)
    # predict_class_distribution(pred_labels_encoded, model_type)


def few_shot_prompting(x_train, y_train, x_test, y_test):
    y_unencoded = remap_labels(y_test, "encode_to_label")
    pred_labels = []


    for i in range(0, len(x_test)):
        sample_df = sample_training_data(x_train, y_train)
        sample_y = remap_labels(sample_df['encoded_label'], "encode_to_label")

        zero_shot_template = f"""
                    You are a physical activity classifier.

                    Predict the activity label from wrist accelerometer features.
                    
                    Feature interpretation guide:
                    - High std/rms/jerk = movement present. Low values = stillness.
                    - sleep: near-zero std, jerk, and rms on all axes. Low dominant frequency.
                    - sedentary: very low movement, some postural shifts. Low jerk.
                    - light: moderate std and rms. Some rhythmic pattern in dominant frequency.
                    - moderate-vigorous: high std, rms, jerk across axes. Elevated dominant frequency.

                    Possible labels:
                    sleep, sedentary, moderate-vigorous, light

                    Example 1:
                    Features:
                    x-mean = {sample_df.iloc[0]['x_mean']:.2f}, y-mean = {sample_df.iloc[0]['y_mean']:.2f}, z-mean = {sample_df.iloc[0]['z_mean']:.2f}
                    x-std = {sample_df.iloc[0]['x_std']:.2f}, y-std = {sample_df.iloc[0]['y_std']:.2f}, z-std = {sample_df.iloc[0]['z_std']:.2f}
                    x-min = {sample_df.iloc[0]['x_min']:.2f}, y-min = {sample_df.iloc[0]['y_min']:.2f}, z-min = {sample_df.iloc[0]['z_min']:.2f}
                    x-max = {sample_df.iloc[0]['x_max']:.2f}, y-max = {sample_df.iloc[0]['y_max']:.2f}, z-max = {sample_df.iloc[0]['z_max']:.2f}
                    magnitude-mean = {sample_df.iloc[0]['magnitude_mean']:.2f}, magnitude-std = {sample_df.iloc[0]['magnitude_std']:.2f}
                    rms = {sample_df.iloc[0]['rms']:.2f}, jerk = {sample_df.iloc[0]['jerk']:.2f}, dominant frequency = {sample_df.iloc[0]['dominant_freq']:.2f}
                    Answer = {sample_y[0]}

                    Example 2:
                    Features:
                    x-mean = {sample_df.iloc[1]['x_mean']:.2f}, y-mean = {sample_df.iloc[1]['y_mean']:.2f}, z-mean = {sample_df.iloc[1]['z_mean']:.2f}
                    x-std = {sample_df.iloc[1]['x_std']:.2f}, y-std = {sample_df.iloc[1]['y_std']:.2f}, z-std = {sample_df.iloc[1]['z_std']:.2f}
                    x-min = {sample_df.iloc[1]['x_min']:.2f}, y-min = {sample_df.iloc[1]['y_min']:.2f}, z-min = {sample_df.iloc[1]['z_min']:.2f}
                    x-max = {sample_df.iloc[1]['x_max']:.2f}, y-max = {sample_df.iloc[1]['y_max']:.2f}, z-max = {sample_df.iloc[1]['z_max']:.2f}
                    magnitude-mean = {sample_df.iloc[1]['magnitude_mean']:.2f}, magnitude-std = {sample_df.iloc[1]['magnitude_std']:.2f}
                    rms = {sample_df.iloc[1]['rms']:.2f}, jerk = {sample_df.iloc[1]['jerk']:.2f}, dominant frequency = {sample_df.iloc[1]['dominant_freq']:.2f}
                    Answer = {sample_y[1]}

                    Example 3:
                    Features:
                    x-mean = {sample_df.iloc[2]['x_mean']:.2f}, y-mean = {sample_df.iloc[2]['y_mean']:.2f}, z-mean = {sample_df.iloc[2]['z_mean']:.2f}
                    x-std = {sample_df.iloc[2]['x_std']:.2f}, y-std = {sample_df.iloc[2]['y_std']:.2f}, z-std = {sample_df.iloc[2]['z_std']:.2f}
                    x-min = {sample_df.iloc[2]['x_min']:.2f}, y-min = {sample_df.iloc[2]['y_min']:.2f}, z-min = {sample_df.iloc[2]['z_min']:.2f}
                    x-max = {sample_df.iloc[2]['x_max']:.2f}, y-max = {sample_df.iloc[2]['y_max']:.2f}, z-max = {sample_df.iloc[2]['z_max']:.2f}
                    magnitude-mean = {sample_df.iloc[2]['magnitude_mean']:.2f}, magnitude-std = {sample_df.iloc[2]['magnitude_std']:.2f}
                    rms = {sample_df.iloc[2]['rms']:.2f}, jerk = {sample_df.iloc[2]['jerk']:.2f}, dominant frequency = {sample_df.iloc[2]['dominant_freq']:.2f}
                    Answer = {sample_y[2]}
                    
                    Example 4:
                    Features:
                    x-mean = {sample_df.iloc[3]['x_mean']:.2f}, y-mean = {sample_df.iloc[3]['y_mean']:.2f}, z-mean = {sample_df.iloc[3]['z_mean']:.2f}
                    x-std = {sample_df.iloc[3]['x_std']:.2f}, y-std = {sample_df.iloc[3]['y_std']:.2f}, z-std = {sample_df.iloc[3]['z_std']:.2f}
                    x-min = {sample_df.iloc[3]['x_min']:.2f}, y-min = {sample_df.iloc[3]['y_min']:.2f}, z-min = {sample_df.iloc[3]['z_min']:.2f}
                    x-max = {sample_df.iloc[3]['x_max']:.2f}, y-max = {sample_df.iloc[3]['y_max']:.2f}, z-max = {sample_df.iloc[3]['z_max']:.2f}
                    magnitude-mean = {sample_df.iloc[3]['magnitude_mean']:.2f}, magnitude-std = {sample_df.iloc[3]['magnitude_std']:.2f}
                    rms = {sample_df.iloc[3]['rms']:.2f}, jerk = {sample_df.iloc[3]['jerk']:.2f}, dominant frequency = {sample_df.iloc[3]['dominant_freq']:.2f}
                    Answer = {sample_y[3]}

                    Now classify this sample:
                    Features:
                    x-mean = {x_test.iloc[i]['x_mean']:.2f}, y-mean = {x_test.iloc[i]['y_mean']:.2f}, z-mean = {x_test.iloc[i]['z_mean']:.2f}
                    x-std = {x_test.iloc[i]['x_std']:.2f}, y-std = {x_test.iloc[i]['y_std']:.2f}, z-std = {x_test.iloc[i]['z_std']:.2f}
                    x-min = {x_test.iloc[i]['x_min']:.2f}, y-min = {x_test.iloc[i]['y_min']:.2f}, z-min = {x_test.iloc[i]['z_min']:.2f}
                    x-max = {x_test.iloc[i]['x_max']:.2f}, y-max = {x_test.iloc[i]['y_max']:.2f}, z-max = {x_test.iloc[i]['z_max']:.2f}
                    magnitude-mean = {x_test.iloc[i]['magnitude_mean']:.2f}, magnitude-std = {x_test.iloc[i]['magnitude_std']:.2f}
                    rms = {x_test.iloc[i]['rms']:.2f}, jerk = {x_test.iloc[i]['jerk']:.2f}, dominant frequency = {x_test.iloc[i]['dominant_freq']:.2f}
                    Return only the single best activity.

                    Answer:
                """

        input_ids = tokenizer(zero_shot_template, return_tensors="pt").to("cuda")

        outputs = model.generate(**input_ids,
                                 max_new_tokens=10,
                                 temperature=0.1,
                                 do_sample=False)

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(response)
        # print(f"Actual label: {y_test[i]}")
        r = extract_label(response)
        pred_labels.append(r)
        print(r)
        print(f"Actual label: {y_test[i]}, Activity: {y_unencoded[i]}")

        if i == run_lim:
            break

    pred_labels_encoded = remap_labels(pred_labels, "label_to_encode")

    acc = accuracy_score(y_test[:len(pred_labels_encoded)], pred_labels_encoded)
    prec = precision_score(y_test[:len(pred_labels_encoded)], pred_labels_encoded, average="weighted")
    recall = recall_score(y_test[:len(pred_labels_encoded)], pred_labels_encoded, average="weighted")
    f1 = f1_score(y_test[:len(pred_labels_encoded)], pred_labels_encoded, average="weighted")

    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1-score: {f1 * 100:.2f}%")

    # record results
    model_type = "Few-shot Prompting"
    model_data = [acc, prec, recall, f1]

    save_results(model_type, "assess", model_data)

    # create confusion matrix
    generate_cm(y_test[:len(pred_labels_encoded)], pred_labels_encoded, "few-shot")

def extract_label(response):
    if "Answer:" in response:
        response = response.split("Answer:")[-1]

    response = response.strip().lower()
    valid_labels = ["sleep", "sedentary", "light", "moderate-vigorous"]

    for label in valid_labels:
        if label in response:
            return label

    return "N/A"


def sample_training_data(x_train, y_train):
    # take a random sample for each label for few shot prompting

    df = x_train.copy()
    df['encoded_label'] = y_train

    sample_df = df.groupby("encoded_label", group_keys=False).sample(n=1)

    # print(sample_df)
    return sample_df



if __name__ == "__main__":
    # test_llm()
    remap_labels([0,1,2,3], "encode_to_label")

    x_train, x_test, y_train, y_test = get_llm_dataset()

    # sample_training_data(x_train, y_train)

    zero_shot_prompting(x_test, y_test)
    # few_shot_prompting(x_train, y_train, x_test, y_test)

