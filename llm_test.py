# pip install -U "huggingface_hub"
import json
import pickle

import numpy as np
# hf auth login
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
import pandas as pd
import re

import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig # pip install torch transformers accelerate bitsandbytes

from model_analysis import save_results, generate_cm, metric_comparison, class_comparison, predict_class_distribution
from feature_preprocessing import get_feature_dataset, get_llm_dataset
from utils import remap_labels


model_id = "google/gemma-2b-it"

config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=config, device_map="cuda")

run_lim = 10000000
# run_lim = 10


def test_llm():
    input_text = "Write me a poem about Physical Activity Classification."
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
        r = extract_label(response)
        print(r)
        pred_labels.append(r)
        print(f"Actual label: {y_test[i]}, Activity: {y_unencoded[i]}")

        # if i == run_lim:
        #     break

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

        # if i == run_lim:
        #     break

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
    model_type = "FEW-SHOT"
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


def rf_cot(x_test, y_test):
    """
        Use test data to predict labels and confidences scores - if margin between first and second probable 
        labels is too close, use Chain-of-Thought prompting to assess whether the most likely label 
        is correctly predicted or not
        
        Args:
            x_train (pandas.DataFrame): Dataframe containing training data.
            y_train (pandas.Series): Series containing training labels.
            dataset_type (String): String containing dataset type for saving file

        Returns:
            None
       """
    with open('models\\rf_llm.pkl', 'rb') as file:
        rf = pickle.load(file)

    # predict
    y_pred = rf.predict_proba(x_test) # returns array per row of predicted confidence levels

    final_preds = [] # final predictions using full pipeline
    rf_margin_preds = [] # final predictions of rf which flagged with small margin
    cot_margin_preds = [] # final predictions of LLM which where analysed due to small margin
    margin_labels = [] # correct labels for all predictions that had small margin
    total_rf_preds = [] # all rf predictions
    count = 0

    for i in range(len(x_test)):
        sample_pred = y_pred[i] # change name of variable
        pred_id = np.argmax(sample_pred)
        confidence = sample_pred[pred_id]
        pred_label = rf.classes_[pred_id] # map to label

        total_rf_preds.append(pred_label) # record all rf predictions

        # print(f"\nsample_pred: {sample_pred}")
        # print(f"pred_id: {pred_id}")
        # print(f"Confidence: {confidence}")
        # print(f"Pred label: {pred_label}")

        # get margin between top 2 predictions
        sorted_confidences = np.sort(sample_pred)
        first = sorted_confidences[-1]
        second = sorted_confidences[-2]

        margin = first - second
        if margin > 0.15:
            final_preds.append(pred_label)

        else:
            # low confidence, consult LLM with CoT
            count +=1
            features = x_test.iloc[i]
            final_label = cot(features, pred_label, confidence)
            print(f"actual label: {y_test[i]}\nrf label: {pred_label}\nllm label: {final_label}")

            final_preds.append(final_label)
            cot_margin_preds.append(final_label)
            rf_margin_preds.append(pred_label)
            margin_labels.append(y_test[i])

            if count > run_lim:
                break

    confidences = np.max(y_pred, axis=1)
    print(f"Mean confidence: {np.mean(confidences)}")
    print(f"Min confidence: {np.min(confidences)}")
    print(f"Max confidence: {np.max(confidences)}")
    print(f"No of low confidence: {count}/{len(x_test)}") # 892/5000 - 5hrs


    # Full model results
    model_acc = accuracy_score(y_test[:len(final_preds)], final_preds)
    model_prec = precision_score(y_test[:len(final_preds)], final_preds, average="weighted")
    model_recall = recall_score(y_test[:len(final_preds)], final_preds, average="weighted")
    model_f1 = f1_score(y_test[:len(final_preds)], final_preds, average="weighted")

    model_type = "RF-CoT"
    model_data = [model_acc, model_prec, model_recall, model_f1]
    save_results(model_type, "assess", model_data)

    generate_cm(y_test[:len(final_preds)], final_preds, model_type)
    metric_comparison(model_data, model_type)
    class_report = classification_report(y_test[:len(final_preds)], final_preds,
                                         target_names=["light", "moderate-vigorous", "sedentary", "sleep"],
                                         output_dict=True)
    class_comparison(class_report, model_type)
    predict_class_distribution(final_preds, model_type)

    print(F"\nTotal acc: {model_acc}, model prec: {model_prec}, model recall: {model_recall}, model f1: {model_f1}")


    # just CoT results
    cot_acc = accuracy_score(margin_labels[:len(cot_margin_preds)], cot_margin_preds)
    cot_prec = precision_score(y_test[:len(cot_margin_preds)], cot_margin_preds, average="weighted")
    cot_recall = recall_score(y_test[:len(cot_margin_preds)], cot_margin_preds, average="weighted")
    cot_f1 = f1_score(y_test[:len(cot_margin_preds)], cot_margin_preds, average="weighted")

    model_type = "CoT"
    model_data = [cot_acc, cot_prec, cot_recall, cot_f1]
    save_results(model_type, "assess", model_data)

    generate_cm(y_test[:len(margin_labels)], cot_margin_preds, model_type)
    metric_comparison(model_data, model_type)
    class_report = classification_report(margin_labels[:len(cot_margin_preds)], cot_margin_preds,
                                         target_names=["light", "moderate-vigorous", "sedentary", "sleep"],
                                         output_dict=True)
    class_comparison(class_report, model_type)
    predict_class_distribution(cot_margin_preds, model_type)

    print(F"\nCoT acc: {cot_acc}, CoT prec: {cot_prec}, CoT recall: {cot_recall}, CoT f1: {cot_f1}")


    # just RF results
    rf_acc = accuracy_score(margin_labels[:len(rf_margin_preds)], rf_margin_preds)
    rf_prec = precision_score(y_test[:len(rf_margin_preds)], rf_margin_preds, average="weighted")
    rf_recall = recall_score(y_test[:len(rf_margin_preds)], rf_margin_preds, average="weighted")
    rf_f1 = f1_score(y_test[:len(rf_margin_preds)], rf_margin_preds, average="weighted")

    model_type = "RF-LLM"
    model_data = [rf_acc, rf_prec, rf_recall, rf_f1]
    save_results(model_type, "assess", model_data)

    generate_cm(margin_labels[:len(rf_margin_preds)], rf_margin_preds, model_type)
    metric_comparison(model_data, model_type)
    class_report = classification_report(margin_labels[:len(rf_margin_preds)], rf_margin_preds,
                                         target_names=["light", "moderate-vigorous", "sedentary", "sleep"],
                                         output_dict=True)
    class_comparison(class_report, model_type)
    predict_class_distribution(rf_margin_preds, model_type)

    print(F"\nRF acc: {rf_acc}, RF prec: {rf_prec}, RF recall: {rf_recall}, RF f1: {rf_f1}")


    # full rf results
    full_rf_acc = accuracy_score(y_test[:len(total_rf_preds)], total_rf_preds)
    full_rf_prec = precision_score(y_test[:len(total_rf_preds)], total_rf_preds, average="weighted")
    full_rf_recall = recall_score(y_test[:len(total_rf_preds)], total_rf_preds, average="weighted")
    full_rf_f1 = f1_score(y_test[:len(total_rf_preds)], total_rf_preds, average="weighted")

    model_type = "RF-LLM-FULL"
    model_data = [rf_acc, rf_prec, rf_recall, rf_f1]
    save_results(model_type, "assess", model_data)

    generate_cm(y_test[:len(total_rf_preds)], total_rf_preds, model_type)
    metric_comparison(model_data, model_type)
    class_report = classification_report(y_test[:len(total_rf_preds)], total_rf_preds,
                                         target_names=["light", "moderate-vigorous", "sedentary", "sleep"],
                                         output_dict=True)
    class_comparison(class_report, model_type)
    predict_class_distribution(total_rf_preds, model_type)

    print(F"\nFull RF acc: {full_rf_acc}, Full RF prec: {full_rf_prec}, Full RF recall: {full_rf_recall}, Full RF f1: {full_rf_f1}")


def cot(x_test, pred_label, confidence):
    """
        Send a CoT prompt to the LLM to assess ML label prediction

        Args:
            x_test (list): list containing feature data of classified sample
            pred_label (int): mapped predicted label from ML classifier
            confidence (float): confidence score of predicted label

        Returns:
            int: returns predicted encoded label
        """
    # get unencoded predicted label
    pred_label_unencoded = remap_labels([pred_label], "encode_to_label")

    cot_template = f"""
                        You are an expert in physical activity classification.
                        Review the following accelerometer feature set, compare it to interpretation guide and
                        Determine whether a previous ML classifier correctly classified this data.
                        If incorrect, provide a better label.

                        Feature interpretation guide:
                        - High std/rms/jerk = movement present. Low values = stillness.
                        - sleep: near-zero std, jerk, and rms on all axes. Low dominant frequency.
                        - sedentary: very low movement, some postural shifts. Low jerk.
                        - light: moderate std and rms. Some rhythmic pattern in dominant frequency.
                        - moderate-vigorous: high std, rms, jerk across axes. Elevated dominant frequency.

                        Possible labels:
                        light, moderate-vigorous, sedentary, sleep

                        Features:
                        x-mean = {x_test['x_mean']:.2f}, y-mean = {x_test['y_mean']:.2f}, z-mean = {x_test['z_mean']:.2f}
                        x-std = {x_test['x_std']:.2f}, y-std = {x_test['y_std']:.2f}, z-std = {x_test['z_std']:.2f}
                        x-min = {x_test['x_min']:.2f}, y-min = {x_test['y_min']:.2f}, z-min = {x_test['z_min']:.2f}
                        x-max = {x_test['x_max']:.2f}, y-max = {x_test['y_max']:.2f}, z-max = {x_test['z_max']:.2f}
                        magnitude-mean = {x_test['magnitude_mean']:.2f}, magnitude-std = {x_test['magnitude_std']:.2f}
                        rms = {x_test['rms']:.2f}, jerk = {x_test['jerk']:.2f}, dominant frequency = {x_test['dominant_freq']:.2f}

                        The ML Classifier predicted '{pred_label_unencoded[0]}' with {confidence}, but the margin to the next class was too close.

                        This sample was considered unreasonable because the confidence between top predicted classes was too close.
                        You must still choose the best activity label.

                        Think step by step, then respond in EXACTLY this format:

                        Reasoning: <step-by-step logic here>
                        Final Answer: <one of: light, moderate-vigorous, sedentary, sleep>
                    """

    # print(cot_template)
    input_ids = tokenizer(cot_template, return_tensors="pt").to("cuda")

    outputs = model.generate(**input_ids,
                             max_new_tokens=220,
                             temperature=0.1,
                             do_sample=False,
                             eos_token_id=tokenizer.eos_token_id,
                             pad_token_id=tokenizer.eos_token_id
                             )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(response)

    # extract final label
    final_label = extract_label_cot(response)
    print(f"Extracted label: {final_label}")

    prediction = remap_labels([final_label], "label_to_encode")
    return prediction[0]


def extract_label(response):
    if "Answer:" in response:
        response = response.split("Answer:")[-1]

    response = response.strip().lower()
    valid_labels = ["sleep", "sedentary", "light", "moderate-vigorous"]

    for label in valid_labels:
        if label in response:
            return label

    return "N/A"

def extract_label_cot(response):
    if "Answer:" in response:
        response = response.split("Answer:")[-1]

    response = response.strip().lower()
    valid_labels = ["sleep", "sedentary", "light", "moderate-vigorous"]

    last_pos, last_label = -1, None
    for label in valid_labels:
        pos = response.rfind(label)  # rightmost occurrence
        if pos > last_pos:
            last_pos = pos
            last_label = label

    return last_label if last_label else "N/A"


def sample_training_data(x_train, y_train):
    # take a random sample for each label for few shot prompting

    df = x_train.copy()
    df['encoded_label'] = y_train

    sample_df = df.groupby("encoded_label", group_keys=False).sample(n=1)

    # print(sample_df)
    return sample_df



if __name__ == "__main__":
    # test_llm()
    # remap_labels([0,1,2,3], "encode_to_label")

    x_train, x_test, y_train, y_test = get_llm_dataset()

    # sample_training_data(x_train, y_train)



    rf_cot(x_test, y_test)

    # DO NOT RUN AGAIN
    # #zero_shot_prompting(x_test, y_test)
    # #few_shot_prompting(x_train, y_train, x_test, y_test)



