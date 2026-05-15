# pip install -U "huggingface_hub"
import json

# hf auth login
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
import pandas as pd
import re

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay, \
    confusion_matrix
from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig # pip install torch transformers accelerate bitsandbytes

# from llm import remap_labels
from random_forest import save_results
from feature_preprocessing import get_feature_dataset, get_llm_dataset
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

activities = ["sleep", "sitting", "walking", "bicycling", "mixed-activity", "standing", "manual-work", "sports"]

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

def zero_shot_prompting(x_test, y_test, dataset_type):

    y_unencoded = remap_labels(y_test, "encode_to_label")

    for i in range(len(x_test)):

        zero_shot_template = f"""
                    You are a physical activity classifier. Predict the activity label from wrist accelerometer features.
                    
                    Possible labels:
                    sleep, sedentary, moderate-vigorous, light
                    
                    Features are Z-score normalized where 0 is the population mean.
                    
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
                                 max_new_tokens=20)

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(response)
        print(f"Actual label: {y_test[i]}")
        print(f"Actual label: {y_test[i]}, Activity: {y_unencoded[i]}")

        if i == 3:
            break

    # acc = accuracy_score(testLabel_values.head(11), pred_labels_encoded)


def few_shot_prompting(x_test, y_test, dataset_type):
    # print(x_test.head())
    # print(x_test.columns)
    # 'x_mean', 'y_mean', 'z_mean', 'x_std', 'y_std', 'z_std', 'x_min',
    #        'y_min', 'z_min', 'x_max', 'y_max', 'z_max', 'magnitude_mean'
    # print(y_test)
    y_unencoded = remap_labels(y_test, "encode_to_label")


    for i in range(0, len(x_test), 5):

        zero_shot_template = f"""
                    You are a physical activity classifier.

                    Predict the activity label from wrist accelerometer features.

                    Possible labels:
                    sleep, sedentary, moderate-vigorous, light


                    Features are Z-score normalized where 0 is the population mean.
                    
                    Example 1:
                    Features:
                    x-mean = {x_test.iloc[i]['x_mean']:.2f}, y-mean = {x_test.iloc[i]['y_mean']:.2f}, z-mean = {x_test.iloc[i]['z_mean']:.2f}
                    x-std = {x_test.iloc[i]['x_std']:.2f}, y-std = {x_test.iloc[i]['y_std']:.2f}, z-std = {x_test.iloc[i]['z_std']:.2f}
                    x-min = {x_test.iloc[i]['x_min']:.2f}, y-min = {x_test.iloc[i]['y_min']:.2f}, z-min = {x_test.iloc[i]['z_min']:.2f}
                    x-max = {x_test.iloc[i]['x_max']:.2f}, y-max = {x_test.iloc[i]['y_max']:.2f}, z-max = {x_test.iloc[i]['z_max']:.2f}
                    magnitude-mean = {x_test.iloc[i]['magnitude_mean']:.2f}, magnitude-std = {x_test.iloc[i]['magnitude_std']:.2f}
                    rms = {x_test.iloc[i]['rms']:.2f}, jerk = {x_test.iloc[i]['jerk']:.2f}, dominant frequency = {x_test.iloc[i]['dominant_freq']:.2f}
                    Label = {y_unencoded[i]}
                    
                    Example 2:
                    Features:
                    x-mean = {x_test.iloc[i+1]['x_mean']:.2f}, y-mean = {x_test.iloc[i+1]['y_mean']:.2f}, z-mean = {x_test.iloc[i+1]['z_mean']:.2f}
                    x-std = {x_test.iloc[i+1]['x_std']:.2f}, y-std = {x_test.iloc[i+1]['y_std']:.2f}, z-std = {x_test.iloc[i+1]['z_std']:.2f}
                    x-min = {x_test.iloc[i+1]['x_min']:.2f}, y-min = {x_test.iloc[i+1]['y_min']:.2f}, z-min = {x_test.iloc[i+1]['z_min']:.2f}
                    x-max = {x_test.iloc[i+1]['x_max']:.2f}, y-max = {x_test.iloc[i+1]['y_max']:.2f}, z-max = {x_test.iloc[i+1]['z_max']:.2f}
                    magnitude-mean = {x_test.iloc[i+1]['magnitude_mean']:.2f}, magnitude-std = {x_test.iloc[i+1]['magnitude_std']:.2f}
                    rms = {x_test.iloc[i+1]['rms']:.2f}, jerk = {x_test.iloc[i+1]['jerk']:.2f}, dominant frequency = {x_test.iloc[i+1]['dominant_freq']:.2f}
                    Label = {y_unencoded[i+1]}
                    
                    Example 3:
                    Features:
                    x-mean = {x_test.iloc[i+2]['x_mean']:.2f}, y-mean = {x_test.iloc[i+2]['y_mean']:.2f}, z-mean = {x_test.iloc[i+2]['z_mean']:.2f}
                    x-std = {x_test.iloc[i+2]['x_std']:.2f}, y-std = {x_test.iloc[i+2]['y_std']:.2f}, z-std = {x_test.iloc[i+2]['z_std']:.2f}
                    x-min = {x_test.iloc[i+2]['x_min']:.2f}, y-min = {x_test.iloc[i+2]['y_min']:.2f}, z-min = {x_test.iloc[i+2]['z_min']:.2f}
                    x-max = {x_test.iloc[i+2]['x_max']:.2f}, y-max = {x_test.iloc[i+2]['y_max']:.2f}, z-max = {x_test.iloc[i+2]['z_max']:.2f}
                    magnitude-mean = {x_test.iloc[i+2]['magnitude_mean']:.2f}, magnitude-std = {x_test.iloc[i+2]['magnitude_std']:.2f}
                    rms = {x_test.iloc[i+2]['rms']:.2f}, jerk = {x_test.iloc[i+2]['jerk']:.2f}, dominant frequency = {x_test.iloc[i+2]['dominant_freq']:.2f}
                    Label = {y_unencoded[i+2]}
                    
                    Now classify this sample:
                    Features:
                    x-mean = {x_test.iloc[i+3]['x_mean']:.2f}, y-mean = {x_test.iloc[i+3]['y_mean']:.2f}, z-mean = {x_test.iloc[i+3]['z_mean']:.2f}
                    x-std = {x_test.iloc[i+3]['x_std']:.2f}, y-std = {x_test.iloc[i+3]['y_std']:.2f}, z-std = {x_test.iloc[i+3]['z_std']:.2f}
                    x-min = {x_test.iloc[i+3]['x_min']:.2f}, y-min = {x_test.iloc[i+3]['y_min']:.2f}, z-min = {x_test.iloc[i+3]['z_min']:.2f}
                    x-max = {x_test.iloc[i+3]['x_max']:.2f}, y-max = {x_test.iloc[i+3]['y_max']:.2f}, z-max = {x_test.iloc[i+3]['z_max']:.2f}
                    magnitude-mean = {x_test.iloc[i+3]['magnitude_mean']:.2f}, magnitude-std = {x_test.iloc[i+3]['magnitude_std']:.2f}
                    rms = {x_test.iloc[i+3]['rms']:.2f}, jerk = {x_test.iloc[i+3]['jerk']:.2f}, dominant frequency = {x_test.iloc[i+3]['dominant_freq']:.2f}

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
        print(f"Actual label: {y_test[i]}")
        print(f"Actual label: {y_test[i]}, Activity: {y_unencoded[i]}")


        if i == (3*5):
            break


def remap_labels(label_list, direction):
    # --- New label mapping ---
    annotation_label_dir = r"data\annotation-label-encoded.csv"
    label_df = pd.read_csv(annotation_label_dir)

    # get all unique pairs
    labels = label_df[['label:Walmsley2020', 'encoded_label']].drop_duplicates().reset_index(drop=True)
    print(labels)

    if direction == "label_to_encode":
        mapping = dict(zip(
            labels["label:Walmsley2020"],
            labels["encoded_label"]
        ))

        return [mapping.get(label, None) for label in label_list]

    elif direction == "encode_to_label":
        mapping = dict(zip(
            labels["encoded_label"],
            labels["label:Walmsley2020"]
        ))

        return [mapping.get(code, None) for code in label_list]

if __name__ == "__main__":
    # test_llm()

    # dataset_type = "standard"
    dataset_type = "feature"

    if dataset_type == "standard":
        x_train, x_test, y_train, y_test = get_standard_dataset()



    elif dataset_type == "feature":
        # x_train, x_test, y_train, y_test = get_feature_dataset()
        x_train, x_test, y_train, y_test = get_llm_dataset()

        zero_shot_prompting(x_test, y_test, dataset_type)
        few_shot_prompting(x_test, y_test, dataset_type)



# zero_template = f"""
#                     Context:
#                     You are an expert in Physical Activity Classification.
#                     The data was recorded from Axivity AX3 wrist-worn tri-axial accelerometer on their dominant hand.
#                     The accelerometer was set to capture tri-axial acceleration data at 100 Hz with a dynamic range of ±8g.
#                     Wearable cameras were used to collect ground truths of the participants’ activities while wearing the accelerometers.
#                     Participants were given an OMG Life Autographer, a wearable camera worn around the neck which automatically takes photographs every 20 - 40 seconds
#
#                     Labels: {', '.join(activities)}
#                     Data:
#                         x-mean = {x_test.iloc[i]['x_mean']}, y-mean = {x_test.iloc[i]['y_mean']}, z-mean = {x_test.iloc[i]['z_mean']}
#                         x-std = {x_test.iloc[i]['x_std']}, y-std = {x_test.iloc[i]['y_std']}, z-std = {x_test.iloc[i]['z_std']}
#                         x-min = {x_test.iloc[i]['x_min']}, y-min = {x_test.iloc[i]['y_min']}, z-min = {x_test.iloc[i]['z_min']}
#                         x-max = {x_test.iloc[i]['x_max']}, y-max = {x_test.iloc[i]['y_max']}, z-max = {x_test.iloc[i]['z_max']}
#                         magnitude-mean = {x_test.iloc[i]['magnitude_mean']}
#                     Question: Which activity from the options above best matches the data?
#                     Answer:
#                 """
#
# few_template = 0