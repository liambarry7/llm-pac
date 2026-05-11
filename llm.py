# pip install -U "huggingface_hub"
import json

# hf auth login
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
import pandas as pd
import re

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay, \
    confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it", device_map="auto") # pip install accelerate

# activities = ["sitting", "standing", "mixed-activity", "walking", "manual-work", "sports", "bicycling", "sleep"]
# input_text = "Write me a poem about Machine Learning."
# input_text = (f"You are classifying different physical activities. These activities are {activities}. "
#               "Given data from an accelerometer, what class of physical activity are you doing:"
#               "x=0.07865859, y=-1.0151173, z=-0.056527406.")
# input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
#
# input_text = (
#     f"Options: {' '.join(activities)}\n"
#     f"Data: x={0.07865859}, y={-1.0151173}, z={-0.056527406}\n"
#     "Question: Which activity from the options above best matches this raw accelerometer data?\n"
#     "Answer: "
# )
# input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
#
# outputs = model.generate(**input_ids, max_new_tokens=50, eos_token_id=tokenizer.encode("\n", add_special_tokens=False)[-1])
# print(tokenizer.decode(outputs[0]))

# https://huggingface.co/google/gemma-2b

activities = ["sleep", "sitting", "walking", "bicycling", "mixed-activity", "standing", "manual-work", "sports"]

x,y,z = 0, 0, 0


cot_template = f"""
    Context: 
        You are an expert in Physical Activity Classification.
        The data was recorded from Axivity AX3 wrist-worn tri-axial accelerometer on their dominant hand.
        The accelerometer was set to capture tri-axial acceleration data at 100 Hz with a dynamic range of ±8g.
        Wearable cameras were used to collect ground truths of the participants’ activities while wearing the accelerometers.
        Participants were given an OMG Life Autographer, a wearable camera worn around the neck which automatically takes photographs every 20 - 40 seconds
    
    Think step by step before classifying.
    
    Labels: {', '.join(activities)}
    Data: x = {x}, y = {y}, z = {z}
    
    Reasoning steps:
    1. Analyze the signal patterns
    2. Compare with known activity patterns
    3. Choose the most likely class
    
    Answer:
"""

dir = r"D:\kimia\Documents\University\UEA\Yr3 Project\Dataset\data\MASTER-DATA.csv"
df = pd.read_csv(dir)
# x = df.drop(columns=['time', 'annotation', 'label']).to_numpy()
# y = df['label'].to_numpy()


df_sample = df.sample(n=6000, replace=False, random_state=41) # take a random sample of 5000 from dataset
x = df_sample.drop(columns=['time', 'annotation', 'label'])
y = df_sample['label']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.8, random_state=41)
print(f"train size: {len(x_train)}, test size: {len(y_test)}")


"""
    - CHANGES
    - Few shot needs improving
"""

def zero_shot_prompting():

    """
    1. get data from dataset
    2. loop through data, prompt each time
    3. save predictions
    4. get accuracies
    5. save results

    """

    # only use test data
    print(x_test)
    print(y_test)
    print(len(x_test))
    print(len(y_test))

    pred_labels = []

    for i in range(len(x_test)):
        zero_shot_template = f"""
            Context:
            You are an expert in Physical Activity Classification.
            The data was recorded from Axivity AX3 wrist-worn tri-axial accelerometer on their dominant hand.
            The accelerometer was set to capture tri-axial acceleration data at 100 Hz with a dynamic range of ±8g.
            Wearable cameras were used to collect ground truths of the participants’ activities while wearing the accelerometers.
            Participants were given an OMG Life Autographer, a wearable camera worn around the neck which automatically takes photographs every 20 - 40 seconds

            Labels: {', '.join(activities)}
            Data: x = {x_test.iloc[i]['x']}, y = {x_test.iloc[i]['y']}, z = {x_test.iloc[i]['z']}
            Question: Which activity from the options above best matches the data?
            Answer:
        """

        # print(zero_shot_template)
        print(i)
        print(y_test.iloc[i])



        input_ids = tokenizer(zero_shot_template, return_tensors="pt").to("cuda")

        outputs = model.generate(**input_ids,
                                 max_new_tokens = 10,
                                 do_sample = False,
                                 eos_token_id = tokenizer.eos_token_id,
                                 pad_token_id = tokenizer.eos_token_id)

        response = tokenizer.decode(outputs[0], skip_special_tokens = True)
        print(response)

        try:
            # extract label
            label = re.findall(r"\*\*(.*?)\*\*", response)[0]
            print(label)

            # check label in activities
            if label in activities:
                pred_labels.append(label)
            else:
                pred_labels.append("N/A")

        except:
            pred_labels.append("N/A")


        if i == 50:
            break


    print(pred_labels)

    # map predicitions into encoded labels
    pred_labels_encoded = remap_labels(pred_labels, 'label_to_encode')
    print(pred_labels_encoded)

    # calc performance metrics
    # acc = accuracy_score(testLabel_values, pred_labels_encoded)
    # prec = precision_score(testLabel_values, pred_labels_encoded, average="weighted")
    # recall = recall_score(testLabel_values, pred_labels_encoded, average="weighted")
    # f1 = f1_score(testLabel_values, pred_labels_encoded, average="weighted")

    acc = accuracy_score(y_test.head(11), pred_labels_encoded)
    prec = precision_score(y_test.head(11), pred_labels_encoded, average="weighted")
    recall = recall_score(y_test.head(11), pred_labels_encoded, average="weighted")
    f1 = f1_score(y_test.head(11), pred_labels_encoded, average="weighted")

    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1-score: {f1 * 100:.2f}%")

    # record results
    model_type = "Zero-shot Prompting"
    model_data = [acc, prec, recall, f1]

    # cm = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, pred_labels_encoded))
    # cm.set_title("Zero-shot")


    save_results(model_type, model_data)


def few_shot_promptingTEST():
    """
        1. get data from dataset
        2. loop through data, prompt each time
        3. save predictions
        4. get accuracies
        5. save results

        """
    # only use training data for examples
    print(x_test)
    print(y_test)
    print(len(x_test))
    print(len(y_test))

    y_unencoded = remap_labels(y_train, "encode_to_label")
    pred_labels = []

    # training = x_train.copy()
    # training['label'] = y_unencoded
    # training_examples = training.groupby('label', group_keys=False).sample(n=1)

    for i in range(len(x_test)):
        # get random 1 sample of each class type to use as example

        # print(x_train)
        # print(y_train)

        training = x_train.copy()
        training['label'] = y_unencoded
        training_examples = training.groupby('label', group_keys=False).sample(n=1)
        # print(training_examples)
        # print(training_examples['label'].unique())

        few_shot_template = f"""
            Context:
            You are an expert in Physical Activity Classification.
            The data was recorded from Axivity AX3 wrist-worn tri-axial accelerometer on their dominant hand.
            The accelerometer was set to capture tri-axial acceleration data at 100 Hz with a dynamic range of ±8g.
            Wearable cameras were used to collect ground truths of the participants’ activities while wearing the accelerometers.
            Participants were given an OMG Life Autographer, a wearable camera worn around the neck which automatically takes photographs every 20 - 40 seconds

            Examples of correctly labelled data:
            x = {training_examples.iloc[0]['x']}, y = {training_examples.iloc[0]['y']}, z = {training_examples.iloc[0]['z']}, Label = {training_examples.iloc[0]['label']}
            x = {training_examples.iloc[1]['x']}, y = {training_examples.iloc[1]['y']}, z = {training_examples.iloc[1]['z']}, Label = {training_examples.iloc[1]['label']}
            x = {training_examples.iloc[2]['x']}, y = {training_examples.iloc[2]['y']}, z = {training_examples.iloc[2]['z']}, Label = {training_examples.iloc[2]['label']}
            x = {training_examples.iloc[3]['x']}, y = {training_examples.iloc[3]['y']}, z = {training_examples.iloc[3]['z']}, Label = {training_examples.iloc[3]['label']}
            x = {training_examples.iloc[4]['x']}, y = {training_examples.iloc[4]['y']}, z = {training_examples.iloc[4]['z']}, Label = {training_examples.iloc[4]['label']}
            x = {training_examples.iloc[5]['x']}, y = {training_examples.iloc[5]['y']}, z = {training_examples.iloc[5]['z']}, Label = {training_examples.iloc[5]['label']}
            x = {training_examples.iloc[6]['x']}, y = {training_examples.iloc[6]['y']}, z = {training_examples.iloc[6]['z']}, Label = {training_examples.iloc[6]['label']}
            x = {training_examples.iloc[7]['x']}, y = {training_examples.iloc[7]['y']}, z = {training_examples.iloc[7]['z']}, Label = {training_examples.iloc[7]['label']}

            Labels: {', '.join(activities)}
            Data: x = {x_test.iloc[i]['x']}, y = {x_test.iloc[i]['y']}, z = {x_test.iloc[i]['z']}
            Question: Which activity from the options above best matches the data?
            Answer:
        """



        # print(few_shot_template)
        print(i)
        print(y_test.iloc[i])



        input_ids = tokenizer(few_shot_template, return_tensors="pt").to("cuda")

        outputs = model.generate(**input_ids,
                                 max_new_tokens=15,
                                 do_sample=False,
                                 eos_token_id=tokenizer.eos_token_id,
                                 pad_token_id=tokenizer.eos_token_id)

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(response)

        try:
            # extract label
            label = re.findall(r"\*\*(.*?)\*\*", response)[0]
            print(label)

            # check label in activities
            if label in activities:
                pred_labels.append(label)
            else:
                pred_labels.append("N/A")

        except:
            pred_labels.append("N/A")

        if i == 50:
            break

    print(pred_labels)

    # map predicitions into encoded labels
    pred_labels_encoded = remap_labels(pred_labels, "label_to_encode")
    print(pred_labels_encoded)

    # # calc performance metrics
    # acc = accuracy_score(testLabel_values, pred_labels_encoded)
    # prec = precision_score(testLabel_values, pred_labels_encoded, average="weighted")
    # recall = recall_score(testLabel_values, pred_labels_encoded, average="weighted")
    # f1 = f1_score(testLabel_values, pred_labels_encoded, average="weighted")

    acc = accuracy_score(y_test.head(11), pred_labels_encoded)
    prec = precision_score(y_test.head(11), pred_labels_encoded, average="weighted")
    recall = recall_score(y_test.head(11), pred_labels_encoded, average="weighted")
    f1 = f1_score(y_test.head(11), pred_labels_encoded, average="weighted")

    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1-score: {f1 * 100:.2f}%")

    # record results
    model_type = "Few-shot Prompting"
    model_data = [acc, prec, recall, f1]

    save_results(model_type, model_data)


def few_shot_prompting():
    """
        1. get data from dataset
        2. loop through data, prompt each time
        3. save predictions
        4. get accuracies
        5. save results

        """
    print(df.head())
    print(x)
    print(y)
    print(len(x))
    print(len(y))

    # for i in range(0, df.shape[0], 5):
    #     testX = df['x'].iloc[i]
    #     print(i)

    # get every 5th record from dataset
    testX_values = df['x']  # gets every 5th value
    testY_values = df['y']
    testZ_values = df['z']
    testLabel_values = df['label']

    testLabels_unencoded = remap_labels(testLabel_values, "encode_to_label")

    print(len(testX_values))  # 302000

    pred_labels = []

    for i in range(0, len(testX_values), 5):  # every 5th index
        few_shot_template = f"""
            Context: 
            You are an expert in Physical Activity Classification.
            The data was recorded from Axivity AX3 wrist-worn tri-axial accelerometer on their dominant hand.
            The accelerometer was set to capture tri-axial acceleration data at 100 Hz with a dynamic range of ±8g.
            Wearable cameras were used to collect ground truths of the participants’ activities while wearing the accelerometers.
            Participants were given an OMG Life Autographer, a wearable camera worn around the neck which automatically takes photographs every 20 - 40 seconds

            Examples of correctly labelled data:
            x = {testX_values[i+1]}, y = {testY_values[i+1]}, z = {testZ_values[i+1]}, Label = {testLabels_unencoded[i+1]}
            x = {testX_values[i+2]}, y = {testY_values[i+2]}, z = {testZ_values[i+2]}, Label = {testLabels_unencoded[i+2]}
            x = {testX_values[i+3]}, y = {testY_values[i+3]}, z = {testZ_values[i+3]}, Label = {testLabels_unencoded[i+3]}
            x = {testX_values[i+4]}, y = {testY_values[i+4]}, z = {testZ_values[i+4]}, Label = {testLabels_unencoded[i+4]}

            Labels: {', '.join(activities)}
            Data: x = {testX_values[i]}, y = {testY_values[i]}, z = {testZ_values[i]}
            Question: Which activity from the options above best matches the data?
            Answer:
        """

        # print(few_shot_template)
        print(i)

        input_ids = tokenizer(few_shot_template, return_tensors="pt").to("cuda")

        outputs = model.generate(**input_ids,
                                 max_new_tokens=10,
                                 do_sample=False,
                                 eos_token_id=tokenizer.eos_token_id,
                                 pad_token_id=tokenizer.eos_token_id)

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(response)

        try:
            # extract label
            label = re.findall(r"\*\*(.*?)\*\*", response)[0]
            print(label)

            # check label in activities
            if label in activities:
                pred_labels.append(label)
            else:
                pred_labels.append("N/A")

        except:
            pred_labels.append("N/A")



        if i == 50:
            break
        # 30200 + (5)

    print(pred_labels)

    # map predicitions into encoded labels
    pred_labels_encoded = remap_labels(pred_labels, "label_to_encode")
    print(pred_labels_encoded)

    # # calc performance metrics
    # acc = accuracy_score(testLabel_values, pred_labels_encoded)
    # prec = precision_score(testLabel_values, pred_labels_encoded, average="weighted")
    # recall = recall_score(testLabel_values, pred_labels_encoded, average="weighted")
    # f1 = f1_score(testLabel_values, pred_labels_encoded, average="weighted")

    acc = accuracy_score(testLabel_values.head(11), pred_labels_encoded)
    prec = precision_score(testLabel_values.head(11), pred_labels_encoded, average="weighted")
    recall = recall_score(testLabel_values.head(11), pred_labels_encoded, average="weighted")
    f1 = f1_score(testLabel_values.head(11), pred_labels_encoded, average="weighted")

    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1-score: {f1 * 100:.2f}%")

    # record results
    model_type = "Few-shot Prompting"
    model_data = [acc, prec, recall, f1]

    save_results(model_type, model_data)


def remap_labels(label_list, direction):
    # --- New label mapping ---
    annotation_label_dir = r"D:\kimia\Documents\University\UEA\Yr3 Project\Dataset\data\annotation-label-encoded.csv"
    label_df = pd.read_csv(annotation_label_dir)

    # get all unique pairs
    labels = label_df[['label:WillettsSpecific2018', 'encoded_label']].drop_duplicates().reset_index(drop=True)
    labels = labels.drop(labels[labels['encoded_label'].isin([8, 9])].index)
    print(labels)

    if direction == "label_to_encode":
        mapping = dict(zip(
            labels["label:WillettsSpecific2018"],
            labels["encoded_label"]
        ))

        return [mapping.get(label, None) for label in label_list]

    elif direction == "encode_to_label":
        mapping = dict(zip(
            labels["encoded_label"],
            labels["label:WillettsSpecific2018"]
        ))

        return [mapping.get(code, None) for code in label_list]



def test():
    labels = ['sitting', 'walking', 'sitting']
    labels_encoded = remap_labels(labels, 'label_to_encode')
    print(labels_encoded)

    labels_enc = [0, 1, 5, 4]
    labels_unencoded = remap_labels(labels_enc, 'encode_to_label')
    print(labels_unencoded)

    metrics = [0.5564, 0.1452, 0.6542, 0.9856]
    save_results("test", metrics)

if __name__ == "__main__":
    # test()
    #
    zero_shot_prompting()
    # few_shot_prompting()


    # few_shot_promptingTEST()
