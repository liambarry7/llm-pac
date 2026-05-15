from statistics import mode

import pandas as pd
import numpy as np

import os

from sklearn.preprocessing import StandardScaler



# create split of train and test participants
participants = [f"P{i:03d}" for i in range(1, 152)]
training_participants = participants[:120]
test_participants = participants[120:]

def label_annotation_mapping():
    # use this function to create a new csv file for the
    # annotation-label dictionary, with each value mapped to a corresponding number
    dir = r"D:\kimia\Documents\University\UEA\Yr3 Project\Dataset\capture24\annotation-label-dictionary.csv"

    label_df = pd.read_csv(dir)

    print(label_df.head())
    print(label_df.columns)
    # print(label_df['label:WillettsSpecific2018'].value_counts())
    print(label_df['label:Walmsley2020'].value_counts())

    label_df = label_df[['annotation', 'label:Walmsley2020']].copy()
    print(label_df.head())
    print(label_df.columns)

    label_df['encoded_label'] = label_df['label:Walmsley2020'].astype('category').cat.codes

    # print label counts
    print(label_df[['label:Walmsley2020', 'encoded_label']].value_counts())

    # save as new csv for cross-reference when mapping labels for data
    save_dir = r'data\annotation-label-encoded.csv'
    label_df.to_csv(save_dir, mode='w', index=False)

def preprocess_dir(subjects, output_dir):
    # function to loop through all raw csvs and preprocess their data
    data_dir = r"D:\kimia\Documents\University\UEA\Yr3 Project\Dataset\capture24-csv"

    for f in os.scandir(data_dir):
        if f.is_file():
            raw_name = os.path.splitext(f.name)[0] # get file name without file extension (.csv)


            # only process selected participants
            if raw_name not in subjects:
                continue

            print("\n" + raw_name)
            print("\n" + os.path.join(data_dir, f.name))
            preprocess_file(os.path.join(data_dir, f.name), raw_name, output_dir)



def preprocess_file(file, f_name, output_dir):
    """
            NEW PIPELINE
            - split subjects (120 train/31 test)
            - remove missing values/duplicates
            - map labels onto data
            - remove unwanted labels
            - feature extraction
            - save

            - normalise (scaler - fit on train, transform on test, only after both have been computed and combined into full train/test)
            - random sample
            - save sampled set
        """

    # used to preprocess raw data in a given file and save the cleaned data
    df = pd.read_csv(file)
    print(df.head())
    print(f"File: ")
    print(f"Columns: {df.columns}")
    print(f"File size (total data points): {df.size}")
    print(f"df shape: {df.shape}")

    # get label values
    print(df['annotation'].value_counts())

    # remove missing values and duplicate rows
    df_na_dup = remove_na_dup(df)

    # map labels onto df
    df_clean = map_labels(df_na_dup)

    # remove unwanted labels
    # df_clean = remove_unwanted_labels(df_mapped, [8,9])

    # feature engineering
    feature_df = window_data(df_clean)
    feature_df = window_data(df_clean)

    print(feature_df.head())
    print(f"File: ")
    print(f"Columns: {feature_df.columns}")
    print(f"File size (total data points): {feature_df.size}")
    print(f"df shape: {feature_df.shape}")

    # save csv file (P001-S -> S for sampled)
    # dir = r"D:\kimia\Documents\University\UEA\Yr3 Project\Dataset\data"
    dir = r"data"
    feature_df.to_csv(f"{dir}\\{output_dir}\\{f_name}-F.csv", mode='w', index=False)



def remove_na_dup(df):
    # remove any rows with na values and duplicates

    print("\n-- Drop NA Rows --")
    row_len_before_drop = df.shape[0]
    nan_rows = df[df.isna().any(axis=1)]
    # print(f"nan rows: {nan_rows}")
    print(f"no of nan rows (rows with no labels): {len(nan_rows)}")
    df = df.dropna().reset_index(drop=True)

    print("\n-- Drop Duplicate Rows --")
    df = df.drop_duplicates().reset_index(drop=True)

    row_len_after_drop = df.shape[0]
    print(f"Rows removed: {row_len_before_drop - row_len_after_drop}")

    return df

def map_labels(df):
    # --- New label mapping ---
    print("\n-- Label Mapping --")
    # annotation_label_dir = r"D:\kimia\Documents\University\UEA\Yr3 Project\Dataset\data\annotation-label-encoded.csv"
    annotation_label_dir = r"data\annotation-label-encoded.csv"
    label_df = pd.read_csv(annotation_label_dir)
    print(label_df[['label:Walmsley2020', 'encoded_label']].value_counts())

    # make into dict for mapping
    label_dict = label_df.set_index('annotation')['encoded_label']

    # map onto main df
    df['label'] = df['annotation'].map(label_dict)

    print(f"\nannotation + label + counts: \n {df[['annotation', 'label']].value_counts()}")
    print(f"\nlabels with counts for each annotations: \n{df['annotation'].groupby(df['label']).value_counts()}")

    return df

# def remove_unwanted_labels(df, labels):
#     # remove unwanted labels from df
#     print("\n-- Remove Labels --")
#     df = df.drop(df[df['label'].isin(labels)].index)
#     print(f"\nannotation + label + counts: \n {df[['annotation', 'label']].value_counts()}")
#
#     return df

def window_data(df):
    windows = [] # list of rows (one per window)
    size = 500 # window size
    step_size = 250 # 50% overlap

    for start in range(0, len(df) - size, step_size):
        end = start + size
        window = df.iloc[start:end] # slice df to return window

        features = feature_extraction(window)

        windows.append(features)

    # convert list of windowed features back into df
    feature_df = pd.DataFrame(windows)
    return feature_df

def feature_extraction(window):
    features = {}

    # mean
    features['x_mean'] = window['x'].mean()
    features['y_mean'] = window['y'].mean()
    features['z_mean'] = window['z'].mean()

    # standard deviation
    features['x_std'] = window['x'].std()
    features['y_std'] = window['y'].std()
    features['z_std'] = window['z'].std()

    # min values
    features['x_min'] = window['x'].min()
    features['y_min'] = window['y'].min()
    features['z_min'] = window['z'].min()

    # max values
    features['x_max'] = window['x'].max()
    features['y_max'] = window['y'].max()
    features['z_max'] = window['z'].max()

    # avg magnitude
    magnitude = np.sqrt(window['x']**2 + window['y']**2 + window['z']**2)
    features['magnitude_mean'] = magnitude.mean()
    features['magnitude_std'] = magnitude.std()

    # RMS magnitude
    features['rms'] = np.sqrt(np.mean(magnitude**2))

    # jerk
    jerk = np.diff(magnitude)
    features['jerk'] = np.mean(np.abs(jerk)) if len(jerk) > 0 else 0

    # dominant frequency
    features['dominant_freq'] = 0
    fs = 100 # dataset sample rate (Hz)
    if len(magnitude) > 4:
        ffts_values = np.fft.rfft(magnitude - magnitude.mean())
        fft_mag = np.abs(ffts_values)
        freqs = np.fft.rfftfreq(len(magnitude), d=1/fs)
        dominant_idx =  np.argmax(fft_mag[1:]) + 1
        features['dominant_freq'] = freqs[dominant_idx]

    # get most common label
    features['label'] = window['label'].mode()[0]

    # print(features)

    return features



def combine_data(dir):
    # function to loop through cleaned csvs and concat them together to form a master cleaned file w/ all data
    # data_dir = r"D:\kimia\Documents\University\UEA\Yr3 Project\Dataset\data\cleaned-data"
    # master_df = pd.DataFrame(columns=['time', 'x', 'y', 'z', 'annotation', 'label'])

    # data_dir = r"D:\kimia\Documents\University\UEA\Yr3 Project\Dataset\data"
    data_dir = r"data"
    selected_dir = os.path.join(data_dir, dir)
    print(selected_dir)

    master_df = pd.DataFrame(columns=['x_mean', 'y_mean', 'z_mean', 'x_std', 'y_std', 'z_std', 'x_min',
     'y_min', 'z_min', 'x_max', 'y_max', 'z_max', 'magnitude_mean', 'magnitude_std', 'rms', 'jerk', 'dominant_freq', 'label'])

    for f in os.scandir(selected_dir):
        if f.is_file():
            # print(f.path)

            # combine all files in same directory as before
            # but have one folder for each train and test subjects

            print(f"File: {f.name}")
            df = pd.read_csv(f.path)
            master_df = pd.concat([master_df, df], axis=0, ignore_index=True)
            print("File combined.\n")

    print(master_df.info)

    # save master df as csv
    if dir == "train":
        # master_df.to_csv(r"D:\kimia\Documents\University\UEA\Yr3 Project\Dataset\data\TRAIN-DATA.csv", mode='w', index=False)
        master_df.to_csv(r"data\TRAIN-DATA.csv", mode='w', index=False)
    elif dir == "test":
        master_df.to_csv(r"data\TEST-DATA.csv", mode='w', index=False)

def scale_and_sample_data():
    # train_df = pd.read_csv(r"D:\kimia\Documents\University\UEA\Yr3 Project\Dataset\data\TRAIN-DATA.csv")
    # test_df = pd.read_csv(r"D:\kimia\Documents\University\UEA\Yr3 Project\Dataset\data\TEST-DATA.csv")
    train_df = pd.read_csv(r"data\TRAIN-DATA.csv")
    test_df = pd.read_csv(r"data\TEST-DATA.csv")
    
    scaler = StandardScaler()

    # separate labels from data
    train_labels = train_df['label']
    test_labels = test_df['label']

    x_train = train_df.drop(columns=['label'])
    x_test = test_df.drop(columns=['label'])

    # scale data
    train_scaled = scaler.fit_transform(x_train)
    test_scaled = scaler.transform(x_test)

    # convert back into df
    train_scaled = pd.DataFrame(train_scaled, columns=x_train.columns)
    test_scaled = pd.DataFrame(test_scaled, columns=x_test.columns)

    # read labels
    train_scaled['label'] = train_labels.values
    test_scaled['label'] = test_labels.values

    # sample datasets
    target_size = 100000
    train_sampled = sample_dataset(train_scaled, int(target_size*0.8), training=True)
    test_sampled = sample_dataset(test_scaled, int(target_size*0.2), training=False)

    print(f"Training size: {train_sampled.shape}")
    print(f"Testing size: {test_sampled.shape}")

    # save datasets - csv file (-SS for scaled + sampled)
    # train_sampled.to_csv(r"D:\kimia\Documents\University\UEA\Yr3 Project\Dataset\data\TRAIN-DATA-SS.csv", mode = 'w', index = False)
    # test_sampled.to_csv(r"D:\kimia\Documents\University\UEA\Yr3 Project\Dataset\data\TEST-DATA-SS.csv", mode = 'w', index = False)
    train_sampled.to_csv(r"data\TRAIN-DATA-SS.csv", mode = 'w', index = False)
    test_sampled.to_csv(r"data\TEST-DATA-SS.csv", mode = 'w', index = False)


def llm_sample_data():
    train_df = pd.read_csv(r"data\TRAIN-DATA.csv")
    test_df = pd.read_csv(r"data\TEST-DATA.csv")

    # sample datasets
    target_size = 25000 # 25% of the size for ML
    train_sampled = sample_dataset(train_df, int(target_size * 0.8), training=True)
    test_sampled = sample_dataset(test_df, int(target_size * 0.2), training=False)

    print(f"Training size: {train_sampled.shape}")
    print(f"Testing size: {test_sampled.shape}")

    # save datasets - csv file (-SS for scaled + sampled)
    train_sampled.to_csv(r"data\TRAIN-DATA-LLM-SS.csv", mode='w', index=False)
    test_sampled.to_csv(r"data\TEST-DATA-LLM-SS.csv", mode='w', index=False)

def sample_dataset(df, sample_size, training):
    # return a sample from the df
    print(f"\ndf size before sampling: {df.shape}")

    random_state = 42

    # if training set...
    if training:
        # stratified sampling to reduce class imbalance
        n_classes = df['label'].nunique()
        samples_per_class = sample_size // n_classes # int division to ensure whole number returned

        df_sample = df.groupby('label', group_keys=False).sample(n=samples_per_class, replace=False, random_state=random_state).reset_index(drop=True)

    else:
        df_sample = df.sample(n=sample_size, replace=False, random_state=random_state).reset_index(drop=True)

    print(f"\ndf size after sampling: {df_sample.shape}")
    print(df_sample.head())

    return df_sample

def get_feature_dataset():
    # dir1 = r"D:\kimia\Documents\University\UEA\Yr3 Project\Dataset\data\TRAIN-DATA-SS.csv"
    dir1 = r"data\TRAIN-DATA-SS.csv"
    df_train = pd.read_csv(dir1)

    # dir2 = r"D:\kimia\Documents\University\UEA\Yr3 Project\Dataset\data\TEST-DATA-SS.csv"
    dir2 = r"data\TEST-DATA-SS.csv"
    df_test = pd.read_csv(dir2)

    x_train = df_train.drop(columns=['label'])
    y_train = df_train['label']

    x_test = df_test.drop(columns=['label'])
    y_test = df_test['label']

    return x_train, x_test, y_train, y_test

def get_llm_dataset():
    dir1 = r"data\TRAIN-DATA-LLM-SS.csv"
    df_train = pd.read_csv(dir1)

    dir2 = r"data\TEST-DATA-LLM-SS.csv"
    df_test = pd.read_csv(dir2)

    x_train = df_train.drop(columns=['label'])
    y_train = df_train['label']

    x_test = df_test.drop(columns=['label'])
    y_test = df_test['label']

    return x_train, x_test, y_train, y_test

if __name__ == "__main__":
    # label_annotation_mapping()

    # preprocess data, split into training and test
    # preprocess_dir(training_participants, "train")
    # preprocess_dir(test_participants, "test")

    # combine individual preprocessed csvs
    # combine_data("train")
    # combine_data("test")

    # scale data (normalise)
    scale_and_sample_data()
    # llm_sample_data()


    # df = pd.read_csv(r"D:\kimia\Documents\University\UEA\Yr3 Project\Dataset\data\MASTER-DATA.csv")
    # print(df.shape)
    # print(df.columns)
    # print(f"\nannotation + label + counts: \n {df[['label']].value_counts()}")

    # test_harness()