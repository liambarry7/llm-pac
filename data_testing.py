import pandas as pd
import numpy as np

import os

# dataset_dir = "D:\\kimia\\Documents\\University\\UEA\\Yr3 Project\\Dataset\\capture24"


def label_annotation_mapping():
    # use this function to create a new csv file for the
    # annotation-label dictionary, with each value mapped to a corresponding number
    dir = r"D:\kimia\Documents\University\UEA\Yr3 Project\Dataset\capture24\annotation-label-dictionary.csv"

    label_df = pd.read_csv(dir)

    print(label_df.head())
    print(label_df.columns)
    print(label_df['label:WillettsSpecific2018'].value_counts())

    # drop unneeded label types, keep 'WillettsSpecific2018' - explain why
    label_df = label_df.drop(['label:WillettsMET2018',
                              'label:DohertySpecific2018', 'label:Willetts2018', 'label:Doherty2018',
                              'label:Walmsley2020'], axis=1)

    print(label_df.head())
    print(label_df.columns) # Index(['annotation', 'label:WillettsSpecific2018'], dtype='object'

    # encode the labels for later use in classification
    # label_df['encoded_label'] = label_df['label:WillettsSpecific2018'].astype('category').cat.codes
    # print(label_df.head())

    # hardcode labels so that classes that will be removed will keep labels in order 1-N, without gaps
    labels = label_df['label:WillettsSpecific2018'].unique().tolist()
    redundant_classes = ['vehicle', 'household-chores'] # list of classes that will be omitted from ML models
    important_classes = [c for c in labels if c not in redundant_classes] # create list of classes that will be used, ignoring those that won't
    classes = important_classes + redundant_classes
    label_map = {label: l for l, label in enumerate(classes)}
    label_df['encoded_label'] = label_df['label:WillettsSpecific2018'].map(label_map)

    # print label counts
    print(label_df[['label:WillettsSpecific2018', 'encoded_label']].value_counts())

    # save as new csv for cross-reference when mapping labels for data
    save_dir = r'D:\kimia\Documents\University\UEA\Yr3 Project\Dataset\data\annotation-label-encoded.csv'
    label_df.to_csv(save_dir, mode='w', index=False)

def preprocess_dir():
    # function to loop through all raw csvs and preprocess their data
    data_dir = r"D:\kimia\Documents\University\UEA\Yr3 Project\Dataset\capture24-csv"

    for f in os.scandir(data_dir):
        if f.is_file():
            raw_name = os.path.splitext(f.name)[0] # get file name without file extension (.csv)
            print("\n" + raw_name)
            # print("\n" + os.path.join(data_dir, f.name))
            preprocess_file(os.path.join(data_dir, f.name), raw_name)

def combine_data():
    # function to loop through cleaned csvs and concat them together to form a master cleaned file w/ all data
    data_dir = r"D:\kimia\Documents\University\UEA\Yr3 Project\Dataset\data\cleaned-data"
    master_df = pd.DataFrame(columns=['time', 'x', 'y', 'z', 'annotation', 'label'])
    for f in os.scandir(data_dir):
        if f.is_file():
            # print(f.path)
            print(f"File: {f.name}")
            df = pd.read_csv(f.path)
            master_df = pd.concat([master_df, df], axis=0, ignore_index=True)
            print("File combined.\n")

    print(master_df.info)

    # save master df as csv
    master_df.to_csv(r"D:\kimia\Documents\University\UEA\Yr3 Project\Dataset\data\MASTER-DATA.csv", mode='w', index=False)


def preprocess_file(file, f_name):
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
    df_mapped = map_labels(df_na_dup)

    # remove unwanted labels
    df_clean = remove_unwanted_labels(df_mapped, [8,9])

    # normalise the dataset (x,y,z) by participant using z-score - explain why
    df_norm = normalisation(df_clean, ["x", "y", "z"])

    # sample dataset
    df_sampled = sample_dataset(df_norm, 1000)

    # save csv file (P001-S -> S for sampled)
    dir = r"D:\kimia\Documents\University\UEA\Yr3 Project\Dataset\data\cleaned-data"
    df_sampled.to_csv(f"{dir}\\{f_name}-S.csv", mode='w', index=False)

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
    annotation_label_dir = r"D:\kimia\Documents\University\UEA\Yr3 Project\Dataset\data\annotation-label-encoded.csv"
    label_df = pd.read_csv(annotation_label_dir)
    print(label_df[['label:WillettsSpecific2018', 'encoded_label']].value_counts())

    # make into dict for mapping
    label_dict = label_df.set_index('annotation')['encoded_label']

    # map onto main df
    df['label'] = df['annotation'].map(label_dict)

    print(f"\nannotation + label + counts: \n {df[['annotation', 'label']].value_counts()}")
    print(f"\nlabels with counts for each annotations: \n{df['annotation'].groupby(df['label']).value_counts()}")

    return df

def remove_unwanted_labels(df, labels):
    # remove unwanted labels from df
    print("\n-- Remove Labels --")
    df = df.drop(df[df['label'].isin(labels)].index)
    print(f"\nannotation + label + counts: \n {df[['annotation', 'label']].value_counts()}")

    return df

def normalisation(df, columns):
    # normalise raw data in df
    for col in columns:
        df[col] = zscore(df[col])

    return df

def zscore(column):
    # z-score = (data - population mean) / population sd
    col_m = np.mean(column)
    col_std = np.std(column)
    return (column - col_m) / col_std

def sample_dataset(df, sample_size):
    # return a sample from the df
    print(f"\ndf size before sampling: {df.shape}")

    # sample with even balance of classes? - stratified sampling -----
    random_state = 42
    # take 10,000 samples from each individual
    df_sample = df.sample(n=sample_size, replace=False, random_state=random_state)

    print(f"\ndf size after sampling: {df_sample.shape}")
    print(df_sample.head())

    return df_sample

def test_play():
    print("-- Initial data --")
    dir = "D:\\kimia\\Documents\\University\\UEA\\Yr3 Project\\Dataset\\P001-T.csv"

    # df = pd.read_csv(f"{dataset_dir}\\P001.csv.gz\\P001.csv")
    df = pd.read_csv(dir)
    print(df.head())
    print(f"Columns: {df.columns}")
    print(f"File size (total data points): {df.size}")
    print(f"df shape: {df.shape}")

    # get label values
    print(df['annotation'].value_counts())

    # drop na
    row_len_before_drop = df.shape[0]
    print("\n-- Drop NA Rows --")
    nan_rows = df[df.isna().any(axis=1)]
    print(f"nan rows: {nan_rows}")
    print(f"no of nan rows (rows with no labels): {len(nan_rows)}")
    df = df.dropna().reset_index(drop=True)

    row_len_after_drop = df.shape[0]
    nan_rows_removed = row_len_before_drop - row_len_after_drop
    print(f"Rows removed: {nan_rows_removed}")
    print(df.head())
    print(df.columns)
    print(df.size)
    print(df.shape)

    # drop duplicates
    print("\n-- Drop Duplicate Rows --")
    df = df.drop_duplicates().reset_index(drop=True)

    # df = remove_na_dup(df)

    print(df.head())
    print(df.columns)
    print(df.size)
    print(df.shape)

    """
    # map labels onto activity annotation
    print("\n-- Label Mapping --")
    annotation_label_dir = "D:\\kimia\\Documents\\University\\UEA\\Yr3 Project\\Dataset\\capture24\\annotation-label-dictionary.csv"
    label_df = pd.read_csv(annotation_label_dir)
    print(label_df.head())
    print(label_df.columns)

    # drop unneeded label types - explain why
    label_df = label_df.drop(['label:WillettsMET2018',
                              'label:DohertySpecific2018', 'label:Willetts2018', 'label:Doherty2018',
                              'label:Walmsley2020'], axis=1)

    print(label_df.head())
    print(label_df.columns)

    # make into dict for mapping
    label_dict = label_df.set_index('annotation')['label:WillettsSpecific2018']

    # map onto main df
    df['label'] = df['annotation'].map(label_dict)

    print(df.head())
    print(df.columns)
    """

    # --- New label mapping ---
    print("\n-- New Label Mapping --")
    annotation_label_dir = r"D:\kimia\Documents\University\UEA\Yr3 Project\Dataset\data\annotation-label-encoded.csv"
    label_df = pd.read_csv(annotation_label_dir)
    print(label_df.head())
    print(label_df.columns)
    print(label_df[['label:WillettsSpecific2018', 'encoded_label']].value_counts())

    # make into dict for mapping
    label_dict = label_df.set_index('annotation')['encoded_label']

    # map onto main df
    df['label'] = df['annotation'].map(label_dict)

    print(df.head())
    print(df.columns)
    print(f"\nannotation + label + counts: \n {df[['annotation', 'label']].value_counts()}")
    print(f"\nlabels with counts for each matching annotations: \n{df['annotation'].groupby(df['label']).value_counts()}")

    # df = map_labels(df)

    # --- remove unwanted labels ----
    # remove any unwanted labels (rows)
    unwanted_labels = [8, 9] # vehicle, household chores
    df = df.drop(df[df['label'].isin(unwanted_labels)].index)
    print(f"\nannotation + label + counts: \n {df[['annotation', 'label']].value_counts()}")

    # df = remove_unwanted_labels(df, [8,9])

    # normalised by window (e.g. every x seconds)
    # -- do this before random sampling
    # -- with feature extraction??


    # --- random sample to reduce dataset size ---
    print(f"\ndf size before sampling: {df.shape}")

    # sample with even balance of classes? - stratified sampling -----
    random_state = 42
    # take 10,000 samples from each individual
    df_sample = df.sample(n=10000, replace=False, random_state=random_state)

    print(f"\ndf size after sampling: {df_sample.shape}")
    print(df_sample.head())

    # df = sample_dataset(df)

    # save csv file (P001-T -> T for test)
    df_sample.to_csv("D:\\kimia\\Documents\\University\\UEA\\Yr3 Project\\Dataset\\P001-T.csv", mode='w', index=False)

def test_harness():
    dir = "D:\\kimia\\Documents\\University\\UEA\\Yr3 Project\\Dataset\\P001-T.csv"

    df = pd.read_csv(dir)
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
    df_mapped = map_labels(df_na_dup)

    # remove unwanted labels
    df_clean = remove_unwanted_labels(df_mapped, [8, 9])

    df_norm = normalisation(df_clean, ["x", "y", "z"])
    print(f"Columns: {df_norm.columns}")
    print(f"{df_norm.head()}")

    # sample dataset
    df_sampled = sample_dataset(df_norm, 1000)

    print(f"Columns: {df_sampled.columns}")
    print(f"File size (total data points): {df_sampled.size}")
    print(f"df shape: {df_sampled.shape}")
    print(df.head())



if __name__ == "__main__":
    # test_play()
    # label_annotation_mapping()
    # preprocess_dir()
    # combine_data()

    df = pd.read_csv(r"D:\kimia\Documents\University\UEA\Yr3 Project\Dataset\data\MASTER-DATA.csv")
    print(df.shape)
    print(df.columns)
    print(f"\nannotation + label + counts: \n {df[['label']].value_counts()}")

    # test_harness()