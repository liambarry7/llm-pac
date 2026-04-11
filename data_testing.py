import pandas as pd

# dataset_dir = "D:\\kimia\\Documents\\University\\UEA\\Yr3 Project\\Dataset\\capture24"
dir = "D:\\kimia\\Documents\\University\\UEA\\Yr3 Project\\Dataset\\P001.csv"


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

def test():
    print("-- Initial data --")

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



    # --- remove unwanted labels ----
    # remove any unwanted labels (rows)
    unwanted_labels = [8, 9] # vehicle, household chores
    df = df.drop(df[df['label'].isin(unwanted_labels)].index)
    print(f"\nannotation + label + counts: \n {df[['annotation', 'label']].value_counts()}")


    # --- random sample to reduce dataset size ---
    print(f"\ndf size before sampling: {df.shape}")

    # sample with even balance of classes? - stratified sampling -----
    random_state = 42
    # take 10,000 samples from each individual
    df_sample = df.sample(n=10000, replace=False, random_state=random_state)

    print(f"\ndf size after sampling: {df_sample.shape}")
    print(df_sample.head())

    # normalised by window (e.g. every x seconds)
    # -- do this after random sampling



if __name__ == "__main__":
    test()
    # label_annotation_mapping()