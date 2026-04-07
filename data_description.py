import pandas as pd

# dataset_dir = "D:\\kimia\\Documents\\University\\UEA\\Yr3 Project\\Dataset\\capture24"
dir = "D:\\kimia\\Documents\\University\\UEA\\Yr3 Project\\Dataset\\P001.csv"


if __name__ == "__main__":
    print("-- Initial data --")


    # df = pd.read_csv(f"{dataset_dir}\\P001.csv.gz\\P001.csv")
    df = pd.read_csv(dir)
    print(df.head())
    print(df.columns)
    print(df.size)
    print(df.shape)

    # get label values
    print(df['annotation'].value_counts())

    # drop na
    print("\n-- Drop NA Rows --")
    nan_rows = df[df.isna().any(axis=1)]
    print(f"nan rows: {nan_rows}")
    print(f"nan rows len: {len(nan_rows)}")
    df = df.dropna().reset_index(drop=True)

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


    # ---
    # remove any unwanted labels (rows)
    # remove/encode labels => do this from label.csv file so uniform for all datasets?
    # sample with even balance of classes?

    # normalised by window (e.g. every x seconds)
    # -- do this after random sampling
    r_df = df.sample(n=1000)
    print(r_df.head(15))
