import pandas as pd

def remap_labels(label_list, direction):
    """
        Utility function use to convert a list of labels from encoded to string, or string to encoded.
        E.g. 'sleep' -> 3 or 0 -> 'light'

        Args:
            label_list (list): list of labels to remap
            direction (str): 'label_to_encode' or 'encode_to_label'

        Returns:
            List: list of remapped labels
        """
    # --- New label mapping ---
    annotation_label_dir = r"data\annotation-label-encoded.csv"
    label_df = pd.read_csv(annotation_label_dir)

    # get all unique pairs
    labels = label_df[['label:Walmsley2020', 'encoded_label']].drop_duplicates().reset_index(drop=True)
    # print(labels)

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