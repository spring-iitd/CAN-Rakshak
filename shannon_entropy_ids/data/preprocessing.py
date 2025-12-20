# data/preprocessing.py
import ast

def extract_byte_values(df):
    """
    Robustly parses Raw_Data_Bytes column into list of int bytes.
    Handles:
    - "14 00 00 00 00 00 00 00"
    - "['01', '0A', 'FF', '00', ...]"
    """

    if "Raw_Data_Bytes" not in df.columns:
        raise ValueError("Expected column 'Raw_Data_Bytes' not found")

    def parse_bytes(x):
        if isinstance(x, list):
            return [int(b, 16) for b in x]

        x = str(x).strip()

        # Case 1: Python-list-like string
        if x.startswith("[") and x.endswith("]"):
            try:
                byte_list = ast.literal_eval(x)
                return [int(b, 16) for b in byte_list]
            except Exception:
                return []

        # Case 2: space-separated hex string
        try:
            return [int(b, 16) for b in x.split()]
        except Exception:
            return []

    df["Byte_Values"] = df["Raw_Data_Bytes"].apply(parse_bytes)
    return df
