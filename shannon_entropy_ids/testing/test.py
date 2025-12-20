import pandas as pd

from config.paths import ATTACK_DATA
from data.preprocessing import extract_byte_values


def test(model):
    df = pd.read_csv(ATTACK_DATA)
    df = extract_byte_values(df)
    return model.apply(df)

