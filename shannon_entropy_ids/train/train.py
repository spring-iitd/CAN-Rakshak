import pandas as pd

from config.paths import NORMAL_DATA
from data.preprocessing import extract_byte_values
from model.shannon import ShannonIDS
from config.constants import TIME_WINDOW, K_FACTOR


def train():
    df = pd.read_csv(NORMAL_DATA)
    df = extract_byte_values(df)

    model = ShannonIDS(TIME_WINDOW, K_FACTOR)
    model.fit(df)

    return model
