import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_dataset():
    # link for dataset
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"

    # load the dataset directly from the web
    df = pd.read_csv(url, on_bad_lines="skip")

    # normalize column names just in case
    df.columns = [c.lower() for c in df.columns]

    # identify the temperature column
    if "temp" in df.columns:
        col = "temp"
    elif "mintemp" in df.columns:
        col = "mintemp"
    elif "tmin" in df.columns:
        col = "tmin"
    elif len(df.columns) == 2:
        col = df.columns[1]
    else:
        raise Exception(f"Could not identify temperature column. Found: {df.columns}")

    # convert to numeric and drop invalid rows
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[col])

    return df[col].values.reshape(-1, 1)


def scale_data(values):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)
    return scaled, scaler


def create_sequences(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window])
    return np.array(X), np.array(y)
