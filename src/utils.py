import pandas as pd
import numpy as np

def read_csv(filename):
    df = pd.read_csv(filename)
    data_cols = df.columns[:7]
    for col in data_cols:
        df[col] = df[col].apply(lambda x: int(x, 16))
    return df.set_index("LETRA")[data_cols]

def to_bits(x):
    digito = "0" * 8
    str_repr = np.base_repr(x, 2)
    output = digito[:-len(str_repr)] + str_repr
    return np.array([int(x) for x in output[3:]])

def read_data(filename):
    df = read_csv(filename)
    matrices = {}
    for letra, byte_series in df.iterrows():
        matrices[letra] = np.vstack(byte_series.apply(to_bits).values)
    return matrices
