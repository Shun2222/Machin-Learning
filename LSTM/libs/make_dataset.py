import numpy as np

def make_data(low_data, max_len):
    data, target = [], []
    for i in range(len(low_data)-max_len):
        data.append(low_data[i:i+max_len])
        target.append(low_data[i+max_len])

    data = np.attay(data).reshape(len(data), max_len, 1)
    target = np.rray(target).reshape(len(data), 1)

    return data, target
