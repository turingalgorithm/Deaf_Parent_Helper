import pandas as pd
import numpy as np
# test_csv_path = "/tmp/pycharm/files/test_vision.csv"
# tmp = pd.read_csv(test_csv_path, header=None)
# tmp["label"] = -1
# tmp.to_csv("/tmp/pycharm_project_917/files/test_vision.csv", index=False, header=False)

data = pd.read_csv('./data/files/train.csv')

count_dict = {}

for idx in range(len(data)):
    entry = data.iloc[idx]
    values = np.load(entry['values'])
    # values = values.reshape(-1, 128, self.length)

    target = entry["target"]

    if target in count_dict:

        count_dict[target] += 1
    else:
        count_dict[target] = 1

print(count_dict)