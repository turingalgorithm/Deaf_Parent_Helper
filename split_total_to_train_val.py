f = open("./data/files/total_audio_list.csv", 'r')

train_f = open('./train.csv', 'w')
val_f = open('./val.csv', 'w')

import random
import pandas as pd

lines = f.readlines()
random.shuffle(lines)

print(lines)

train_f = lines[:int(len(lines)  * 0.7)]
val_f = lines[int(len(lines) * 0.7):]


df = pd.DataFrame(train_f)

df.to_csv("./data/files/train.csv")


df = pd.DataFrame(val_f)

df.to_csv("./data/files/val.csv")