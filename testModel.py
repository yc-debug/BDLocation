"""
@Project ：BDLocation 
@File    ：testModel.py
@IDE     ：PyCharm 
@Author  ：姚聪
@Date    ：2023/12/19 13:30 
"""
import pandas as pd
import numpy as np
import torch
from torch import nn

accel_path = 'data/CLD/train.csv'
test_data = pd.read_csv(accel_path)
df = pd.DataFrame(test_data)
test_x = df.iloc[:, 1:len(df.columns) - 1]
# print(test)
test_x = torch.from_numpy(np.array(test_x)).float()
model_path = 'models/model_12_18.pth'
model = torch.load(model_path)
model.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = model.to(device)
test_x = test_x.to(device)
outputs = (model(test_x) > 0.5).float()
length = len(outputs)
test_y = df.iloc[:, -1]
test_y = torch.from_numpy(np.array(test_y).reshape(-1, 1)).float()
test_y = test_y.to(device)
accuracy = (outputs == test_y).float().mean()
print(accuracy.item())
for i in range(length):
    if outputs[i].item() == 1:
        print(outputs[i].item())
