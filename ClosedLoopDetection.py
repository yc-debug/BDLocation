"""
@Project ：BDLocation 
@File    ：ClosedLoopDetection.py
@IDE     ：PyCharm 
@Author  ：姚聪
@Date    ：2023/12/18 17:31 
"""
import os
import pandas as pd
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

torch.set_printoptions(threshold=np.inf)
train = pd.read_csv('data/CLD/train.csv')
test = pd.read_csv('data/CLD/test.csv')
df_train = pd.DataFrame(train)
df_test = pd.DataFrame(test)
train_x = df_train.iloc[:, 1:len(df_train.columns) - 1]
train_y = df_train.iloc[:, -1]
test_x = df_test.iloc[:, 1:len(df_test.columns) - 1]
test_y = df_test.iloc[:, -1]
train_x = torch.from_numpy(np.array(train_x)).float()
train_y = torch.from_numpy(np.array(train_y).reshape(-1, 1)).float()
test_x = torch.from_numpy(np.array(test_x)).float()
test_y = torch.from_numpy(np.array(test_y).reshape(-1, 1)).float()
model_path = 'models/model_12_19_500000.pth'
# if os.path.exists(model_path):
#     print("Exist!")
#     model = torch.load(model_path)
#     model.eval()
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model = model.to('cpu')
#     # test_x = test_x.to(device)
#     # test_y = test_y.to(device)
#     outputs = (model(test_x) > 0.5).float()
#     accuracy = (outputs == test_y).float().mean()
#     print(accuracy.item())
#     length = len(outputs)
#     for i in range(length):
#         if outputs[i].item() == 1:
#             print(outputs[i].item())
# else:
model = nn.Sequential(nn.Linear(6, 64), nn.ReLU(), nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, 4), nn.ReLU(),
                      nn.Linear(4, 1), nn.Sigmoid())
Loss = nn.BCELoss()
optim = torch.optim.SGD(params=model.parameters(), lr=0.01)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model = model.to(device)
train_x = train_x.to(device)
train_y = train_y.to(device)
test_x = test_x.to(device)
test_y = test_y.to(device)

loss_collection = []
n = 500000
for i in range(n):
    optim.zero_grad()
    yp = model(train_x)
    loss = Loss(yp, train_y)
    loss.backward()
    optim.step()
    loss_collection.append(loss.item())
    if i % 10000 == 0:
        outputs = (model(test_x) > 0.5).float()
        accuracy = (outputs == test_y).float().mean()
        print('epoch: {}, loss: {}, accuracy:{}'.format(i, loss.item(), accuracy))
# 保存模型

torch.save(model, model_path)
# 加载模型
# model = torch.load(model_path)
# model.eval()
outputs = (model(test_x) > 0.5).float()
accuracy = (outputs == test_y).float().mean()
print(accuracy)
length = len(outputs)
# print(length)
num = 0
for i in range(length):
    if outputs[i] == 1.0:
        num += 1.0
print(num / length)
#
x_train_loss = range(n)
plt.figure()
plt.xlabel('iters')  # x轴标签
plt.ylabel('loss')  # y轴标签
plt.plot(x_train_loss, loss_collection, label='train loss')
plt.legend()
plt.title('loss curve')
plt.show()
