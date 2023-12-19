"""
@Project ：BDLocation 
@File    ：annotatedBPDataSet.py
@IDE     ：PyCharm 
@Author  ：姚聪
@Date    ：2023/12/18 16:31 
"""
import pandas as pd
from tqdm import tqdm, trange
import csv
import random
import numpy as np
import generatePointSet

accel_path = 'D:\\Desktop\\研究生毕设\\2023_11_20_12_16_52\\gyro_accel.csv'
data = pd.read_csv(accel_path)

sites_data = generatePointSet.generateStations()
sites = []
for item in sites_data:
    sites.append(item[0] - 1)
    sites.append(item[0])
    sites.append(item[0] + 1)
time_stamp = data['Unix time[nanosec]']
gx = data['gx[rad/s]']
gy = data['gy[rad/s]']
gz = data['gz[rad/s]']
ax = data['ax[m/s^2]']
ay = data['ay[m/s^2]']
az = data['az[m/s^2]']
title = ['time', 'gx', 'gy', 'gz', 'ax', 'ay', 'az', 'label']
true_results = []
false_results = []
for i in tqdm(range(len(time_stamp))):
    temp_time = int(time_stamp[i] / 1000000000)
    if temp_time in sites:
        true_results.append([temp_time, gx[i], gy[i], gz[i], ax[i], ay[i], az[i], 1])
    else:
        false_results.append([temp_time, gx[i], gy[i], gz[i], ax[i], ay[i], az[i], 0])
arr1 = np.array(true_results)
arr2 = np.array(false_results)
np.random.shuffle(arr1)
np.random.shuffle(arr2)
decline_ratio = int(0.03 * len(arr2))
u, n_u = arr2[:decline_ratio], arr2[decline_ratio:]
true_size = int(0.7 * len(arr1))
false_size = int(0.7 * len(u))
t1, t2 = arr1[:true_size], arr1[true_size:]
t3, t4 = u[:false_size], u[false_size:]
train = np.concatenate((t1, t3), axis=0)
test = np.concatenate((t2, t4), axis=0)
results = np.concatenate((train, test), axis=0)

file_name = 'data/CLD/gro_accel.csv'
with open(file_name, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(title)
with open(file_name, 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(results)
train_file_name = 'data/CLD/train.csv'
with open(train_file_name, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(title)
with open(train_file_name, 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(train)
test_file_name = 'data/CLD/test.csv'
with open(test_file_name, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(title)
with open(test_file_name, 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(test)
