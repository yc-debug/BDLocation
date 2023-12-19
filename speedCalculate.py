"""
@Project ：BDLocation 
@File    ：speedCalculate.py
@IDE     ：PyCharm 
@Author  ：姚聪
@Date    ：2023/12/14 20:42 
"""

import pandas as pd
import matplotlib.pyplot as plt
import geometricCalculate
import csv
from tqdm import tqdm, trange

# accel_path = 'D:\\Desktop\\研究生毕设\\2023_10_17_11_58_07\\gyro_accel.csv'
accel_path = 'D:\\Desktop\\研究生毕设\\2023_10_17_11_58_07\\gyro_accel.csv'
data = pd.read_csv(accel_path)
new_accel_path = 'D:\\Desktop\\研究生毕设\\2023_10_17_11_58_07\\gyro_accel_1.csv'

u_t = data['Unix time[nanosec]']
ac_x = data['ax[m/s^2]']
ac_y = data['ay[m/s^2]']
ac_z = data['az[m/s^2]']

file_name = 'data/gro_accel.csv'

res = [['time', 'ax', 'ay', 'az', 'vx', 'vy', 'vz', 's'], [0, ac_x[0], ac_x[0], ac_x[0], 0, 0, 0, 0]]
scale = 0.380229
for i in range(1, len(u_t)):
    time_dis = (u_t[i] - u_t[i - 1]) / 1000000000
    item = res[-1]
    vx = ac_x[i] * scale
    vy = ac_y[i] * scale
    vz = ac_z[i] * scale
    # vx = dvx
    # vy = dvy
    # vz = dvz
    # sx = vx * time_dis
    # sy = vy * time_dis
    # x =
    v = (vx * vx + vy * vy) ** 0.5
    # ds = v * time_dis + 0.5 * a * (time_dis**0.5)
    ds = v * time_dis
    s = item[7] + ds
    # temp = [u_t[i], ac_x[i], ac_y[i], ac_z[i], vx, vy, vz, s]
    temp = [time_dis, ac_x[i], ac_y[i], ac_z[i], vx, vy, vz, s]
    res.append(temp)

print(res[-1][7])
with open(new_accel_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(res)
