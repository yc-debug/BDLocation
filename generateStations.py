"""
@Project ：BDLocation 
@File    ：generateStations.py
@IDE     ：PyCharm 
@Author  ：姚聪
@Date    ：2023/12/19 15:41 
"""
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt

file_name = 'data/stations.csv'
# sites_time = [1700454487, 1700454723, 1700454907, 1700455017]
# sites_location = [[-671.7, -183.6], [-474.69, -238.65], [-359.224, -95.3531], [-215.406586, -61.19653058]]
# res = [['time_stamp', 'x', 'y']]
# for i in range(len(sites_time)):
#     res.append([sites_time[i], sites_location[i][0], sites_location[i][1]])
# with open(file_name, 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows(res)
data = pd.read_csv(file_name)
img = plt.imread("data/1.jpg")
fig, ax = plt.subplots()
ax.imshow(img, extent=[-850, 220, -350, 130])
x = data['x']
y = data['y']
plt.scatter(x, y, label='Data Points', s=5)
plt.show()
