"""
@Project ：BDLocation 
@File    ：arrangeMapData.py
@IDE     ：PyCharm 
@Author  ：姚聪
@Date    ：2023/12/8 12:33 
"""

import pandas as pd
import matplotlib.pyplot as plt
import geometricCalculate
import csv
from tqdm import tqdm, trange

data = pd.read_csv('data/mapData.csv')
start_point_x = data['Start_x']
start_point_y = data['Start_y']
end_point_x = data['End_x']
end_point_y = data['End_y']
points = []
for i in range(0, len(start_point_x)):
    point1 = [start_point_x[i], start_point_y[i]]
    if point1 not in points:
        points.append(point1)
    point2 = [end_point_x[i], end_point_y[i]]
    if point2 not in points:
        points.append(point2)

img = plt.imread("data/1.jpg")
fig, ax = plt.subplots()
ax.imshow(img, extent=[-850, 220, -350, 130])
# for item in points:
#     plt.scatter(item[0], item[1], label='Data Points', color='r', s=10)
# plt.show()

samePoints = set()
size = len(points)
for m in range(size):
    for n in range(size):
        if m != n:
            if 0 < geometricCalculate.computeDistance(points[m], points[n]) < 3:
                samePoints.add(tuple(points[m]))
                samePoints.add(tuple(points[n]))
                print(points[m], '   ', points[n])

print(len(samePoints))
if len(samePoints) == 0:
    # 生成mapPoint
    file_name = 'data/mapPoints.csv'
    data_label = ['Pid', 'x', 'y']

    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data_label)

    for i in range(size):
        with open(file_name, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([i, points[i][0], points[i][1]])
for item in samePoints:
    plt.scatter(item[0], item[1], label='Data Points', s=10)
plt.show()
