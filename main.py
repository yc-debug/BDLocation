import pandas as pd
import numpy as np
import sys
import pyproj
import generatePointSet
import geometricCalculate
import KalmanFilter
import matplotlib.pyplot as plt
import math
from tqdm import tqdm, trange


# 读取数据
def get_data():
    temp = pd.read_csv('data/dataset2.csv')
    return temp


# 均值
u = 4.07
data = get_data()
Longitude = data['Longitude']
Latitude = data['Latitude']

Acceleration_X = data['Acceleration_X']
Acceleration_Y = data['Acceleration_Y']

start_point = [116.3403876, 39.9509437]
proj = pyproj.Proj(proj='tmerc', lon_0=start_point[0], lat_0=start_point[1],
                   preserve_units=False)
# proj = pyproj.Proj(proj='tmerc', lon_0=Longitude[len(Longitude) - 1], lat_0=Latitude[len(Latitude) - 1],
#                    preserve_units=False)
# print(proj(Longitude[0], Latitude[0]))
xy_coor = []
for i in range(0, len(Latitude)):
    xy_coor.append(proj(Longitude[i], Latitude[i]))
# 采样1000次
size = len(data)
# print(size)
x_pred, y_pred = KalmanFilter.KalMan(xy_coor, size, Acceleration_X, Acceleration_Y)
true_data = generatePointSet.generateEdgeSet()


def get_emission_probability(x):
    return (1 / ((2 * math.pi) ** 0.5) * u) * math.e ** (-0.5 * ((x / u) ** 2))


def get_max_probability_point(points, decline_noise_point):
    estimate_point = []
    p = 0
    for item in points:
        probability = get_emission_probability(geometricCalculate.computeDistance(item, decline_noise_point))
        if probability > p:
            estimate_point = item
            p = probability
    return np.array(estimate_point)


# 计算具体位置
result = []
for i in tqdm(range(0, len(xy_coor))):
    p_point = [x_pred[i], y_pred[i]]
    intersections = []
    onlyOneIntersection = True
    for j in range(1, len(true_data)):
        start = np.array(true_data[j][0])
        end = np.array(true_data[j][1])
        res = geometricCalculate.computeIntersection(start=start, end=end, circle_center=np.array(xy_coor[i]), radius=u)
        if len(res) == 0:
            continue
        if len(res) == 1 and onlyOneIntersection:
            distance = geometricCalculate.computeDistance(res[0], xy_coor[i])
            if distance > 4 * u:
                continue
            intersections.append(res[0])

        if len(res) > 1:
            if onlyOneIntersection:
                intersections = []
            onlyOneIntersection = False
            for temp in res:
                intersections.append(temp)
        # if (start == np.array([-85.97110215, -190.6320144])).all() and 69 <= i <= 70:
        #     print(i)
        #     print('intersections:', intersections)
        #     print('res:', res)

    if len(intersections) == 0:
        continue
        # point = np.array(p_point)
    # 计算概率最大点
    point = get_max_probability_point(intersections, p_point)
    result.append(point)

img = plt.imread("data/1.jpg")
# plt.xlim(-850, 220)
# plt.ylim(-350, 130)
fig, ax = plt.subplots()
ax.imshow(img, extent=[-850, 220, -350, 130])
# plt.plot(*zip(*xy_coor), alpha=0.5, color='red', label='measurement value')
# plt.plot(*zip(*result), color='blue', label="estimate value")
plt.scatter(*zip(*xy_coor), s=5, color='red')
# plt.plot(*zip(*result), color='blue', label="estimate value")
plt.scatter(*zip(*result), s=5)
# plt.plot(*zip(*true_data), alpha=0.5, color='green', label='true value')
plt.show()
