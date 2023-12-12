"""
@Project ：BDLocation 
@File    ：main_dev.py
@IDE     ：PyCharm 
@Author  ：姚聪
@Date    ：2023/12/10 15:00 
"""
import pandas as pd
import numpy as np
import sys
import pyproj
import generatePointSet
import geometricCalculate
import getAllPath
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
scale = 1.071379011879281
# 将均值换算到坐标上
u = u * scale
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
    edge = 0
    p = 0
    for key in points.keys():
        for item in points[key]:
            probability_e = get_emission_probability(geometricCalculate.computeDistance(item, decline_noise_point))
            if probability_e > p:
                estimate_point = item
                edge = key
                p = probability_e
    return edge, np.array(estimate_point)


def enhance_result(points, edge):
    enhance_path = [(points[0])]
    for p in range(1, len(points)):
        if edge[p] == edge[p - 1]:
            enhance_path.append(points[p])
            continue
        else:
            edge1 = true_data[edge[p - 1]]
            edge2 = true_data[edge[p]]
            if edge1[0] == edge2[0]:
                enhance_path.append(edge1[0])
            elif edge1[0] == edge2[1]:
                enhance_path.append(edge1[0])
            elif edge1[1] == edge2[0]:
                enhance_path.append(edge1[1])
            elif edge1[1] == edge2[1]:
                enhance_path.append(edge1[1])
            enhance_path.append(points[p])
    return enhance_path


def get_path_length(path):
    length = 0
    for i in range(1, len(path)):
        point1 = path[i-1]
        point2 = path[i]
        length += geometricCalculate.computeDistance(point1, point2)
    return length


def transfer_formula(length, dis):
    return 1-math.fabs(1-(length/dis))


def get_transfer_probability(pre_p, pre_e, p, e, dis):
    # 判断是否在同一条直线上
    if pre_e == e:
        return transfer_formula(get_path_length([pre_p, p]), dis)
    max_p = 0
    # dis = dis * scale
    # 计算路径
    # 开始节点到开始节点
    pre_start_to_p_start_path = getAllPath.getAllPath(pre_e[0], e[0], dis)
    for path in pre_start_to_p_start_path:
        if len(path) < 2:
            continue
        if (np.array(path[1]) == np.array(pre_e[1])).all():
            path[0] = pre_p
        else:
            path.insert(0, pre_p)
        if (np.array(path[-2]) == np.array(e[1])).all():
            path[-1] = p
        else:
            path.append(p)
        # 计算路径长度
        length = get_path_length(path)
        # 计算该条路径的转移概率
        probability_t = transfer_formula(length, 100)
        # 获取最大转移概率
        if probability_t > max_p:
            max_p = probability_t
    return max_p


# 计算具体位置
result = []
edge_nums = []
for i in tqdm(range(0, len(xy_coor))):
    # 获取第i个通过卡尔曼滤波获取的点
    p_point = [x_pred[i], y_pred[i]]
    intersections = {}
    onlyOneIntersection = True
    for j in range(1, len(true_data)):
        start = np.array(true_data[j][0])
        end = np.array(true_data[j][1])
        res = geometricCalculate.computeIntersection(start=start, end=end, circle_center=np.array(xy_coor[i]), radius=u)
        if len(res) == 0:
            continue
        if len(res) == 1 and onlyOneIntersection:
            distance = geometricCalculate.computeDistance(res[0], xy_coor[i])
            if distance > 5 * u:
                continue
            intersections[j] = res

        if len(res) > 1:
            if onlyOneIntersection:
                intersections = {}
            onlyOneIntersection = False
            intersections[j] = res
    if i == 0:
        edge_num, point = get_max_probability_point(intersections, xy_coor[i])
        if len(point) == 0:
            point = np.array(p_point)
        result.append(point)
        edge_nums.append(edge_num)
    else:
        # 计算每个点的转移概率
        est_point = []
        est_edge = 0
        max_probability = 0
        # 67, 66, 65
        if i == 69:
            print(intersections)
        for k in intersections.keys():
            for point in intersections[k]:
                pre_point = result[-1]
                pre_edge = true_data[edge_nums[-1]]
                # 计算转移概率
                # transfer_probability = 1
                transfer_probability = get_transfer_probability(pre_point, pre_edge, point, true_data[k], dis=3)

                # 计算发射概率
                emit_probability = get_emission_probability(geometricCalculate.computeDistance(point, p_point))
                probability = transfer_probability * emit_probability
                # if i == 69:
                #     print(edge_nums[-1])
                #     print('k:', k)
                #     print('transfer_probability:', transfer_probability)
                #     print('emit_probability:', emit_probability)
                #     print('probability:', probability)
                if probability > max_probability:
                    est_point = point
                    est_edge = k
                    max_probability = probability
        if len(est_point) > 0:
            result.append(est_point)
            edge_nums.append(est_edge)

result = enhance_result(result, edge_nums)
img = plt.imread("data/1.jpg")
# plt.xlim(-850, 220)
# plt.ylim(-350, 130)
fig, ax = plt.subplots()
ax.imshow(img, extent=[-850, 220, -350, 130])
# plt.plot(*zip(*xy_coor), alpha=0.5, color='red', label='measurement value')
plt.scatter(*zip(*xy_coor), s=5, color='red')
# plt.plot(*zip(*result), color='blue', label="estimate value")
plt.scatter(*zip(*result), s=5)
# plt.plot(*zip(*true_data), alpha=0.5, color='green', label='true value')
plt.show()
