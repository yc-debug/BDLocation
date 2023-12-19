"""
@Project ：BDLocation 
@File    ：test1.py
@IDE     ：PyCharm 
@Author  ：姚聪
@Date    ：2023/12/10 15:29 
"""
import math
import numpy as np
import pandas as pd
import csv

# def distance(P, A, B):
#     x0, y0 = P
#     x1, y1 = A
#     x2, y2 = B
#     A = y2 - y1
#     B = x1 - x2
#     C = x2 * y1 - x1 * y2
#     d = abs(A * x0 + B * y0 + C) / (A ** 2 + B ** 2) ** 0.5
#     return d
#
#
# point = np.array([-100.11293459060334, -188.88982843899106])
# point1 = np.array([-100.3528226, -86.36454133])
# point2 = np.array([-100.3528226, -103.4428343])
# print(distance(point, point1, point2))


import torch

x1_t = torch.normal(2 * torch.ones(100, 2), 1)
y1_t = torch.zeros(100)

x2_t = torch.normal(-2 * torch.ones(100, 2), 1)
y2_t = torch.ones(100)

x_t = torch.cat((x1_t, x2_t), 0)
y_t = torch.cat((y1_t, y2_t), 0)
print(y_t)
