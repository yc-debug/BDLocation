"""
@Project ：BDLocation 
@File    ：getLengthEachEdge.py
@IDE     ：PyCharm 
@Author  ：姚聪
@Date    ：2023/12/12 10:11 
"""
import pandas as pd
import numpy as np
import csv
import geometricCalculate
import getScale
import generatePointSet
from tqdm import tqdm, trange

file_name = 'data/mapData1.csv'
scale = getScale.getScale()

edge = generatePointSet.generateEdgeSet()
modified_data = [['Lid', 'Start_x', 'Start_y', 'End_x', 'End_y', 'Length_c', 'Length_t']]
i = 0
for i in tqdm(range(len(edge))):
    item = edge[i]
    dis = geometricCalculate.computeDistance(item[0], item[1])
    modified_data.append([i, item[0][0], item[0][1], item[1][0], item[1][1], dis, dis/scale])

with open(file_name, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(modified_data)
