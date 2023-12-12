"""
@Project ：BDLocation
@File    ：genetateAdjacency.py
@IDE     ：PyCharm
@Author  ：姚聪
@Date    ：2023/12/8 12:26
"""

import numpy as np
import generatePointSet

# 获取点和边
points = generatePointSet.generatePointSet()
edges = generatePointSet.generateEdgeSet()

size = len(points)
adjacencyMatrix = np.zeros((size, size))


def getEdgesByStartPoint(point):
    res = []
    for item in edges:
        if item[0] == point:
            res.append(item)
    return res


def getEdgesByEndPoint(point):
    res = []
    for item in edges:
        if item[1] == point:
            res.append(item)
    return res


for i in range(size):
    point1 = points[i][1]
    possibleEdges = getEdgesByStartPoint(point1)
    for j in range(size):
        if i == j:
            continue
        else:
            point2 = points[j][1]
            for tempEdge in possibleEdges:
                if point2 == tempEdge[1]:
                    adjacencyMatrix[i][j] = 1
    possibleEdges = getEdgesByEndPoint(point1)
    for j in range(size):
        if i == j:
            continue
        else:
            point2 = points[j][1]
            for tempEdge in possibleEdges:
                if point2 == tempEdge[0]:
                    adjacencyMatrix[i][j] = 1

np.savetxt('data/adjacencyMatrix.txt', adjacencyMatrix)
print(np.loadtxt('data/adjacencyMatrix.txt'))