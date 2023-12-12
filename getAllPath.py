"""
@Project ：BDLocation 
@File    ：getAllPath.py
@IDE     ：PyCharm 
@Author  ：姚聪
@Date    ：2023/12/9 11:11 
"""
import numpy as np
import generatePointSet
import matplotlib.pyplot as plt


def getData():
    # 获取点和边
    points = generatePointSet.generatePointSet()
    edges = generatePointSet.generateEdgeSet()
    adjacency_matrix = np.loadtxt('data/adjacencyMatrix.txt')
    return points, edges, adjacency_matrix


def getPid(points, point):
    for item in points:
        if item[1] == point:
            return item[0]


def convertFormat(data):
    res = []
    for item in data:
        res.append(np.array(item))
    return res


def findAllPath(points, graph, start, end, maxWight):
    path = []
    stack = [start]
    visited = set()
    visited.add(getPid(points, start))
    seen_path = {}
    total_distance = 0
    while len(stack) > 0:
        start = stack[-1]
        pid = getPid(points, start)
        nodes = graph[pid]
        if pid not in seen_path.keys():
            seen_path[pid] = []
        g = 0
        for i in range(len(nodes)):
            if nodes[i] == 1:
                node = i
                if node not in visited and node not in seen_path[pid]:
                    if total_distance + 1 > maxWight:
                        continue
                    g = g + 1
                    stack.append(points[int(node)][1])
                    total_distance += 1
                    visited.add(node)
                    seen_path[pid].append(node)
                    if points[int(node)][1] == end:
                        path.append(list(stack))
                        old_pop = stack.pop()
                        total_distance -= 1
                        visited.remove(getPid(points, old_pop))
                    break
        if g == 0:
            old_pop = stack.pop()
            total_distance -= 1
            pid = getPid(points, old_pop)
            del seen_path[pid]
            visited.remove(pid)
    return path


def getAllPath(start, end, maxWight):
    points, edges, adjacency_matrix = getData()
    graphMatrix = {}
    for i in range(len(points)):
        point1 = points[i][1]
        adjPoint = []
        for j in range(len(adjacency_matrix[i])):
            if adjacency_matrix[i][j] == 1:
                adjPoint.append(points[j][1])
        graphMatrix[i] = adjPoint
    paths = findAllPath(points=points, graph=adjacency_matrix, start=start, end=end, maxWight=maxWight)
    return paths


def test(maxWight):
    points, edges, adjacency_matrix = getData()
    start = points[0][1]
    end = points[1][1]
    paths = findAllPath(points=points, graph=adjacency_matrix, start=start, end=end, maxWight=maxWight)
    print(paths)
    img = plt.imread("data/1.jpg")
    fig, ax = plt.subplots()
    ax.imshow(img, extent=[-850, 220, -350, 130])
    print(len(paths))
    for path in paths:
        plt.plot(*zip(*convertFormat(path)))
    plt.show()

#
# test(3)
