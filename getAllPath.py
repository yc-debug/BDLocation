"""
@Project ：BDLocation 
@File    ：getAllPath.py
@IDE     ：PyCharm 
@Author  ：姚聪
@Date    ：2023/12/9 11:11 
"""
import numpy as np
import generatePointSet
import geometricCalculate
import getScale
import matplotlib.pyplot as plt


def getData():
    # 获取点和边
    points = generatePointSet.generatePointSet()
    edges = generatePointSet.generateEdgeSetWithLength()
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


def findEdgeByPoints(points_to_edge, pid1, pid2):
    pid1_to_edge = points_to_edge[pid1]
    pid2_to_edge = points_to_edge[pid2]
    edge = set(pid1_to_edge) & set(pid2_to_edge)
    return list(edge)[0]


def findAllPath(points, points_to_edge, edges, graph, start, end, maxWight):
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
                edge_num = findEdgeByPoints(points_to_edge, pid, int(node))
                if node not in visited and node not in seen_path[pid]:
                    if total_distance + edges[edge_num][3] > maxWight:
                        continue
                    g = g + 1
                    stack.append(points[int(node)][1])
                    total_distance += edges[edge_num][3]
                    visited.add(node)
                    seen_path[pid].append(node)
                    if points[int(node)][1] == end:
                        path.append(list(stack))
                        old_pop = stack.pop()
                        o_pid1 = getPid(points, old_pop)
                        if len(stack) == 0:
                            total_distance = 0
                        else:
                            o_pid2 = getPid(points, stack[-1])
                            total_distance -= edges[findEdgeByPoints(points_to_edge, o_pid1, o_pid2)][3]
                        visited.remove(o_pid1)
                    break
        if g == 0:
            old_pop = stack.pop()
            o_pid1 = getPid(points, old_pop)
            if len(stack) == 0:
                total_distance = 0
            else:
                o_pid2 = getPid(points, stack[-1])
                total_distance -= edges[findEdgeByPoints(points_to_edge, o_pid1, o_pid2)][3]
            del seen_path[o_pid1]
            visited.remove(o_pid1)
    return path


def getAllPath(start, end, maxWight):
    points, edges, adjacency_matrix = getData()
    graphMatrix = {}
    points_to_edge = {}
    for i in range(len(points)):
        point1 = points[i][1]
        adjPoint = []
        for j in range(len(adjacency_matrix[i])):
            if adjacency_matrix[i][j] == 1:
                adjPoint.append(points[j][1])
        graphMatrix[i] = adjPoint
    for i in range(len(points)):
        point = points[i][1]
        pid = points[i][0]
        points_to_edge[pid] = []
        for j in range(len(edges)):
            if edges[j][0] == point or edges[j][1] == point:
                points_to_edge[pid].append(j)
    paths = findAllPath(points=points, points_to_edge=points_to_edge, edges=edges, graph=adjacency_matrix, start=start,
                        end=end, maxWight=maxWight)
    return paths


def get_path_length(path):
    length = 0
    for i in range(1, len(path)):
        point1 = path[i-1]
        point2 = path[i]
        length += geometricCalculate.computeDistance(point1, point2)
    return length/getScale.getScale()


def test(maxWight):
    points, edges, adjacency_matrix = getData()
    start = points[34][1]
    end = points[33][1]
    paths = getAllPath(start=start, end=end, maxWight=maxWight)
    print(paths)
    img = plt.imread("data/1.jpg")
    fig, ax = plt.subplots()
    ax.imshow(img, extent=[-850, 220, -350, 130])
    print(len(paths))
    for path in paths:
        plt.plot(*zip(*convertFormat(path)))
    plt.show()


# test(38)
