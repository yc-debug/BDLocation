import pandas as pd


def generatePointSet():
    data = pd.read_csv('data/mapPoints.csv')
    pid = data['Pid']
    point_x = data['x']
    point_y = data['y']
    res = []
    for i in range(len(pid)):
        temp = [pid[i], [point_x[i], point_y[i]]]
        res.append(temp)
    return res


def generateEdgeSet():
    data = pd.read_csv('data/mapData.csv')
    start_point_x = data['Start_x']
    start_point_y = data['Start_y']
    end_point_x = data['End_x']
    end_point_y = data['End_y']
    res = []
    for i in range(0, len(start_point_x)):
        temp = [[start_point_x[i], start_point_y[i]], [end_point_x[i], end_point_y[i]]]
        res.append(temp)
    # res.append(res[0])

    return res


def generateEdgeSetWithLength():
    data = pd.read_csv('data/mapData1.csv')
    start_point_x = data['Start_x']
    start_point_y = data['Start_y']
    end_point_x = data['End_x']
    end_point_y = data['End_y']
    length_c = data['Length_c']
    length_t = data['Length_t']
    res = []
    for i in range(0, len(start_point_x)):
        temp = [[start_point_x[i], start_point_y[i]], [end_point_x[i], end_point_y[i]], length_c[i], length_t[i]]
        res.append(temp)

    return res


def generateStations():
    data = pd.read_csv('data/stations.csv')
    time = data['time_stamp']
    x = data['x']
    y = data['y']
    res = []
    for i in range(len(time)):
        res.append([time[i], [x[i], y[i]]])
    return res
