"""
@Project ：BDLocation 
@File    ：gatherMapData.py
@IDE     ：PyCharm 
@Author  ：姚聪
@Date    ：2023/12/8 12:33 
"""

import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import geometricCalculate

file_name = 'data/mapData.csv'


def get_old_path(pre_data):
    res = []
    for index, row in pre_data.iterrows():
        res.append([row['Start_x'], row['Start_y']])
    return res


# write_data = ['Lid', 'Start_x', 'Start_y', 'End_x', 'End_y']
#
# with open(file_name, 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(write_data)
old_data = pd.read_csv(file_name)
old_path = get_old_path(pre_data=old_data)
data = []
img = plt.imread("data/1.jpg")
fig, ax = plt.subplots()
ax.imshow(img, extent=[-850, 220, -350, 130])
plt.plot(*zip(*old_path), alpha=0.5, color='blue', label="estimate value")
start = []
end = []
i = 0


def on_click(event):
    clicked = [float(event.xdata), float(event.ydata)]
    global start
    global end
    global i
    if len(start) == 0:
        start = clicked
        return
    end = clicked
    data.append([i, start[0], start[1], end[0], end[1]])
    i = i + 1
    # 写入数据
    if geometricCalculate.computeDistance(start, end) < 1:
        print("结束！！！！！！！！！")
        with open(file_name, 'a', newline='') as file1:
            writer1 = csv.writer(file1)
            writer1.writerows(data)
    print(start, ' ', end)
    start = end


fig.canvas.mpl_connect('button_press_event', on_click)

plt.show()
