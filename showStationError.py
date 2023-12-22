"""
@Project ：BDLocation 
@File    ：showStationError.py
@IDE     ：PyCharm 
@Author  ：姚聪
@Date    ：2023/12/22 14:16 
"""

import generatePointSet
import matplotlib.pyplot as plt

file_name = 'data/stations_error.csv'
file_name_cld = 'data/stations_error_cld.csv'
station_id, error = generatePointSet.readStationsError(file_name)
station_id_cld, error_cld = generatePointSet.readStationsError(file_name_cld)
plt.plot(station_id, error, label='no_cld', color='red')
plt.plot(station_id_cld, error_cld, label='cld', color='blue')
# 添加图例，显示在右上角
plt.legend(loc='upper right')

# 添加标题和轴标签
plt.title('Error of each station')
plt.show()