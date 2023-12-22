import pandas as pd
import csv
import numpy as np
from tqdm import tqdm, trange

print("载入数据中。。。")
# gps_path = 'D:\\Desktop\\研究生毕设\\2023_10_17_11_58_07\\all_gps.csv'
gps_path = 'D:\\Desktop\\研究生毕设\\2023_11_20_12_16_52\\all_gps.csv'
gps = pd.read_csv(gps_path)

# accel_path = 'D:\\Desktop\\研究生毕设\\2023_10_17_11_58_07\\gyro_accel.csv'
accel_path = 'D:\\Desktop\\研究生毕设\\2023_11_20_12_16_52\\gyro_accel.csv'
accel = pd.read_csv(accel_path)
print("载入数据完成！！！")

accel_path_1 = 'D:\\Desktop\\研究生毕设\\2023_11_20_12_16_52\\gyro_accel_1.csv'
accel_1 = pd.read_csv(accel_path_1)

s = accel_1['s']


# dataset和dataset1：赵睿采集的数据集
# dataset2：size515，不带加速度index，匹配度间隔为100
# dataset3：size1363，不带加速度index，匹配度间隔为100
# dataset4：size1363，带加速度index,匹配度间隔为5
# dataset4_15：size1363，带加速度index，匹配度间隔为15
# dataset5：size515，带加速度index，匹配度间隔为10


# 解析数据
def get_data():
    res = []
    # 解析GPS数据
    for index, row in tqdm(gps.iterrows(), total=len(gps)):
        temp = []
        for index1, row1 in accel.iterrows():
            if abs(row['Unix time[nanosec]'] - row1['Unix time[nanosec]'] / 1000000) < 100:
                temp.append(row['lat[deg]'])
                temp.append(row['lon[deg]'])
                temp.append(row1['ax[m/s^2]'])
                temp.append(row1['ay[m/s^2]'])
                temp.append(row1['az[m/s^2]'])
                temp.append(row1['gx[rad/s]'])
                temp.append(row1['gy[rad/s]'])
                temp.append(row1['gz[rad/s]'])
                temp.append(index1)
                temp.append(s[index1])
                res.append(temp)
                break

    return res


# 解析数据
def get_data_new(dis):
    res = []
    visited = np.zeros(len(accel))
    # 解析GPS数据
    now_index = 0
    for index, row in tqdm(gps.iterrows(), total=len(gps)):
        temp = []
        for index1, row1 in accel.iterrows():
            if index1 >= now_index and abs(row['Unix time[nanosec]'] - row1['Unix time[nanosec]'] / 1000000) < dis and \
                    visited[index1] == 0:
                visited[index1] = 1
                now_index = index1
                temp.append(row1['Unix time[nanosec]'])
                temp.append(row['lat[deg]'])
                temp.append(row['lon[deg]'])
                temp.append(row1['ax[m/s^2]'])
                temp.append(row1['ay[m/s^2]'])
                temp.append(row1['az[m/s^2]'])
                temp.append(row1['gx[rad/s]'])
                temp.append(row1['gy[rad/s]'])
                temp.append(row1['gz[rad/s]'])
                temp.append(index1)
                temp.append(s[index1])
                res.append(temp)
                break

    return res


# 解析数据
def get_data_by_gps(dis):
    res = []
    # 解析GPS数据
    for index, row in tqdm(accel.iterrows(), total=len(accel)):
        for index1, row1 in gps.iterrows():
            t = abs(row1['Unix time[nanosec]'] - row['Unix time[nanosec]'] / 1000000)
            if 0 <= t <= dis:
                temp = [row['Unix time[nanosec]'] / 1000000,
                        row1['lat[deg]'], row1['lon[deg]'],
                        row['ax[m/s^2]'],
                        row['ay[m/s^2]'],
                        row['az[m/s^2]'],
                        row['gx[rad/s]'],
                        row['gy[rad/s]'],
                        row['gz[rad/s]'],
                        index,
                        s[index]]
                res.append(temp)

    return res


def merge_data(file_name, dis):
    print(file_name, " 生成新数据中。。。")
    data = get_data_new(dis)
    # 写入表头
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ['Time_Stamp', 'Latitude', 'Longitude', 'Acceleration_X', 'Acceleration_Y', 'Acceleration_Z', 'Gyroscope_X',
             'Gyroscope_Y', 'Gyroscope_Z', 'Index', 'Distance'])
    # 写入数据
    with open(file_name, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
    print(file_name, " 生成新数据完成！！！")


# 间隔15
# merge_data('data/dataset4_15.csv', 15)
# 间隔10
# merge_data('data/dataset4_20.csv', 20)
# merge_data('data/dataset4_30.csv', 30)
# merge_data('data/dataset4_40.csv', 40)
merge_data('data/dataset4_200.csv', 200)
# merge_data('data/dataset6/dataset6_10.csv', 10)
# merge_data('data/dataset6/dataset6_20.csv', 20)
# merge_data('data/dataset6/dataset6_30.csv', 30)
# merge_data('data/dataset6/dataset6_40.csv', 40)
# merge_data('data/dataset6/dataset6_50.csv', 50)
