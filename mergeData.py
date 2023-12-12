import pandas as pd
import csv
from tqdm import tqdm, trange

print("载入数据中。。。")
# gps_path = 'D:\\Desktop\\研究生毕设\\2023_10_17_11_58_07\\all_gps.csv'
gps_path = 'D:\\Desktop\\研究生毕设\\2023_11_20_12_16_52\\all_gps.csv'
gps = pd.read_csv(gps_path)

# accel_path = 'D:\\Desktop\\研究生毕设\\2023_10_17_11_58_07\\gyro_accel.csv'
accel_path = 'D:\\Desktop\\研究生毕设\\2023_11_20_12_16_52\\gyro_accel.csv'
accel = pd.read_csv(accel_path)
print("载入数据完成！！！")

# 创建csv文件
file_name = 'data/dataset3.csv'
# 写入表头
with open(file_name, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(
        ['Latitude', 'Longitude', 'Acceleration_X', 'Acceleration_Y', 'Acceleration_Z', 'Gyroscope_X',
         'Gyroscope_Y', 'Gyroscope_Z'])


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
                res.append(temp)
                break

    return res


print("生成新数据中。。。")
data = get_data()
# 写入数据
with open(file_name, 'a', newline='') as file:
    writer = csv.writer(file)
    for row in data:
        writer.writerow(row)
print("生成新数据完成！！！")
