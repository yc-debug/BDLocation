"""
@Project ：BDLocation 
@File    ：preprocessedDataset.py
@IDE     ：PyCharm 
@Author  ：姚聪
@Date    ：2023/12/21 9:22 
"""
import csv
import pandas as pd


def compute_pre(arrays):
    length = len(arrays)
    array_sum = 0
    for item in arrays:
        array_sum += item
    return array_sum / length


def clean_data(file_name, new_file_name):
    data = pd.read_csv(file_name)
    time_stamp = data['Time_Stamp']
    latitude = data['Latitude']
    longitude = data['Longitude']
    i = 0
    res = []
    for index, row in data.iterrows():
        if index < i or index == len(data)-1:
            continue
        temp_time = row['Time_Stamp']
        i = index + 1
        temp_lat = [row['Latitude']]
        temp_lon = [row['Longitude']]
        while time_stamp[i] == temp_time:
            temp_lat.append(latitude[i])
            temp_lon.append(longitude[i])
            i += 1
        pre_lat = compute_pre(temp_lat)
        pre_lon = compute_pre(temp_lon)
        res.append([temp_time, pre_lat, pre_lon, row['Acceleration_X'], row['Acceleration_Y'], row['Acceleration_Z'],
                    row['Gyroscope_X'], row['Gyroscope_Y'], row['Gyroscope_Z'], row['Index'], row['Distance']])

    with open(new_file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Time_Stamp', 'Latitude', 'Longitude', 'Acceleration_X', 'Acceleration_Y', 'Acceleration_Z',
                         'Gyroscope_X', 'Gyroscope_Y', 'Gyroscope_Z', 'Index', 'Distance'])
    with open(new_file_name, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(res)


clean_data('data/dataset6/dataset6_10.csv', 'data/dataset6/dataset6_10_clean.csv')
clean_data('data/dataset6/dataset6_20.csv', 'data/dataset6/dataset6_20_clean.csv')
clean_data('data/dataset6/dataset6_30.csv', 'data/dataset6/dataset6_30_clean.csv')
clean_data('data/dataset6/dataset6_40.csv', 'data/dataset6/dataset6_40_clean.csv')
clean_data('data/dataset6/dataset6_50.csv', 'data/dataset6/dataset6_50_clean.csv')