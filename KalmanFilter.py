import numpy as np
from filterpy.kalman import KalmanFilter


def KalMan(xy_coor, size, Acceleration_X, Acceleration_Y):
    sigma_x2 = 2  # x方向上测量噪声的方差
    sigma_y2 = 2  # y方向上测量噪声的方差

    # 创建实例
    kf = KalmanFilter(dim_x=2, dim_z=2, dim_u=2)

    # 设置采样时间间隔，与前面 x_true=np.linspace 的采样间隔相等，即每隔0.1个单位采样一次
    t = 220 / size

    # 设置初始状态为 x方向位置，y方向位置，x方向加速度，y方向加速度
    kf.x = np.array([xy_coor[0][0], xy_coor[0][1]])
    # 设置状态转移矩阵
    kf.F = np.array([[1, 0], [0, 1]])
    # 设置状态向量到测量值的转换矩阵
    kf.H = np.array([[1, 0], [0, 1]])
    kf.B = np.array([[0.5 * t * t, 0], [0, 0.5 * t * t]])

    # 设置测量噪声的协方差矩阵 R，使用了前面人为制造噪声时的方差
    kf.R = np.array([sigma_x2, sigma_y2])
    # 设置先验误差的协方差矩阵 P，按默认的单位矩阵进行初始化
    kf.P = np.eye(2)

    # 记录估计过程
    x_pred = []
    y_pred = []
    for i in range(size):
        kf.predict(u=np.array([Acceleration_X[i], Acceleration_Y[i]]))
        kf.update(np.array([[xy_coor[i][0]], [xy_coor[i][1]]]))
        # 将预测结果保存
        x_pred.append(kf.x[0])
        y_pred.append(kf.x[1])

    return x_pred, y_pred
