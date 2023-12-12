import numpy as np
import math


def get_linear_equation(point_1, point_2):
    if (point_1 == point_2).all():
        return "两点相同"
    x1 = point_1[0]
    y1 = point_1[1]
    x2 = point_2[0]
    y2 = point_2[1]

    A = y2 - y1
    B = x1 - x2
    C = x1 * (y1 - y2) + y1 * (x2 - x1)

    return [A, B, C]


def get_distance(point, e):
    A = e[0]
    B = e[1]
    C = e[2]
    fm = np.sqrt(A * A + B * B)
    fz = A * point[0] + B * point[1] + C

    if fz == 0:
        return "点在直线上"
    else:
        return math.fabs(fz / fm)


def computeIntersection(start, end, circle_center, radius):
    param = get_linear_equation(start, end)
    dis = get_distance(circle_center, param)
    # mapping_point = [0, 0]
    mapping_point = compute_mapping_point(start, end, circle_center)
    # if len(mapping_point) == 0:
    #     return np.array([])
    # 没有交点
    if dis > radius:
        return np.array(mapping_point).reshape(-1, 2)
    elif len(mapping_point) == 0:
        return np.array([])

    x0, y0 = circle_center
    x1, y1 = start
    x2, y2 = end
    if radius == 0:
        return np.array([x1, y1])
    # 斜率不存在的情况
    if x1 == x2:
        inp = []
        if abs(radius) >= abs(x1 - x0):
            # 下方这个点
            p1 = x1, round(y0 - (radius ** 2 - (x1 - x0) ** 2) ** 0.5, 5)
            # 上方这个点
            p2 = x1, round(y0 + (radius ** 2 - (x1 - x0) ** 2) ** 0.5, 5)
            if max(y1, y2) >= p2[1]:
                inp.append(p2)
            if min(y1, y2) <= p1[1]:
                inp.append(p1)
    else:
        # 求直线y=kx+b的斜率及b
        k = (y1 - y2) / (x1 - x2)
        b0 = y1 - k * x1
        # 直线与圆的方程化简为一元二次方程ax**2+bx+c=0
        a = k ** 2 + 1
        b = 2 * k * (b0 - y0) - 2 * x0
        c = (b0 - y0) ** 2 + x0 ** 2 - radius ** 2
        # 判别式判断解，初中知识
        delta = b ** 2 - 4 * a * c
        if delta >= 0:
            p1x = round((-b - delta ** (0.5)) / (2 * a), 5)
            p2x = round((-b + delta ** (0.5)) / (2 * a), 5)
            p1y = round(k * p1x + b0, 5)
            p2y = round(k * p2x + b0, 5)
            inp = [[p1x, p1y], [p2x, p2y]]
            inp = [p for p in inp if min(x1, x2) <= p[0] <= max(x1, x2)]
        else:
            inp = []

    if inp:
        return np.array(inp).reshape(-1, 2)
    else:
        return np.array(mapping_point).reshape(-1, 2)


def compute_mapping_point(start, end, circle_center):
    x1 = start[0]
    y1 = start[1]
    x2 = end[0]
    y2 = end[1]
    xp = circle_center[0]
    yp = circle_center[1]
    if x1 == x2:
        # The line is vertical, handle this case separately
        xm = x1  # x-coordinate of the vertical line is constant
        ym = yp  # y-coordinate of the mapping point is same as the input point
    else:
        # The line is not vertical, calculate the equation of the line
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2

        # Calculate the coordinates of the mapping point
        xm = (B * B * xp - A * B * yp - A * C) / (A * A + B * B)
        ym = (-A * B * xp + A * A * yp - B * C) / (A * A + B * B)

    if min(x1, x2) <= xm <= max(x1, x2) and min(y1, y2) <= ym <= max(y1, y2):
        return np.array([xm, ym])
    else:
        return np.array([])


def computeDistance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5


# point = np.array([-100.11293459060334, -188.88982843899106])
# point1 = np.array([-100.3528226, -86.36454133])
# point2 = np.array([-100.3528226, -103.4428343])
# res = computeIntersection(point1, point2, point, 4.07)
# print(computeDistance(res[0], point))
# print(res)
