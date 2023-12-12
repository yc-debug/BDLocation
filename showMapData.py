import matplotlib.pyplot as plt
import generatePointSet


mapData = generatePointSet.generateEdgeSet()
img = plt.imread("data/1.jpg")
fig, ax = plt.subplots()
ax.imshow(img, extent=[-850, 220, -350, 130])
for i in range(len(mapData)):
    x1 = mapData[i][0][0]
    x2 = mapData[i][1][0]
    y1 = mapData[i][0][1]
    y2 = mapData[i][1][1]
    plt.plot([x1, x2], [y1, y2])
# 显示图形
plt.show()