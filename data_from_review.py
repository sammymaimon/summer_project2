import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('data_from_review.txt')
data1 = np.genfromtxt('n10_j1001_runs240.dat')
data2 = np.genfromtxt('n15_j1001_runs240.dat')
data3 = np.genfromtxt('n20_j1001_runs240.dat')
data4 = np.genfromtxt('n25_j1001_runs240.dat')
data5 = np.genfromtxt('n30_j1001_runs240.dat')
data6 = np.genfromtxt('n35_j1001_runs240.dat')
data7 = np.genfromtxt('n40_j1001_runs240.dat')
data8 = np.genfromtxt('n45_j1001_runs240.dat')
data9 = np.genfromtxt('n50_j1001_runs240.dat')

x = data[:, 0]
y = data[:, 1]

x1 = data1[:, 0]  # 10 EPSILON =1
y1 = data1[:, 1]  # 10 EPSILON =1
x2 = data2[:, 0]  # 15 EPSILON =1
y2 = data2[:, 1]  # 15 EPSILON =1
x3 = data3[:, 0]  # 20 EPSILON =1
y3 = data3[:, 1]  # 20 EPSILON =1
x4 = data4[:, 0]  # 25 EPSILON =1
y4 = data4[:, 1]  # 25 EPSILON =1
x5 = data5[:, 0]  # 30 EPSILON =1
y5 = data5[:, 1]  # 30 EPSILON =1
x6 = data6[:, 0]  # 35 EPSILON =1
y6 = data6[:, 1]  # 35 EPSILON =1
x7 = data7[:, 0]  # 40 EPSILON =1
y7 = data7[:, 1]  # 40 EPSILON =1
x8 = data8[:, 0]  # 45 EPSILON =1
y8 = data8[:, 1]  # 45 EPSILON =1
x9 = data9[:, 0]  # 50 EPSILON =1
y9 = data9[:, 1]  # 50 EPSILON =1

plt.plot(x, y)
plt.plot(x1, y1, label='10')  # 10
plt.plot(x2, y2, label='15')  # 15
plt.plot(x3, y3, label='20')  # 20
plt.plot(x4, y4, label='25')  # 25
plt.plot(x5, y5, label='30')  # 30
plt.plot(x6, y6, label='35')  # 10
plt.plot(x7, y7, label='40')  # 15
plt.plot(x8, y8, label='45')  # 45
plt.plot(x9, y9, label='50')

plt.legend(loc="upper left")

plt.show()
