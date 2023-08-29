import csv
import matplotlib.pyplot as plt

# 读取CSV文件中的数据
data = []
with open('data.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader)  # 跳过表头
    for row in csvreader:
        t = float(row[0])
        pressure = float(row[1])
        data.append((t, pressure))

# 提取时间和压力数据
time_values = [t for t, _ in data]
pressure_values = [pressure for _, pressure in data]

# 绘制线图
plt.plot(time_values, pressure_values, marker='o')
plt.xlabel('Time')
plt.ylabel('Pressure')
plt.title('Pressure-Time Line Graph')
plt.grid(True)
plt.show()
