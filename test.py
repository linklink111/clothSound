import numpy as np

# 声压和时间数组
result_pressure = [0.1, 0.5, 0.2, 0.5, 0.1, 0.6]
result_t = [1, 2, 3, 5, 6,7]
t = 4

# 在 t=4 处进行插值
interp_pressure = np.interp(t, result_t, result_pressure)

print("在 t=4 处的插值声压:", interp_pressure)