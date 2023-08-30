import numpy as np
import math
from difference import get_v, get_a

# accleration to sound
# 根据加速度应用诺伊曼边界条件产生声压
# 1. 可能不太正确
# 2. 直接对整个加速度这么做可能有点粗糙（但是似乎也可以，因为布料声音就是比较随机的...）

