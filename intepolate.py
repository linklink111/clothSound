import numpy as np
from scipy.io import wavfile
from scipy.interpolate import interp1d

# 读取原始音频文件
original_sr, original_audio = wavfile.read('merged_output.wav')

# 新的采样率和插值函数
new_sr = 44100  # 新的采样率
interp_func = interp1d(np.arange(0, len(original_audio)), original_audio.T, kind='linear', axis=0)

# 计算新的音频长度
new_length = int(len(original_audio) * (new_sr / original_sr))

# 生成新的插值后的音频数据
new_audio = interp_func(np.linspace(0, len(original_audio) - 1, new_length)).astype(np.int16)

# 保存新的音频文件
wavfile.write('new_interpolated.wav', new_sr, new_audio)
