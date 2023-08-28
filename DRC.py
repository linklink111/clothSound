from pydub import AudioSegment
from pydub.playback import play

# 读取原始音频文件
audio = AudioSegment.from_file("merged_output.wav")

# 设置压缩的阈值和比率
threshold_db = -10  # 阈值，单位：分贝
compression_ratio = 4.0  # 压缩比率

# 计算压缩的增益
threshold_amp = 10 ** (threshold_db / 20)  # 转换为线性振幅
compression_gain = 1 / compression_ratio

# 对音频进行动态范围压缩
compressed_samples = []
for sample in audio:
    compressed_sample = [s * compression_gain if abs(s) > threshold_amp else s for s in sample]
    compressed_samples.append(compressed_sample)

# 创建压缩后的音频对象
compressed_audio = AudioSegment(
    samples=compressed_samples,
    frame_rate=audio.frame_rate,
    sample_width=audio.sample_width,
    channels=audio.channels
)

# 保存压缩后的音频
compressed_audio.export("compressed_audio.wav", format="wav")

# 播放压缩后的音频
play(compressed_audio)

