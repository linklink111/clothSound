from readMdd import read_mdd_file
import numpy as np

# get velocity and accleration by differencing on vertex position.
# 怎么感觉有点像数据分析挖掘，算了不管了一切为了效果好。

def get_v(vertices, target_frame):
    num_frames = len(vertices)
    if num_frames < 2:
        return None

    if target_frame <= 0:
        return np.zeros(vertices.shape[1])  # 初始帧速度设为零

    if target_frame >= num_frames:
        return np.zeros(vertices.shape[1])  # 最后一帧速度设为零

    prev_frame = vertices[target_frame - 1]
    current_frame = vertices[target_frame]

    time_diff = 1.0  # 假设每帧时间间隔为1秒

    velocity = (current_frame - prev_frame) / time_diff
    return velocity

def get_a(velocity):
    num_frames = velocity.shape[0]

    acceleration = np.diff(velocity, axis=0)  # 沿帧轴进行差分计算加速度
    return acceleration

if __name__ == "__main__":
    # 指定 MDD 文件
    mdd_file_path = "shakeCloth.mdd"

    # 计算速差
    num_frames, num_verts, frame_times, vertices = read_mdd_file(mdd_file_path)
    target_frame = 2
    velocity = get_v(vertices, target_frame)
    print("Velocity at frame", target_frame, ":", velocity)

    acceleration = get_a(velocity)
    print("Acceleration at frame", target_frame, ":", acceleration)

