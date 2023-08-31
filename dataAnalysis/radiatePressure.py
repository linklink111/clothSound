import numpy as np
import math
from difference import get_v, get_a
from readMdd import read_mdd_file

# accleration to sound
# 根据加速度应用诺伊曼边界条件产生声压
# 1. 可能不太正确
# 2. 直接对整个加速度这么做可能有点粗糙（但是似乎也可以，因为布料声音就是比较随机的...）

def radiate(acceleration, last_acceleration, velocities, last_velocities, positions, listener_position, current_time, dt):
    wavelength_acc = 0.1
    c = 343.0
    omega = 2 * math.pi / wavelength_acc
    listener = np.array(listener_position)
    num_particles = acceleration.shape[0]
    
    sound_pressure = np.zeros(num_particles)
    sound_time = np.zeros(num_particles)
    
    for i in range(num_particles):
        distance_to_listener = np.linalg.norm(positions[i] - listener)
        t_curr = distance_to_listener / c + current_time
        
        A1 = (acceleration[i] - last_acceleration[i]) / dt
        phi_x, phi_y, phi_z = 0.0, 0.0, 0.0  # 可根据实际情况调整相位
        
        px = A1[0] * np.sin(positions[i] - omega * t_curr + phi_x)
        py = A1[1] * np.sin(positions[i] - omega * t_curr + phi_y)
        pz = A1[2] * np.sin(positions[i] - omega * t_curr + phi_z)
        wave_effect = np.array([px, py, pz])
        
        normal_vec = positions[i] - listener
        sound_pressure[i] = np.dot(normal_vec / np.linalg.norm(normal_vec), wave_effect) / (4 * math.pi * c * np.linalg.norm(normal_vec))
        sound_time[i] = t_curr
    
    return sound_pressure, sound_time

if __name__ == "__main__":
    # 指定 MDD 文件
    mdd_file_path = "shakeCloth.mdd"
    listener_position = [0.0, 0.0, 0.0]
    
    # 计算速差
    num_frames, num_verts, frame_times, vertices = read_mdd_file(mdd_file_path)
       
    velocity_data = get_v(vertices)  # 获取速度数据
    acceleration_data = get_a(velocity_data)  # 获取加速度数据 
    
    dt = 0.0001  # 假设帧之间的时间间隔是固定的
    
    num_particles = position_data.shape[0]
    
    for frame in range(num_frames):
        acceleration = acceleration_data[frame]
        last_acceleration = acceleration_data[frame - 1] if frame > 0 else np.zeros_like(acceleration)
        velocities = velocity_data[frame]
        last_velocities = velocity_data[frame - 1] if frame > 0 else np.zeros_like(velocities)
        positions = position_data
        
        sound_pressure, sound_time = radiate(acceleration, last_acceleration, velocities, last_velocities, positions, listener_position, frame * dt, dt)
        
        print("Sound Pressure at frame", frame, ":", sound_pressure[:10])
        print("Sound Time at frame", frame, ":", sound_time[:10])