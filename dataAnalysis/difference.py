from readMdd import read_mdd_file
import numpy as np

# get velocity and accleration by differencing on vertex position.
# 怎么感觉有点像数据分析挖掘，算了不管了一切为了效果好。

def interpolate_x(x, dt0, dt):
    # 根据 dt 将 x 插值到一个可行的长度
    num_frames, num_particles, dim = x.shape
    target_num_frames = int(num_frames * dt0 / dt)  # 计算目标帧数
    interp_x = np.zeros((target_num_frames, num_particles, dim))  # 用以存储各个插值后的数据
    
    for particle in range(num_particles):
        interp_frame_x = np.interp(np.arange(0,num_frames,dt/dt0), np.arange(0, num_frames), x[:, particle, 0])
        interp_frame_y = np.interp(np.arange(0,num_frames,dt/dt0), np.arange(0, num_frames), x[:, particle, 1])
        interp_frame_z = np.interp(np.arange(0,num_frames,dt/dt0), np.arange(0, num_frames), x[:, particle, 2])
        
        interp_frame = np.column_stack((interp_frame_x, interp_frame_y, interp_frame_z))
        interp_x[:, particle] = interp_frame
        
    return interp_x


def get_v(x, dt):
    # 差分 x 求出 v
    num_frames, num_particles, dim = x.shape
    velocities = np.zeros((num_frames-1, num_particles, 3))
    
    for particle in range(num_particles):
        for coord in range(3):
            velocities[:, particle, coord] = np.diff(x[:, particle, coord]) / dt
    # Calculate the velocities for the last frame separately
    last_frame_velocity = velocities[num_frames-2, :, :]
    velocities = np.concatenate((velocities, last_frame_velocity[np.newaxis, :, :]))
    return velocities

def get_a(v, dt):
    num_frames, num_particles, dim = v.shape
    accelerations = np.zeros((num_frames-1, num_particles, 3))
    
    for particle in range(num_particles):
        for coord in range(3):
            accelerations[:, particle, coord] = np.diff(v[:, particle, coord]) / dt
    # Calculate the velocities for the last frame separately
    last_frame_acceleration = accelerations[num_frames-2, :, :]
    accelerations = np.concatenate((accelerations, last_frame_acceleration[np.newaxis, :, :]))
    return accelerations

def get_da(a, dt):
    num_frames, num_particles, dim = a.shape
    das = np.zeros((num_frames-1, num_particles, 3))
    
    for particle in range(num_particles):
        for coord in range(3):
            das[:, particle, coord] = np.diff(a[:, particle, coord]) / dt
    # Calculate the velocities for the last frame separately
    last_frame_da = das[num_frames-2, :, :]
    das = np.concatenate((das, last_frame_da[np.newaxis, :, :]))
    return das




