import taichi as ti
import os
import simpleaudio as sa
import numpy as np
import wavefile as wf
import soundfile as sf
from scipy.interpolate import interp1d
from scipy.signal import butter, lfilter
from scipy.io import wavfile
import wave
import time
import math
ti.init(arch=ti.vulkan)  # Alternatively, ti.init(arch=ti.cpu)

n = 128
quad_size = 1.0 / n
dt = 4e-2 / n
substeps = int(1 / 60 // dt)
current_t = 0.0

gravity = ti.Vector([0, -9.8, 0])
spring_Y = 3e4
dashpot_damping = 1e4
drag_damping = 1

ball_radius = 0.3
ball_center = ti.Vector.field(3, dtype=float, shape=(1, ))
ball_center[0] = [0, 0, 0]

x = ti.Vector.field(3, dtype=float, shape=(n, n))
v = ti.Vector.field(3, dtype=float, shape=(n, n))
acc = ti.Vector.field(3,dtype=float, shape=(n,n))
last_acc = ti.Vector.field(3,dtype=float, shape=(n,n))
last_v = ti.Vector.field(3, dtype=float,shape = (n,n))
sound_time = ti.field(dtype=float, shape=(n, n))
sound_pressure = ti.field(dtype=float, shape=(n, n))
listener = ti.Vector([0, 0, 0])

num_triangles = (n - 1) * (n - 1) * 2
indices = ti.field(int, shape=num_triangles * 3)
vertices = ti.Vector.field(3, dtype=float, shape=n * n)
colors = ti.Vector.field(3, dtype=float, shape=n * n)

bending_springs = False

@ti.kernel
def initialize_mass_points():
    random_offset = ti.Vector([ti.random() - 0.5, ti.random() - 0.5]) * 0.1

    for i, j in x:
        x[i, j] = [
            i * quad_size - 0.5 + random_offset[0], 0.6,
            j * quad_size - 0.5 + random_offset[1]
        ]
        v[i, j] = [0, 0, 0]


@ti.kernel
def initialize_mesh_indices():
    for i, j in ti.ndrange(n - 1, n - 1):
        quad_id = (i * (n - 1)) + j
        # 1st triangle of the square
        indices[quad_id * 6 + 0] = i * n + j
        indices[quad_id * 6 + 1] = (i + 1) * n + j
        indices[quad_id * 6 + 2] = i * n + (j + 1)
        # 2nd triangle of the square
        indices[quad_id * 6 + 3] = (i + 1) * n + j + 1
        indices[quad_id * 6 + 4] = i * n + (j + 1)
        indices[quad_id * 6 + 5] = (i + 1) * n + j

    for i, j in ti.ndrange(n, n):
        if (i // 4 + j // 4) % 2 == 0:
            colors[i * n + j] = (0.22, 0.72, 0.52)
        else:
            colors[i * n + j] = (1, 0.334, 0.52)

initialize_mesh_indices()

spring_offsets = []
if bending_springs:
    for i in range(-1, 2):
        for j in range(-1, 2):
            if (i, j) != (0, 0):
                spring_offsets.append(ti.Vector([i, j]))

else:
    for i in range(-2, 3):
        for j in range(-2, 3):
            if (i, j) != (0, 0) and abs(i) + abs(j) <= 2:
                spring_offsets.append(ti.Vector([i, j]))

@ti.kernel
def substep():
    for i in ti.grouped(x):
        last_v[i] = v[i]
        v[i] += gravity * dt

    for i in ti.grouped(x):
        last_acc[i] = acc[i]
        force = ti.Vector([0.0, 0.0, 0.0])
        for spring_offset in ti.static(spring_offsets):
            j = i + spring_offset
            if 0 <= j[0] < n and 0 <= j[1] < n:
                x_ij = x[i] - x[j]
                v_ij = v[i] - v[j]
                d = x_ij.normalized()
                current_dist = x_ij.norm()
                original_dist = quad_size * float(i - j).norm()
                # Spring force
                force += -spring_Y * d * (current_dist / original_dist - 1)
                # Dashpot damping
                force += -v_ij.dot(d) * d * dashpot_damping * quad_size
        acc[i] = force # mass is 1
        v[i] += force * dt

    for i in ti.grouped(x):
        v[i] *= ti.exp(-drag_damping * dt)
        offset_to_center = x[i] - ball_center[0]
        if offset_to_center.norm() <= ball_radius:
            # Velocity projection
            normal = offset_to_center.normalized()
            v[i] -= min(v[i].dot(normal), 0) * normal
        x[i] += dt * v[i]

    # generate sound
    for i in ti.grouped(x):
        wavelength_acc = 0.1 # wave length for acclerating noise
        distance_to_listener = (x[i] - listener).norm()
        sound_speed = 343.0  # 假设声速为343 m/s，可根据实际情况调整
        t_curr = distance_to_listener / sound_speed + current_t
        omega = 2 * math.pi / wavelength_acc  # 假设波长已知，可根据实际情况调整
        # A1 = (acc[i]-last_acc[i])/dt  # the amplitude of moving noise
        A1 = (v[i]-last_v[i])/dt  # the amplitude of moving noise
        # A2 = v_curr   # the amplitude of friction noise

        phi_x = 0.0  # 可根据实际情况调整相位
        phi_y = 0.0  # 可根据实际情况调整相位
        phi_z = 0.0  # 可根据实际情况调整相位

        # 计算三维正弦波的声压
         # 计算三维正弦波的声压
        px = A1.x * ti.sin(x[i].x - omega * t_curr + phi_x)
        py = A1.y * ti.sin(x[i].y - omega * t_curr + phi_y)
        pz = A1.z * ti.sin(x[i].z - omega * t_curr + phi_z)
        wave_effect = ti.Vector([px, py, pz])
        sound_pressure[i] = (x[i] - listener).dot(wave_effect)
        sound_time[i] = t_curr
        pass

@ti.kernel
def update_vertices():
    for i, j in ti.ndrange(n, n):
        vertices[i * n + j] = x[i, j]

def sliding_window_sum(data, window_size):
    window_samples = int(window_size * len(data))
    merged_data = []

    for i in range(0, len(data), window_samples):
        start_idx = max(0, i)
        end_idx = min(len(data), i + window_samples)
        merged_data.append(np.sum(data[start_idx:end_idx]))

    return np.array(merged_data)

def interpolate_audio(sound_time, sound_pressure, target_time):
    interp_func = interp1d(sound_time, sound_pressure, kind='linear', fill_value="extrapolate")
    interpolated_pressure = interp_func(target_time)
    return interpolated_pressure
def merge_t(sorted_t, sorted_pressure, interval):
    merged_t = [sorted_t[0]]
    merged_pressure = [sorted_pressure[0]]
    accumulated_pressure = sorted_pressure[0]

    for i in range(1, len(sorted_t)):
        time_diff = sorted_t[i] - sorted_t[i - 1]

        if time_diff <= interval:
            accumulated_pressure += sorted_pressure[i]
        else:
            merged_t.append(sorted_t[i])
            merged_pressure.append(accumulated_pressure)
            accumulated_pressure = sorted_pressure[i]
    print(len(merged_t)) #这里输出的值大概是200000就比较好
    return np.array(merged_t), np.array(merged_pressure)
def load_array(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    float_list = [float(x.strip()) for x in lines]
    return np.array(float_list)

def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y
def butter_highpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y
def make_audio(fname, start, end, target_t=-1, high_cut=800, low_cut=-1, scale=1.0):
    all_audio = np.array([])
    sample_rate = 44100

    all_pressure = np.array([])
    all_t = np.array([])
    for i in range(start, end):
        #        pressure = load_array(f"sheetResearch/soundData/pressure_frame{i}.txt")
        #        t = load_array(f"sheetResearch/soundData/t_frame{i}.txt")
        pressure = load_array(f"pressures/pressure_frame{i}.txt")
        t = load_array(f"ts/t_frame{i}.txt")
        all_pressure = np.concatenate((all_pressure, pressure))
        all_t = np.concatenate((all_t, t))

    sort_index = np.argsort(all_t)
    sorted_all_pressure = all_pressure[sort_index]
    sorted_all_t = all_t[sort_index]

    #eps = 0.0001

    result_pressure = sorted_all_pressure
    result_t = sorted_all_t

    eps = 0.00001  # 设置合并时间间隔
    # result_t, result_pressure = merge_t(sorted_all_t, sorted_all_pressure, eps)

    #result_pressure, result_t = merge_t(sorted_all_t,sorted_all_pressure,eps)
    if target_t > 0:
        scale = result_t / target_t
    result_t /= scale

    audio = np.interp(
        np.arange(0, result_t[-1], 1/sample_rate), result_t, result_pressure)
    audio = audio / np.max(np.abs(audio))
    # Apply the low-pass filter
    cutoff = high_cut  # The cutoff frequency
    order = 6  # The order of the filter
    if high_cut > 0:
        audio = butter_lowpass_filter(audio, cutoff, sample_rate, order)
    #original_audio = audio
    #new_audio = original_audio[::3]

    # Apply the high-pass filter
    cutoff = low_cut  # The cutoff frequency
    order = 6  # The order of the filter
    if low_cut > 0:
        audio = butter_highpass_filter(audio, cutoff, sample_rate, order)

    wavfile.write(f"output_audio/{fname}.wav", sample_rate,
          (audio * (2**15-1)).astype(np.int16))
window = ti.ui.Window("Taichi Cloth Simulation on GGUI", (1024, 1024),
                      vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()


initialize_mass_points()

# Initialize the audio buffer
sample_rate = 44100
channels = 1
audio_buffer = np.array([], dtype=float)
total_steps = 0
sub_steps_cnt = 0
restore_steps = 0

pressure_cache = np.array([])
t_cache = np.array([])

while window.running:
    time.sleep(1./60)
    # if current_t > 1.5:
    #     # Reset
    #     initialize_mass_points()
    #     current_t = 0
    print(current_t)
    
    if total_steps > 5000:
        break
    sub_steps_cnt+=1
    for i in range(substeps):
        substep()
        total_steps += 1
        current_t+=dt

    update_vertices()

    camera.position(0.0, 0.0, 3)
    camera.lookat(0.0, 0.0, 0)
    scene.set_camera(camera)

    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.mesh(vertices,
               indices=indices,
               per_vertex_color=colors,
               two_sided=True)

    # Draw a smaller ball to avoid visual penetration
    scene.particles(ball_center, radius=ball_radius * 0.95, color=(0.5, 0.42, 0.8))
    canvas.scene(scene)
    window.save_image(f'justVisual_frame/{sub_steps_cnt:04d}.png')
    window.show()