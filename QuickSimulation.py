import taichi as ti
import simpleaudio as sa
import numpy as np
import wavefile as wf
import math
ti.init(arch=ti.vulkan)  # Alternatively, ti.init(arch=ti.cpu)

n = 128
quad_size = 1.0 / n
dt = 4e-2 / n
substeps = int(1 / 60 // dt)

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
        sound_time[i] = sound_time[i] + distance_to_listener / sound_speed + dt
        omega = 2 * math.pi / wavelength_acc  # 假设波长已知，可根据实际情况调整
        A1 = acc[i]-last_acc[i]  # the amplitude of moving noise
        # A2 = v_curr   # the amplitude of friction noise

        phi_x = 0.0  # 可根据实际情况调整相位
        phi_y = 0.0  # 可根据实际情况调整相位
        phi_z = 0.0  # 可根据实际情况调整相位

        # 计算三维正弦波的声压
        px = A1.x * ti.sin(x[i].x - omega * sound_time[i] + phi_x)
        py = A1.y * ti.sin(x[i].y - omega * sound_time[i] + phi_y)
        pz = A1.z * ti.sin(x[i].z - omega * sound_time[i] + phi_z)
        wave_effect = ti.Vector([px, py, pz])
        sound_pressure[i] = (x[i] - listener).dot(wave_effect)
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



window = ti.ui.Window("Taichi Cloth Simulation on GGUI", (1024, 1024),
                      vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()

current_t = 0.0
initialize_mass_points()

# Initialize the audio buffer
sample_rate = 44100
channels = 1
audio_buffer = np.array([], dtype=float)

while window.running:
    if current_t > 1.5:
        # Reset
        initialize_mass_points()
        current_t = 0

    for i in range(substeps):
        substep()
        # Combine sound_time and sound_pressure arrays
        combined_data = np.column_stack((sound_time.to_numpy(), sound_pressure.to_numpy()))
        # Sort combined_data based on the first column (sound_time)
        sorted_combined_data = combined_data[combined_data[:, 0].argsort()]
        # Append sorted sound_pressure data to audio_buffer
        audio_buffer = np.append(audio_buffer, sorted_combined_data[:, 1])
        current_t += dt
        # Play the audio if the buffer reaches a certain size (e.g., every 0.1 seconds)
        if len(audio_buffer) >= sample_rate // 100:
            # audio_buffer = sliding_window_sum(audio_buffer,0.01)
            audio_buffer /= max(audio_buffer.min(), audio_buffer.max()) * 1.05
            # Convert audio_buffer to a two-dimensional array
            audio_buffer_2d = np.expand_dims(audio_buffer, axis=0)

            # Create a WaveFile object and write audio buffer to it
            audio_file = wf.WaveWriter("output.wav", channels=channels, samplerate=sample_rate)
            audio_file.write(audio_buffer_2d)
            
            # Play the audio using simpleaudio
            play_obj = sa.play_buffer(audio_buffer.tobytes(), channels, 2, sample_rate)
            
            # Clear the audio buffer
            audio_buffer = np.array([], dtype=float)
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
    window.show()