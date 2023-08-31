from readMdd import read_mdd_file
from difference import interpolate_x, get_v, get_a
import matplotlib.pyplot as plt
import numpy as np


def write_vertices_to_file(file_path, vertices):
    with open(file_path, "w") as output_file:
        num_frames, num_verts, _ = vertices.shape

        for vert_index in range(num_verts):
            output_file.write(f"Vertex {vert_index + 1}:\n")
            for frame_index in range(num_frames):
                x, y, z = vertices[frame_index, vert_index]
                output_file.write(f"Frame {frame_index + 1}: x={x:>10.4f}, y={y:>10.4f}, z={z:>10.4f}\n")
            output_file.write("\n")


if __name__ == "__main__":
    # 指定 MDD 文件
    mdd_file_path = "shakeCloth.mdd"

    # 指定npy文件
    npy_file_path = "shakeCloth.npy"
    vertices = np.load(npy_file_path)
    num_frames ,num_verts,dim = vertices.shape

    mdd_output_file_path = "output_vertices.txt"
    write_vertices_to_file(mdd_output_file_path, vertices)
    
    dt0 = 0.1
    dt1 = 0.001 #  但是好像dt0和dt1差距可能会有点大...
    # vertices /= 1e38 # 有时候由于建模或单位的原因x可能有点大，这里换算一下（blender中的单位是cm）
    x = interpolate_x(vertices, dt0, dt1)
    # x /= 1000000 
    v = get_v(x, dt1)
    a = get_a(v, dt1) 
    with open("v.txt", "w") as f:
        for j in range(v.shape[0]):
            f.write(str(v[j][0][0]) + " ")
        f.write("\n")

    with open("a.txt", "w") as f:   # 把第一个粒子的加速度写下来查看情况
        for j in range(a.shape[0]):
            f.write(str(a[j][0][0]) + " ")
        f.write("\n")

    x0_line = vertices[:, 0, 0]  # 提取 x0[:][0][0] 对应的数据

    plt.plot(x0_line)
    plt.xlabel('frame')
    plt.ylabel('x0')
    plt.title('Line Plot of x0[:][0][0]')
    plt.show()

    x_line = x[:, 0, 0]  # 提取 x[:][0][0] 对应的数据

    plt.plot(x_line)
    plt.xlabel('frame')
    plt.ylabel('x')
    plt.title('Line Plot of x[:][0][0]')
    plt.show()



    v_line = v[:, 0, 0]  # 提取 v[:][0][0] 对应的数据

    plt.plot(v_line)
    plt.xlabel('frame')
    plt.ylabel('v')
    plt.title('Line Plot of v[:][0][0]')
    plt.show()

    a_line = a[:, 0, 0]  # 提取 v[:][0][0] 对应的数据

    plt.plot(a_line)
    plt.xlabel('frame')
    plt.ylabel('a')
    plt.title('Line Plot of a[:][0][0]')
    plt.show()
    #  draw image of v and a
    # 绘制速度热图
    plt.figure(figsize=(10, 6))
    plt.title("Velocity Heatmap")
    plt.xlabel("Particle")
    plt.ylabel("Frame")
    plt.imshow(v[:,:,0].T, cmap='viridis', aspect='auto', origin='lower')
    plt.colorbar(label="Velocity")
    plt.show()

    # 绘制加速度热图
    plt.figure(figsize=(10, 6))
    plt.title("Acceleration Heatmap")
    plt.xlabel("Particle")
    plt.ylabel("Frame")
    plt.imshow(a[:,:,0].T, cmap='viridis', aspect='auto', origin='lower')
    plt.colorbar(label="Acceleration")
    plt.show()