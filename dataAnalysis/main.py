from readMdd import read_mdd_file
from difference import interpolate_x, get_v, get_a, get_da
import matplotlib.pyplot as plt
import numpy as np
from math import cos


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
    
    dt0 = 1/24
    dt1 = 0.0001*1/24 #  但是好像dt0和dt1差距可能会有点大...
    # vertices /= 1e38 # 有时候由于建模或单位的原因x可能有点大，这里换算一下（blender中的单位是cm）
    x = interpolate_x(vertices, dt0, dt1)
    num_particle = x.shape[1]
    num_frames_interp = x.shape[0]
    # x /= 1000000 
    v = get_v(x, dt1)
    a = get_a(v, dt1) 
    da = get_da(a, dt1)

    c = 340
    lambda_ = 0.5
    omega = 2*np.pi/lambda_
    rho = 1.29
    l = np.array([0.0,0.0,0.0])

    all_audio = np.zeros(len(np.arange(0, num_frames*1/24, dt1)))
    # traverse delta_aclerations of all particles
    for particle in range(num_particle):
        mass_da_seq = da[:, particle,:]
        p = np.zeros(num_frames_interp)
        # interpolate
        t = np.arange(0, len(mass_da_seq))
        r = x[frame,particle] - l
        r_norm = np.linalg.norm(r)
        for frame in num_frames_interp:
            t[frame] = dt1*frame + r_norm/c
            px = rho*mass_da_seq[frame,0]*np.cos(r_norm+omega*t[frame])/(4*np.pi*c*r_norm)
            py = rho*mass_da_seq[frame,1]*np.cos(r_norm+omega*t[frame])/(4*np.pi*c*r_norm)
            pz = rho*mass_da_seq[frame,2]*np.cos(r_norm+omega*t[frame])/(4*np.pi*c*r_norm)
            p[frame] = np.dot(np.array([px[frame],py[frame],pz[frame]]),r)

        # 编码
        sample_rate = 44100
        audio = np.interp(np.arange(0, num_frames*1/24, dt1),t,p)
        all_audio += audio
        

         


        

    
    