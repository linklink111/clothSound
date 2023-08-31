from readMdd import read_mdd_file
from difference import interpolate1_x, get1_v, get1_a, get1_da
import matplotlib.pyplot as plt
import numpy as np
from math import cos
from scipy.io import wavfile


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
    dt1 = 1/24 #  但是好像dt0和dt1差距可能会有点大... 
    scale = 1
    all_audio = np.zeros(len(np.arange(0, num_frames*1/24, dt1)))
    # vertices /= 1e38 # 有时候由于建模或单位的原因x可能有点大，这里换算一下（blender中的单位是cm）
    for p in range(num_verts):
        print(p)
        particle_data = vertices[:,p,:]
        x = interpolate1_x(particle_data, scale)
        num_particle = x.shape[1]
        num_frames_interp = x.shape[0]
        # x /= 1000000 
        v = get1_v(x, dt1)
        a = get1_a(v, dt1) 
        # 对a进行一个数据清洗，把远远大于平均值的数据写为0
        mean_value = np.mean(a)
        threshold = mean_value*5
        for i in range(a.shape[0]):
            if np.any(a[i]) > threshold:
                a[i] = np.zeros(3)

        da = get1_da(a, dt1)

        c = 340
        lambda_ = 0.5
        omega = 2*np.pi/lambda_
        rho = 1.29
        l = np.array([0.0,0.0,0.0])

        
        # traverse delta_aclerations of all particles
        
        p = np.zeros(num_frames_interp)
        # interpolate
        t = np.arange(0, len(da))
        
        for frame in range(num_frames_interp):
            r = x[frame] - l
            r_norm = np.linalg.norm(r)
            t[frame] = dt1*frame + r_norm/c
            px = rho*da[frame,0]*np.cos(r_norm+omega*t[frame])/(4*np.pi*c*r_norm)
            py = rho*da[frame,1]*np.cos(r_norm+omega*t[frame])/(4*np.pi*c*r_norm)
            pz = rho*da[frame,2]*np.cos(r_norm+omega*t[frame])/(4*np.pi*c*r_norm)
            p[frame] = np.dot(np.array([px,py,pz]),r)

        # 编码
        sample_rate = 44100
        audio = np.interp(np.arange(0, num_frames*1/24, dt1),t,p)
        wavfile.write("tmp.wav", sample_rate,
          (audio * (2**15-1)).astype(np.int16))
    all_audio += audio

    all_audio /= max(-min(all_audio),max(all_audio))*1.05
    wavfile.write("shakeCloth.wav", sample_rate,
          (all_audio * (2**15-1)).astype(np.int16))


        

         


        

    
    