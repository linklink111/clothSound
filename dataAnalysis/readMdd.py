import struct
import numpy as np

# read mdd exported by blender.
# If you want to know how mdd is exported in blender, see Blender Foundation\Blender 3.6\3.6\scripts\addons\io_shape_mdd

def read_mdd_file(file_path):
    with open(file_path, "rb") as mdd_file:
        # 读取头部信息
        num_frames, num_verts = struct.unpack(">2i", mdd_file.read(8))

        # 读取帧时间信息
        frame_times = struct.unpack(">%df" % num_frames, mdd_file.read(num_frames * 4))

        # 读取顶点数据
        vertices = []
        for _ in range(num_frames):
            frame_data = []
            for _ in range(num_verts):
                vertex_data = struct.unpack("fff", mdd_file.read(12))
                frame_data.append(vertex_data)
            vertices.append(frame_data)

    return num_frames, num_verts, frame_times, np.array(vertices)

if  __name__ == "__main__":
    # 指定 MDD 文件路径
    mdd_file_path = "clothSound.mdd"

    # 读取 MDD 文件
    num_frames, num_verts, frame_times, vertices = read_mdd_file(mdd_file_path)

    # 打印信息
    print("Number of frames:", num_frames)
    print("Number of vertices:", num_verts)
    print("Frame times:", frame_times)
    print("Vertices data for first frame:", vertices[0])
