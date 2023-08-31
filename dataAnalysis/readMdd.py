import struct
import numpy as np
import time

# read mdd exported by blender.
# If you want to know how mdd is exported in blender, see Blender Foundation\Blender 3.6\3.6\scripts\addons\io_shape_mdd

def read_mdd_file(file_path):
    with open(file_path, "rb") as mdd_file:
        # 读取头部信息
        num_frames, num_verts = struct.unpack(">2i", mdd_file.read(8))
        print(num_frames)

        # 读取帧时间信息
        frame_times = struct.unpack(">%df" % num_frames, mdd_file.read(num_frames * 4))

        # 读取顶点数据
        vertices = []
        for _ in range(num_frames):
            frame_data = []
            for _ in range(num_verts):
                vertex_data = struct.unpack("fff", mdd_file.read(12))
                print(type(vertex_data))
                print(vertex_data)
                time.sleep(1.00)
                frame_data.append(vertex_data)
            vertices.append(frame_data)

    return num_frames, num_verts, frame_times, np.array(vertices)
def decode_and_print_binary_file(file_path):
    with open(file_path, 'rb') as f:
        while True:
            chunk = f.read(16)  # 逐行读取一部分数据
            if not chunk:
                break
            
            # 在这里添加解码逻辑，将二进制数据解码为可读内容
            decoded_chunk = decode_binary_chunk(chunk)
            
            # 打印解码结果
            print(decoded_chunk)
            time.sleep(1.00)

def decode_binary_chunk(binary_data):
    # 这里是您的解码逻辑，将 binary_data 解码为可读内容
    decoded_result = ...  # 解码操作
    return decoded_result



if  __name__ == "__main__":
    # 指定 MDD 文件路径
    mdd_file_path = "shakeCloth.mdd"

    decode_and_print_binary_file(mdd_file_path)

    # 读取 MDD 文件
    num_frames, num_verts, frame_times, vertices = read_mdd_file(mdd_file_path)

    # 打印信息
    print("Number of frames:", num_frames)
    print("Number of vertices:", num_verts)
    print("Frame times:", frame_times)
    print("Vertices data for first frame:", vertices[0])
