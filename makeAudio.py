import os
import simpleaudio as sa
import numpy as np
import wavefile as wf
import soundfile as sf
from scipy.interpolate import interp1d
from scipy.signal import butter, lfilter
from scipy.io import wavfile

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
    for i in range(start+1, end):
        #        pressure = load_array(f"sheetResearch/soundData/pressure_frame{i}.txt")
        #        t = load_array(f"sheetResearch/soundData/t_frame{i}.txt")
        pressure = np.loadtxt(f"cache/pressure{i}.txt", delimiter=',').flatten()
        t = np.loadtxt(f"cache/t_{i}.txt", delimiter=',').flatten()
        # print(pressure.shape)
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

    wavfile.write(f"{fname}.wav", sample_rate,
          (audio * (2**15-1)).astype(np.int16))
    
if __name__ == '__main__':
    make_audio('test1',start=0,end=1200)