import sys
import torch
import numpy as np
from scipy.fftpack import fft, ifft


# https://www.geeksforgeeks.org/smallest-power-of-2-greater-than-or-equal-to-n/
def next_power_of_2(n):
    if n == 0:
        return 1
    if n & (n - 1) == 0:
        return n
    while n & (n - 1) > 0:
        n &= (n - 1)
    return n << 1


def get_ramp_filter(filter_length, angle_size):
    k = np.arange(-filter_length / 2, filter_length / 2, 1, dtype = np.float32)
    h = np.zeros_like(k, dtype=np.float32)
    h[int(filter_length / 2)] = 1 / 4.0
    odd = np.remainder(k, 2)==1
    h[odd] = -1 / (np.pi * k[odd]) ** 2
    k[int(filter_length / 2)] = 1
    m = (k * angle_size / np.sin(k * angle_size)) ** 2
    m[int(filter_length / 2)] = 1
    return h * m


def get_filter(filter_type, filter_length, angle_size, cut_off=1.0):
    kernel = get_ramp_filter(filter_length, angle_size)
    kernel = np.abs(np.fft.fft(kernel)) * 2
    kernel = kernel[0:int(filter_length / 2 + 1)]
    w = 2 * np.pi * np.arange(0, kernel.shape[0]) / filter_length

    if filter_type=='hann':
        kernel[1:] = kernel[1:] * (1 + np.cos(w[1:] / cut_off)) / 2
    elif filter_type=='ram-lak':
        pass
    elif filter_type=='cosine':
        kernel[1:] = kernel[1:] * np.cos(w[1:] / (2 * cut_off))
    else:
        errstr = filter_type + ': invalid filter type.'
        sys.exit(errstr)

    kernel[w > np.pi * cut_off] = 0
    kernel = np.concatenate((kernel, np.flip(kernel[1:-1], 0)), axis=0)
    return kernel

def filter_sinogram(sinogram, angle_size, filter_type='ram-lak'):
    detector_length, num_angles = sinogram.shape
    filter_length = max(64, next_power_of_2(2 * detector_length))
    angle_size = angle_size * np.pi / 180

    ramp_filter = get_filter('ram-lak', filter_length, angle_size)
    ramp_filter = np.tile(ramp_filter[..., np.newaxis, np.newaxis], (1, num_angles, 2))

    fsinogram = np.zeros_like(ramp_filter)
    fsinogram[
        int(filter_length / 2 - detector_length / 2):
        int(filter_length / 2 + detector_length - detector_length / 2), :, 0
    ] = sinogram
    k = np.arange(-filter_length / 2, filter_length / 2, 1)
    k = np.tile(k[..., np.newaxis, np.newaxis], (1, num_angles, 2))
    fsinogram = fsinogram * np.cos(k * angle_size)

    fsinogram = torch.FloatTensor(fsinogram)
    ramp_filter = torch.FloatTensor(ramp_filter)
    fsinogram = fsinogram.transpose(0, 1)
    ramp_filter = ramp_filter.transpose(0, 1)

    fsinogram = torch.fft(fsinogram, 1)
    fsinogram = fsinogram * ramp_filter
    fsinogram = torch.ifft(fsinogram, 1)[..., 0].t()

    fsinogram = fsinogram[
        int(filter_length / 2 - detector_length / 2):
        int(filter_length / 2 + detector_length - detector_length / 2), :
    ]
    fsinogram = fsinogram.numpy()
    return fsinogram


def design_filter(detector_length, filt_type='ram-lak', d=1.0):
    order = max(64, next_power_of_2(2 * detector_length))
    n = np.arange(order // 2 + 1)
    filtImpResp = np.zeros(order // 2 + 1)
    filtImpResp[0] = 0.25
    filtImpResp[1::2] = -1 / ((np.pi * n[1::2]) ** 2)
    filtImpResp = np.concatenate(
        [filtImpResp, filtImpResp[len(filtImpResp) - 2:0:-1]]
    )
    filt = 2 * fft(filtImpResp).real
    filt = filt[:order // 2 + 1]
    w = 2 * np.pi * np.arange(filt.shape[0]) / order

    if filt_type == 'ram-lak':
        pass
    elif filt_type == 'shepp-logan':
        filt[1:] *= np.sin(w[1:] / (2 * d)) / (w[1:] / (2 * d))
    elif filt_type == 'cosine':
        filt[1:] *= np.cos(w[1:] / (2 * d))
    elif filt_type == 'hamming':
        filt[1:] *= (0.54 + 0.46 * np.cos(w[1:] / d))
    elif filt_type == 'hann':
        filt[1:] *= (1 + np.cos(w[1:] / d)) / 2
    else:
        raise ValueError("Invalid filter type")

    filt[w > np.pi * d] = 0.0
    filt = np.concatenate([filt, filt[len(filt) - 2:0:-1]])
    return filt


def arange(start, stop, step):
    """ Matlab-like arange
    """
    r = np.arange(start, stop, step).tolist()
    if r[-1] + step == stop:
        r.append(stop)
    return np.array(r)