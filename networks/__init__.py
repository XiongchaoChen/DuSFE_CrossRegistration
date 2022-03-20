import torch.nn as nn
import numpy as np
from utils import arange
from networks.networks import DuRegister_DuSE
import pdb


def set_gpu(network, gpu_ids):
    network.to(gpu_ids[0])  # Default to the 1st GPU
    network = nn.DataParallel(network, device_ids=gpu_ids)  # Parallel computing on multiple GPU
    return network

def get_generator(name, opts):
    if name == "DuRegister_DuSE":
        # Num of channels
        ic_c1 = 1
        ic_c2 = 1
        network = DuRegister_DuSE(n_channels_c1=ic_c1, n_channels_c2=ic_c2, nchannels_extract=opts.net_filter)

    num_param = sum([p.numel() for p in network.parameters() if p.requires_grad])
    print('Number of parameters of Generator: {}'.format(num_param))

    return set_gpu(network, opts.gpu_ids)
