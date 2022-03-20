import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.utils import AverageMeter, get_scheduler, get_gan_loss, psnr, get_nonlinearity
import pdb


'''
Dual-Branch Cross-modility Registration module for SPECT (DuSE)
'''

class DuRegister_DuSE(nn.Module):
    def __init__(self, n_channels_c1=1, n_channels_c2=1, nchannels_extract=32):
        super(DuRegister_DuSE, self).__init__()

        # (1) The first downsampling and feature extraction module; 
        self.conv_in_c1 = nn.Conv3d(n_channels_c1, nchannels_extract, kernel_size=3, padding=1, bias=True)
        self.bn_in_c1 = nn.BatchNorm3d(nchannels_extract)
        self.RDB1_c1 = RDB(nchannels_extract, nDenselayer=4, growthRate=32, norm='BN')
        self.RDB2_c1 = RDB(nchannels_extract, nDenselayer=4, growthRate=32, norm='BN')
        self.RDB3_c1 = RDB(nchannels_extract, nDenselayer=4, growthRate=32, norm='BN')

        self.conv_in_c2 = nn.Conv3d(n_channels_c2, nchannels_extract, kernel_size=3, padding=1, bias=True)
        self.bn_in_c2 = nn.BatchNorm3d(nchannels_extract)
        self.RDB1_c2 = RDB(nchannels_extract, nDenselayer=4, growthRate=32, norm='BN')
        self.RDB2_c2 = RDB(nchannels_extract, nDenselayer=4, growthRate=32, norm='BN')
        self.RDB3_c2 = RDB(nchannels_extract, nDenselayer=4, growthRate=32, norm='BN')  # Depth of the dense layers = 4

        # DuSE Attention Block for each layer
        self.DuSE1 = DuSEAttention(nchannels_extract)
        self.DuSE2 = DuSEAttention(nchannels_extract)
        self.DuSE3 = DuSEAttention(nchannels_extract)

        # (2) Residual Dense Connection
        self.RDB_comb = RDB(nchannels_extract*2, nDenselayer=4, growthRate=64, norm='BN')
        self.conv1_comb = nn.Conv3d(nchannels_extract*2, nchannels_extract, kernel_size=3, padding=1, bias=True)  # 32
        self.bn1_comb = nn.BatchNorm3d(nchannels_extract)
        self.conv2_comb = nn.Conv3d(nchannels_extract, 16, kernel_size=3, padding=1, bias=True)  # 16

        # (4) Full-Connected layers
        self.fc1 = nn.Linear(16*10*10*5, 1024, bias=True)
        self.fc2 = nn.Linear(1024, 128, bias=True)
        self.fc3 = nn.Linear(128, 16, bias=True)
        self.fc4 = nn.Linear(16, 6, bias=True)


    def forward(self, inp_c1, inp_c2):
        # (1) Downsampling
        c1_in_bn = F.relu(self.bn_in_c1(self.conv_in_c1(inp_c1)))
        c2_in_bn = F.relu(self.bn_in_c2(self.conv_in_c2(inp_c2)))  
        # Layer 1
        c1_RDB1 = self.RDB1_c1(c1_in_bn)
        c2_RDB1 = self.RDB1_c2(c2_in_bn)
        c1_DuSE1, c2_DuSE1 = self.DuSE1(c1_RDB1, c2_RDB1)  
        c1_pool1 = F.avg_pool3d(c1_DuSE1, 2)
        c2_pool1 = F.avg_pool3d(c2_DuSE1, 2)  
        # Layer 2
        c1_RDB2 = self.RDB2_c1(c1_pool1)
        c2_RDB2 = self.RDB2_c2(c2_pool1)
        c1_DuSE2, c2_DuSE2 = self.DuSE2(c1_RDB2, c2_RDB2)  
        c1_pool2 = F.avg_pool3d(c1_DuSE2, 2)
        c2_pool2 = F.avg_pool3d(c2_DuSE2, 2)  
        # Layer 3
        c1_RDB3 = self.RDB3_c1(c1_pool2)
        c2_RDB3 = self.RDB3_c2(c2_pool2)
        c1_DuSE3, c2_DuSE3 = self.DuSE3(c1_RDB3, c2_RDB3)  
        c1_pool3 = F.avg_pool3d(c1_DuSE3, 2)  
        c2_pool3 = F.avg_pool3d(c2_DuSE3, 2)  

        # (2) Residual Dense Conection
        comb_inp = torch.cat((c1_pool3, c2_pool3), 1)  
        comb_RDB = self.RDB_comb(comb_inp)  
        comb_conv1 = F.relu(self.bn1_comb(self.conv1_comb(comb_RDB)))  
        comb_conv2 = self.conv2_comb(comb_conv1)  

        # (3) Fully-Connected Layers
        comb_flatten = torch.flatten(comb_conv2, start_dim=1, end_dim=- 1)  
        comb_fc1 = self.fc1(comb_flatten)  
        comb_fc2 = self.fc2(comb_fc1)  
        comb_fc3 = self.fc3(comb_fc2)  
        comb_fc4 = self.fc4(comb_fc3)  

        out = comb_fc4
        return out


'''
Dual-Branch Squeeze-and-Excitation Attention Module 
'''
class DuSEAttention(nn.Module):
    def __init__(self, n_channels_extract=32):
        super(DuSEAttention, self).__init__()
        # (1) Spatial-Squeeze + Channel-Excitation
        self.avg_pool_ch1 = nn.AdaptiveAvgPool3d((1,1,1))
        self.avg_pool_ch2 = nn.AdaptiveAvgPool3d((1,1,1))
        self.fc_comb = nn.Linear(n_channels_extract * 2, n_channels_extract, bias=True)
        self.fc_ch1 = nn.Linear(n_channels_extract, n_channels_extract, bias=True)
        self.fc_ch2 = nn.Linear(n_channels_extract, n_channels_extract, bias=True)

        # (2) Channel-Squeeze + Spatial-Excitation
        self.conv_squeeze_ch1 = nn.Conv3d(n_channels_extract, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_squeeze_ch2 = nn.Conv3d(n_channels_extract, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_comb = nn.Conv3d(2, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_adjust_ch1 = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_adjust_ch2 = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=1, bias=True)

        # (3) Concatenation + Feature Fusion
        self.conv_fuse_ch1 = nn.Conv3d(n_channels_extract*3, n_channels_extract, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn_fuse_ch1 = nn.BatchNorm3d(n_channels_extract)
        self.conv_fuse_ch2 = nn.Conv3d(n_channels_extract*3, n_channels_extract, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn_fuse_ch2 = nn.BatchNorm3d(n_channels_extract)


    def forward(self, inp_ch1, inp_ch2):
        # Basic Information
        batch_size, n_channels, D, H, W = inp_ch1.size()

        # (1) Spatial-Squeeze + Channel-Excitation
        squeeze_ch1 = self.avg_pool_ch1(inp_ch1).view(batch_size, n_channels)  # [B, C, 1,1,1] to [B, C]
        squeeze_ch2 = self.avg_pool_ch2(inp_ch2).view(batch_size, n_channels)  # [B, C, 1,1,1] to [B, C]
        squeeze_comb = torch.cat((squeeze_ch1, squeeze_ch2), 1)  # [B, C*2]

        # Fully connected layers
        fc_comb = self.fc_comb(squeeze_comb)
        # fc_ch1 = self.fc_ch1(fc_comb)
        # fc_ch2 = self.fc_ch2(fc_comb)
        fc_ch1 = torch.sigmoid(self.fc_ch1(fc_comb))
        fc_ch2 = torch.sigmoid(self.fc_ch2(fc_comb))

        # Multiplication
        inp_ch1_scSE = torch.mul(inp_ch1, fc_ch1.view(batch_size, n_channels, 1, 1, 1))
        inp_ch2_scSE = torch.mul(inp_ch2, fc_ch2.view(batch_size, n_channels, 1, 1, 1))  # [B, C, D,H,W]


        # (2) Channel-Squeeze + Spatial-Excitation
        squeeze_volume_ch1 = self.conv_squeeze_ch1(inp_ch1)
        squeeze_volume_ch2 = self.conv_squeeze_ch2(inp_ch2)  # [B, 1, D,H,W]
        squeeze_volume_comb = torch.cat((squeeze_volume_ch1, squeeze_volume_ch2), 1)  # [B, 2, D,H,W]

        # Fusion Layer
        conv_comb = self.conv_comb(squeeze_volume_comb)  # [B, 1, D,H,W]
        # conv_adjust_ch1 = self.conv_adjust_ch1(conv_comb)
        # conv_adjust_ch2 = self.conv_adjust_ch2(conv_comb)  # [B, 1, D,H,W]
        conv_adjust_ch1 = torch.sigmoid(self.conv_adjust_ch1(conv_comb))
        conv_adjust_ch2 = torch.sigmoid(self.conv_adjust_ch2(conv_comb))  # [B, 1, D,H,W]

        # Multiplication
        inp_ch1_csSE = torch.mul(inp_ch1, conv_adjust_ch1.view(batch_size, 1, D, H, W))
        inp_ch2_csSE = torch.mul(inp_ch2, conv_adjust_ch2.view(batch_size, 1, D, H, W))  # [B, C, D,H,W]


        # (3) Concatenation + Feature Fusion
        inp_ch1_fuse = self.bn_fuse_ch1(inp_ch1 + inp_ch1_scSE + inp_ch1_csSE)
        inp_ch2_fuse = self.bn_fuse_ch2(inp_ch2 + inp_ch2_scSE + inp_ch2_csSE)

        return inp_ch1_fuse, inp_ch2_fuse




# Residual dense block
class RDB(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate, norm='None'):
        super(RDB, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):
            modules.append(make_dense(nChannels_, growthRate, norm=norm))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv3d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        out = self.dense_layers(x)

        out = self.conv_1x1(out)

        out = out + x # Residual
        return out


# Dense Block
class make_dense(nn.Module):
    def __init__(self, nChannels, growthRate, norm='None'):
        super(make_dense, self).__init__()
        self.conv = nn.Conv3d(nChannels, growthRate, kernel_size=3, padding=1, bias=False)
        self.norm = norm
        self.bn = nn.BatchNorm3d(growthRate)


    def forward(self, x):
        out = self.conv(x)
        if self.norm == 'BN':
            out = self.bn(out)
        out = F.relu(out)

        out = torch.cat((x, out), 1)
        return out



def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)


if __name__ == '__main__':
    pass

