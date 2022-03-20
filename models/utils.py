import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import log10
from torch.optim import lr_scheduler
from skimage.metrics import structural_similarity as ssim
import pdb


def get_nonlinearity(name):
    """Helper function.py to get non linearity module, choose from relu/softplus/swish/lrelu"""
    if name == 'relu':
        return nn.ReLU(inplace=True)
    elif name == 'softplus':
        return nn.Softplus()
    elif name == 'swish':
        return Swish(inplace=True)
    elif name == 'lrelu':
        return nn.LeakyReLU()


class Swish(nn.Module):
    def __init__(self, inplace=False):
        """The Swish non linearity function.py"""
        super().__init__()
        self.inplace = True

    def forward(self, x):
        if self.inplace:
            x.mul_(F.sigmoid(x))
            return x
        else:
            return x * F.sigmoid(x)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_scheduler(optimizer, opts, last_epoch=-1):
    if 'lr_policy' not in opts or opts.lr_policy == 'constant':
        scheduler = None

    elif opts.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opts.step_size,
                                        gamma=opts.gamma, last_epoch=last_epoch)
    elif opts.lr_policy == 'lambda':
        def lambda_rule(ep):
            lr_l = 1.0 - max(0, ep - opts.epoch_decay) / float(opts.n_epochs - opts.epoch_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=last_epoch)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opts.lr_policy)
    return scheduler


def get_recon_loss(opts):
    loss = None
    if opts['recon'] == 'L2':
        loss = nn.MSELoss()
    elif opts['recon'] == 'L1':
        loss = nn.L1Loss()

    return loss


def get_gan_loss(opts):
    if opts.gan_loss == 'mse':
        loss = LSGANLoss()
        real_target = 1.0
        fake_target = 0.0

    elif opts.gan_loss == 'hinge':
        loss = HingeGANLoss()
        real_target = 1.0
        fake_target = -1.0

    else:
        raise NotImplementedError

    return loss, real_target, fake_target


class MaskL1Loss(nn.Module):
    """ Calculate Masked L1 Loss:

            loss_ij = |x_ij - gt_ij|  if  mask_ij = 1
                      0               if  mask_ij = 0

    """
    def __init__(self):
        super(MaskL1Loss, self).__init__()

    def forward(self, x, gt, mask):
        numel = mask.sum()
        loss = (x - gt).abs() * mask

        return loss.sum() / numel

# Least Square GAN Loss
class LSGANLoss(nn.Module):
    def __init__(self):
        super(LSGANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(1.0))  # Creat non-network parameters
        self.register_buffer('fake_label', torch.tensor(0.0))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)  # Expand dimension

    def __call__(self, input, target_is_real, phase=None):  # Call the method as this function.py
        target_tensor = self.get_target_tensor(input, target_is_real)
        return (input - target_tensor).pow(2).mean()


class HingeGANLoss(nn.Module):
    def __init__(self):
        super(HingeGANLoss, self).__init__()

        self.relu = nn.ReLU()

    def forward(self, input, target_is_real, phase=None):
        if phase == 'gen':
            return -input.mean()
        if target_is_real:
            return self.relu(1.0 - input).mean()

        else:
            return self.relu(1.0 + input).mean()


class MaskGANLoss(nn.Module):
    def __init__(self, use_lsgan=True):
        super(MaskGANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(1.0))
        self.register_buffer('fake_label', torch.tensor(0.0))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, mask, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class FPLoss(nn.Module):
    def __init__(self):
        super(FPLoss, self).__init__()
        isocenter_distance = 1075
        image_shape = 416
        fan_rotation_inremental = 360.0 / 320
        self.FP = FanBeamForwardProjection(image_shape=image_shape,
                                      isocenter_distance=isocenter_distance,
                                      rotation_resolution=fan_rotation_inremental).cuda()
        self.scale = 512 / 416 * 0.03

    def __call__(self, input, target):
        input_proj = self.FP(input.squeeze(1))
        target_proj = self.FP(target.squeeze(1))

        return (input_proj - target_proj).pow(2).mean() * self.scale


def mar_psnr(sr_image, gt_image, mask):
    assert sr_image.size(0) == gt_image.size(0) == 1

    window = np.array([-175, 275]) / 1000 * 0.192 + 0.192
    peak_signal = (175 + 275) / 1000 * 0.192
    numel = (1 - mask).sum().item()

    sr_image = sr_image.clamp(*window)
    gt_image = gt_image.clamp(*window)
    se = ((sr_image - gt_image) * (1 - mask)).pow(2).sum().item()
    mse = se / numel

    return 10 * log10(peak_signal ** 2 / mse)

# PSNR, self-design
def psnr(sr_image, gt_image):
    assert sr_image.size(0) == gt_image.size(0) == 1

    peak_signal = (gt_image.max() - gt_image.min()).item()  # This is self-design part

    mse_ = (sr_image - gt_image).pow(2).mean().item()

    psnr_ = 10 * log10(peak_signal ** 2 / mse_)

    return psnr_

# MSE
def mse(sr_image, gt_image):
    assert sr_image.size(0) == gt_image.size(0) == 1

    mse_ = (sr_image - gt_image).pow(2).mean().item()

    return mse_

# NMSE
def nmse(sr_image, gt_image):
    assert sr_image.size(0) == gt_image.size(0) == 1

    mse_ = (sr_image - gt_image).pow(2).mean().item()
    nmse_ = mse_/(gt_image.pow(2).mean().item())

    return nmse_

# NMAE
def nmae(sr_image, gt_image):
    # assert sr_image.size(0) == gt_image.size(0) == 1

    mae_ = (sr_image - gt_image).abs().mean().item()
    nmae_ = mae_/(gt_image.abs().mean().item())

    return nmae_

# MAE
def mae(sr_image, gt_image):
    assert sr_image.size(0) == gt_image.size(0) == 1

    mae_ = (sr_image - gt_image).abs().mean().item()

    return mae_

# SSIM | self-design
def cal_ssim(sr_image, gt_image):
    ssim_ = ssim(sr_image, gt_image, gaussian_weights=True, data_range=gt_image.max() - gt_image.min())
    return ssim_


class SparseLoss(nn.Module):
    def __init__(self):
        super(SparseLoss, self).__init__()
        self.register_buffer('target', torch.tensor(0.0))
        self.loss = nn.L1Loss()

    def forward(self, input):
        target_tensor = self.target
        return self.loss(input, target_tensor.expand_as(input))


class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, x):
        h_tv = (x[:, :, 1:, :] - x[:, :, :-1, :]).pow(2).mean()
        w_tv = (x[:, :, :, 1:] - x[:, :, :, :-1]).pow(2).mean()
        return h_tv + w_tv
