import os
from abc import ABC

import numpy as np
from collections import OrderedDict
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm
from scipy.special import entr
import pdb
import csv

from networks import get_generator
from networks.networks import gaussian_weights_init
from models.utils import AverageMeter, get_scheduler, psnr, mse, nmse, nmae, cal_ssim, mae
from skimage.metrics import structural_similarity as ssim
from utils.data_patch_util import *


class REGModel(nn.Module):
    def __init__(self, opts):
        super(REGModel, self).__init__()
        # Basic parameters
        self.loss_names = []  # list
        self.networks = []  # list
        self.optimizers = []  # list
        self.is_train = True if hasattr(opts, 'lr') else False

        # Networks
        self.net_G = get_generator(opts.net_G, opts)
        self.networks.append(self.net_G)

        # Optimizer
        if self.is_train:
            self.lr = opts.lr
            self.loss_names += ['loss_G_L1']
            self.optimizer_G = torch.optim.Adam(self.net_G.parameters(),
                                                lr=self.lr,  # initilize the learning rate for the optimizer
                                                betas=(opts.beta1, opts.beta2),
                                                weight_decay=opts.weight_decay)

            self.optimizers.append(self.optimizer_G)

        # Loss Fuction
        self.criterion = nn.L1Loss()  # L1 loss function.py
        self.opts = opts

    def setgpu(self, gpu_ids):
        self.device = torch.device('cuda:{}'.format(gpu_ids[0]))  # Choose GPU for CUDA computing; For input setting

    def initialize(self):
        [net.apply(gaussian_weights_init) for net in self.networks]

    # LR decay can be realized here
    def set_scheduler(self, opts, epoch=-1):
        self.schedulers = [get_scheduler(optimizer, opts, last_epoch=epoch) for optimizer in self.optimizers]


    def set_input(self, data):
        self.vol_Amap_Trans = data['vol_Amap_Trans'].to(self.device).float()  # [batch_size, 1, 72,72,32]
        self.vol_Amap_CT = data['vol_Amap_CT'].to(self.device).float()
        self.vol_SPECT_NC = data['vol_SPECT_NC'].to(self.device).float()
        self.vol_SPECT_SC = data['vol_SPECT_SC'].to(self.device).float()
        self.vol_Index_Trans = data['vol_Index_Trans'].to(self.device).float() # [batch_size, 6]

    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, name))  # get self.loss_G_L1
        return errors_ret

    def set_epoch(self, epoch):
        self.curr_epoch = epoch

    def forward(self):
        inp_c1 = self.vol_Amap_Trans
        inp_c2 = self.vol_SPECT_NC

        # Require auto grad
        inp_c1.requires_grad_(True)
        inp_c2.requires_grad_(True)
        self.vol_Index_Pred = self.net_G(inp_c1, inp_c2)  # Output data, [batch,1,32,32,32]

    def update_G(self):
        self.optimizer_G.zero_grad()  # Zero gradient
        loss_G_L1 = self.criterion(self.vol_Index_Pred, self.vol_Index_Trans)
        self.loss_G_L1 = loss_G_L1.item()  # <class 'dict_items'>, for discription

        total_loss = loss_G_L1
        total_loss.backward()
        self.optimizer_G.step()

    def optimize(self):  # Use the last 2 functions
        self.forward()
        self.update_G()

    @property  # Only for this function.py
    def loss_summary(self):
        message = ''
        message += 'G_L1: {:.4e} '.format(self.loss_G_L1)
        return message

    # learning rate decay
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()  # learning rate update
        self.lr = self.optimizers[0].param_groups[0]['lr']
        # print('learning rate = {:7f}'.format(lr))

        # self.lr = self.lr_decay * self.lr
        # for param_group in self.optimizer_G.param_groups:
        #     param_group['lr'] = self.lr  # Update the lr for the optimizer

    def save(self, filename, epoch, total_iter):  # Save the net/optimizer state data
        state = {}  # dict
        state['net_G'] = self.net_G.module.state_dict()
        state['opt_G'] = self.optimizer_G.state_dict()

        state['epoch'] = epoch
        state['total_iter'] = total_iter

        torch.save(state, filename)

        print('Saved {}'.format(filename))


    def resume(self, checkpoint_file):
        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        self.net_G.module.load_state_dict(checkpoint['net_G'])
        if self.is_train:
            self.optimizer_G.load_state_dict(checkpoint['opt_G'])

        print('Loaded {}'.format(checkpoint_file))
        return checkpoint['epoch'], checkpoint['total_iter']


    # -------------- Evaluation, Calculate PSNR ---------------
    def evaluate(self, loader):
        val_bar = tqdm(loader)

        # For calculating metrics
        avg_mae_All = AverageMeter()
        avg_mae_X   = AverageMeter()
        avg_mae_Y   = AverageMeter()
        avg_mae_Z   = AverageMeter()

        avg_rot_All = AverageMeter()
        avg_rot_X   = AverageMeter()
        avg_rot_Y   = AverageMeter()
        avg_rot_Z   = AverageMeter()

        for data in val_bar:
            self.set_input(data)  # [batch_size=1, 1, 72,72,32]
            self.forward()
            self.vol_Index_Pred = (100*self.vol_Index_Pred).round()/100  # Precision: 0.01

            # Calculate the metrics
            mae_All = mae(self.vol_Index_Pred[:,0:3], self.vol_Index_Trans[:,0:3])  # Size [1, 3]
            mae_X   = mae(self.vol_Index_Pred[:,0],   self.vol_Index_Trans[:,0])  # Size [1]
            mae_Y   = mae(self.vol_Index_Pred[:,1],   self.vol_Index_Trans[:,1])  # Size [1]
            mae_Z   = mae(self.vol_Index_Pred[:,2],   self.vol_Index_Trans[:,2])  # Size [1]

            rot_All = mae(self.vol_Index_Pred[:,3:6], self.vol_Index_Trans[:,3:6])  # Size [1, 3]
            rot_X   = mae(self.vol_Index_Pred[:,3],   self.vol_Index_Trans[:,3])  # Size [1]
            rot_Y   = mae(self.vol_Index_Pred[:,4],   self.vol_Index_Trans[:,4])  # Size [1]
            rot_Z   = mae(self.vol_Index_Pred[:,5],   self.vol_Index_Trans[:,5])  # Size [1]

            avg_mae_All.update(mae_All)
            avg_mae_X.update(mae_X)
            avg_mae_Y.update(mae_Y)
            avg_mae_Z.update(mae_Z)

            avg_rot_All.update(rot_All)
            avg_rot_X.update(rot_X)
            avg_rot_Y.update(rot_Y)
            avg_rot_Z.update(rot_Z)

            # Descrip show MAE
            message  = 'MAE_All: {:4f} '.format(avg_mae_All.avg)
            message += 'ROT_All: {:4f} '.format(avg_rot_All.avg)
            val_bar.set_description(desc=message)

        # Calculate the average metrics
        self.mae_All = avg_mae_All.avg
        self.mae_X   = avg_mae_X.avg
        self.mae_Y   = avg_mae_Y.avg
        self.mae_Z   = avg_mae_Z.avg

        self.rot_All = avg_rot_All.avg
        self.rot_X   = avg_rot_X.avg
        self.rot_Y   = avg_rot_Y.avg
        self.rot_Z   = avg_rot_Z.avg

    # --------------- Save the Predicted indexes ------------------------------
    def save_indexes(self, loader, folder):
        val_bar = tqdm(loader)
        val_bar.set_description(desc='Saving indexes ...')

        # csv to store the indexes
        with open(os.path.join(folder, 'Index_Trans.csv'), 'w') as f:   # Write csv files
            writer = csv.writer(f)
            writer.writerow(['Case', 'Index_X', 'Index_Y', 'Index_Z', 'Rot_X', 'Rot_Y', 'Rot_Z'])

        with open(os.path.join(folder, 'Index_Pred.csv'), 'w') as f:   # Write csv files
            writer = csv.writer(f)
            writer.writerow(['Case', 'Index_X', 'Index_Y', 'Index_Z', 'Rot_X', 'Rot_Y', 'Rot_Z'])

        # Load data for each batch
        num_count = 0
        for data in val_bar:
            num_count += 1
            self.set_input(data)  # [batch_size, 1, 72,72,32]
            self.forward()
            self.vol_Index_Pred = (100*self.vol_Index_Pred).round()/100  # Precision: 0.1

            with open(os.path.join(folder, 'Index_Trans.csv'), 'a') as f:  # Write csv files
                writer = csv.writer(f)
                writer.writerow([num_count, self.vol_Index_Trans[:, 0].cpu().numpy()[0], self.vol_Index_Trans[:, 1].cpu().numpy()[0], self.vol_Index_Trans[:, 2].cpu().numpy()[0],
                                            self.vol_Index_Trans[:, 3].cpu().numpy()[0], self.vol_Index_Trans[:, 4].cpu().numpy()[0], self.vol_Index_Trans[:, 5].cpu().numpy()[0]])

            with open(os.path.join(folder, 'Index_Pred.csv'), 'a') as f:  # Write csv files
                writer = csv.writer(f)
                writer.writerow([num_count, self.vol_Index_Pred[:, 0].cpu().numpy()[0], self.vol_Index_Pred[:, 1].cpu().numpy()[0], self.vol_Index_Pred[:, 2].cpu().numpy()[0],
                                            self.vol_Index_Pred[:, 3].cpu().numpy()[0], self.vol_Index_Pred[:, 4].cpu().numpy()[0], self.vol_Index_Pred[:, 5].cpu().numpy()[0]])





