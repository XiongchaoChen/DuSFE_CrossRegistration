import os
import argparse
import json
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.backends.cudnn as cudnn

from torchvision.utils import save_image
from utils import prepare_sub_folder, prepare_sub_folder_v2
from datasets import get_datasets
from models import create_model

import scipy.io as sio
import csv


# New a parser
parser = argparse.ArgumentParser(description='CardiacSPECT')

# model name
parser.add_argument('--experiment_name', type=str, default='test_RDN_1GD_1BMI_1ST', help='give a model name before training')    
parser.add_argument('--model_type', type=str, default='model_cnn', help='give a model name before training')
parser.add_argument('--resume', type=str, default=None, help='Filename of the checkpoint to resume')

# dataset
parser.add_argument('--data_root', type=str, default='../Data/Processed/', help='data root folder')
parser.add_argument('--dataset', type=str, default='CardiacSPECT', help='dataset name')

# network architectures
parser.add_argument('--net_G', type=str, default='RDN', help='generator network')   
parser.add_argument('--net_filter', type=int, default=64, help='number of network filters')
parser.add_argument('--use_amap_trans', default=False, action='store_true', help='use trans attmap to input into the network')   
parser.add_argument('--use_amap_pred', default=False, action='store_true', help='use pred attmap to input into the network')  
parser.add_argument('--use_spect_nc', default=False, action='store_true', help='use spect_nc')   
parser.add_argument('--use_spect_sc', default=False, action='store_true', help='use spect_sc')   
parser.add_argument('--norm', type=str, default='None', help='Normalization for each convolution')  

# training options
parser.add_argument('--n_epochs', type=int, default=1000, help='number of epoch')
parser.add_argument('--batch_size', type=int, default=12, help='training batch size')
parser.add_argument('--n_patch_train', type=int, default=36, help='number of patch to crop for training')
parser.add_argument('--patch_size_train', nargs='+', type=int, default=[32, 32, 32], help='randomly cropped patch size for train')
# parser.add_argument('--AUG', default=False, action='store_true', help='use augmentation')

# evaluation options
parser.add_argument('--eval_epochs', type=int, default=5, help='evaluation epochs')
parser.add_argument('--save_epochs', type=int, default=5, help='save evaluation for every number of epochs')
parser.add_argument('--n_patch_test', type=int, default=1, help='number of patch to crop for evaluation')
parser.add_argument('--patch_size_test', nargs='+', type=int, default=[32, 32, 32], help='ordered cropped patch size for evaluation')
parser.add_argument('--test_pad', nargs='+', type=int, default=[0, 0, 8], help='edge padding for testing data')
parser.add_argument('--n_patch_valid', type=int, default=1, help='number of patch to crop for evaluation')
parser.add_argument('--patch_size_valid', nargs='+', type=int, default=[32, 32, 32], help='ordered cropped patch size for evaluation')
parser.add_argument('--valid_pad', nargs='+', type=int, default=[0, 0, 8], help='edge padding for validation data')

# optimizer
# parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for ADAM')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for ADAM')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')

# learning rate policy
parser.add_argument('--lr_policy', type=str, default='step', help='learning rate decay policy')
parser.add_argument('--step_size', type=int, default=1000, help='step size for step scheduler')
parser.add_argument('--gamma', type=float, default=1, help='decay ratio for step scheduler')

# logger options
parser.add_argument('--snapshot_epochs', type=int, default=5, help='save model for every number of epochs')
parser.add_argument('--log_freq', type=int, default=100, help='save model for every number of epochs')
parser.add_argument('--output_path', default='./', type=str, help='Output path.')

# other
parser.add_argument('--num_workers', type=int, default=8, help='number of threads to load data')
parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0], help='list of gpu ids')
opts = parser.parse_args()

options_str = json.dumps(opts.__dict__, indent=4, sort_keys=False)

print("------------------- Options -------------------")
print(options_str[2:-2])
print("-----------------------------------------------")

cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = create_model(opts)
model.setgpu(opts.gpu_ids)

num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Number of parameters: {} \n'.format(num_param))

# Resume the training model
if opts.resume is None:
    model.initialize()
    ep0 = -1
    total_iter = 0
else:
    ep0, total_iter = model.resume(opts.resume)


# select dataset
_ , _, test_set = get_datasets(opts)
test_loader = DataLoader(dataset=test_set, num_workers=opts.num_workers, batch_size=1, shuffle=False)

# Setup directories
output_directory = os.path.join(opts.output_path, 'outputs', opts.experiment_name)
image_directory = prepare_sub_folder_v2(output_directory)

# evaluation
print('Normal Evaluation ......')
model.eval()
with torch.no_grad():
    model.evaluate(test_loader)  
    model.save_indexes(test_loader, image_directory)  

# Record the epoch, psnr, ssim and mse
with open(os.path.join(image_directory, 'test_metrics.csv'), 'w') as f:   
    writer = csv.writer(f)
    writer.writerow(['Epoch', 'MAE_All', 'ROT_All', 'MAE_X', 'MAE_Y', 'MAE_Z', 'ROT_X', 'ROT_Y', 'ROT_Z'])  
    writer.writerow([ep0, model.mae_All, model.rot_All, model.mae_X, model.mae_Y, model.mae_Z, model.rot_X, model.rot_Y, model.rot_Z])





