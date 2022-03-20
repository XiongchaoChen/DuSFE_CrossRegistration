import os
import argparse
import json
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import pdb

from torchvision.utils import save_image
from utils import prepare_sub_folder
from datasets import get_datasets
from models import create_model

import scipy.io as sio  # import and output
import csv  # Save .csv files (similar to .xls file)

# New a Parser
parser = argparse.ArgumentParser(description='CardiacSPECT')

# model name
parser.add_argument('--experiment_name', type=str, default='train_SERDUNet_1GD_1BMI_1ST', help='give a model name before training') 
parser.add_argument('--model_type', type=str, default='model_cnn', help='give a model name before training')
parser.add_argument('--resume', type=str, default=None, help='Filename of the checkpoint to resume')

# dataset
parser.add_argument('--data_root', type=str, default='../../Data/Processed_02x29x2020/', help='data root folder')
parser.add_argument('--dataset', type=str, default='CardiacSPECT', help='dataset name')

# network architectures, (discriminators e.g. cD, sD, are not used in the paper
parser.add_argument('--net_G', type=str, default='UNet', help='generator network')   
parser.add_argument('--net_filter', type=int, default=64, help='number of network filters')
parser.add_argument('--use_amap_trans', default=False, action='store_true', help='use trans attmap to input into the network')  
parser.add_argument('--use_amap_pred', default=False, action='store_true', help='use pred attmap to input into the network')  
parser.add_argument('--use_spect_nc', default=False, action='store_true', help='use spect_nc')   
parser.add_argument('--use_spect_sc', default=False, action='store_true', help='use spect_sc')  
parser.add_argument('--norm', type=str, default='None', help='Normalization for each convolution')  

# training options
parser.add_argument('--n_epochs', type=int, default=1000, help='number of epoch')
parser.add_argument('--batch_size', type=int, default=2, help='training batch size')
parser.add_argument('--n_patch_train', type=int, default=1, help='number of patch to crop for training')
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
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for ADAM')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for ADAM')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')

# learning rate policy
parser.add_argument('--lr_policy', type=str, default='step', help='learning rate decay policy')
parser.add_argument('--step_size', type=int, default=1000, help='step size for step scheduler')
parser.add_argument('--gamma', type=float, default=1, help='decay ratio for step scheduler')

# logger options
parser.add_argument('--snapshot_epochs', type=int, default=5, help='save model for every number of epochs')
parser.add_argument('--log_freq', type=int, default=100, help='save loss for every number of epochs')
parser.add_argument('--output_path', default='./', type=str, help='Output path.')

# other
parser.add_argument('--num_workers', type=int, default=8, help='number of threads to load data')
parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0], help='list of gpu ids')
opts = parser.parse_args()  # <class 'argparse.Namespace'>

options_str = json.dumps(opts.__dict__, indent=4, sort_keys=False)  # class 'str', sth like 'dict'

print("------------------- Options -------------------")
print(options_str[2:-2])
print("-----------------------------------------------")

cudnn.benchmark = True   
model = create_model(opts)  
model.setgpu(opts.gpu_ids) 


# Network initialize
if opts.resume is None:
    model.initialize()   # Gaussian Initialize the Network
    ep0 = -1
    total_iter = 0
else:
    ep0, total_iter = model.resume(opts.resume)   # When testing, inherite training.pt file

# Schedule: Learning rate decrease policy
# ep0 = last epoch
model.set_scheduler(opts, ep0)  # Learning rate decay policy;
ep0 += 1                        # ep0 = 0; Training can be resumed and start from the next epoch
print('Start training at epoch {} \n'.format(ep0))  # 0

# select dataset
train_set, valid_set, test_set = get_datasets(opts)
train_loader = DataLoader(dataset=train_set, num_workers=opts.num_workers, batch_size=opts.batch_size, shuffle=True)  
valid_loader = DataLoader(dataset=valid_set, num_workers=opts.num_workers, batch_size=1, shuffle=False)
test_loader  = DataLoader(dataset=test_set,  num_workers=opts.num_workers, batch_size=1, shuffle=False)

# Setup directories + save json&csv files
output_directory = os.path.join(opts.output_path, 'outputs', opts.experiment_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)   

# Json files
with open(os.path.join(output_directory, 'options.jsonoptions'), 'w') as f:
    f.write(options_str)

# New CSV files
with open(os.path.join(output_directory, 'train_loss.csv'), 'w') as f:  
    writer = csv.writer(f)
    writer.writerow(['epoch'] + model.loss_names)  

with open(os.path.join(output_directory, 'table_metrics_valid.csv'), 'w') as f:  
    writer = csv.writer(f)
    writer.writerow(['Epoch', 'MAE_All', 'ROT_All', 'MAE_X', 'MAE_Y', 'MAE_Z', 'ROT_X', 'ROT_Y', 'ROT_Z'])  # empty here
    # Averaged error; x,y,z errors; Max/Min values of errors in x,y,z

with open(os.path.join(output_directory, 'table_metrics_test.csv'), 'w') as f:   # Write CSV, some metadata
    writer = csv.writer(f)
    writer.writerow(['Epoch', 'MAE_All', 'ROT_All', 'MAE_X', 'MAE_Y', 'MAE_Z', 'ROT_X', 'ROT_Y', 'ROT_Z'])  # empty here


# ########### Traing Loop ###############
for epoch in range(ep0, opts.n_epochs): 
    train_bar = tqdm(train_loader)  

    model.train()
    model.set_epoch(epoch)

    for it, data in enumerate(train_bar):  
        total_iter += 1  # 1
        model.set_input(data)
        model.optimize()
        train_bar.set_description(desc='[Epoch {}, lr={:.6f}]'.format(model.curr_epoch, model.lr) + model.loss_summary)  

    # Save loss per epoch
    with open(os.path.join(output_directory, 'train_loss.csv'), 'a') as f:  
        writer = csv.writer(f)
        writer.writerow([epoch] + list(model.get_current_losses().values()))

    model.update_learning_rate()

    # save checkpoint
    if (epoch+1) % opts.snapshot_epochs == 0:  
        checkpoint_name = os.path.join(checkpoint_directory, 'model_{}.pt'.format(epoch))
        print('Saving Trained model ......')
        model.save(checkpoint_name, epoch, total_iter)

########## evaluation ############
    if (epoch+1) % opts.eval_epochs == 0:
        print('Normal Evaluation ......')

        # Validation
        model.eval()
        with torch.no_grad():
            model.evaluate(valid_loader)

        with open(os.path.join(output_directory, 'table_metrics_valid.csv'), 'a') as f:   
            writer = csv.writer(f)
            writer.writerow([epoch, model.mae_All, model.rot_All, model.mae_X, model.mae_Y, model.mae_Z, model.rot_X, model.rot_Y, model.rot_Z])

        # Testing
        model.eval()
        with torch.no_grad():
            model.evaluate(test_loader)

        with open(os.path.join(output_directory, 'table_metrics_test.csv'), 'a') as f:  
            writer = csv.writer(f)
            writer.writerow([epoch, model.mae_All, model.rot_All, model.mae_X, model.mae_Y, model.mae_Z, model.rot_X, model.rot_Y, model.rot_Z])

