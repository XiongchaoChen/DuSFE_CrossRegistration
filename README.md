# DuSFE: Dual-Channel Squeeze-Fusion-Excitation co-attention for cross-modality registration of cardiac SPECT and CT  **_(Medical Image Analysis 2023 & MICCAI Travel Award 2022)_**

Xiongchao Chen, Bo Zhou, Huidong Xie, Xueqi Guo, Jiazhen Zhang, James S. Duncan, Edward J. Miller, Albert J. Sinusas, John A. Onofrey, and Chi Liu

[[Paper Link](https://doi.org/10.1016/j.media.2023.102840)]

![image](IMAGE/DuSFE.png)

This repository contains the PyTorch implementation of Dual-Branch Squeeze-Fusion-Excitation (DuSFE) Module for cross-modality SPECT-CT registration.

### Citation
If you use this code for your research or project, please cite:

    @inproceedings{chen2022dual,
      title={Dual-Branch Squeeze-Fusion-Excitation Module for Cross-Modality Registration of Cardiac SPECT and CT},
      author={Chen, Xiongchao and Zhou, Bo and Xie, Huidong and Guo, Xueqi and Zhang, Jiazhen and Sinusas, Albert J and Onofrey, John A and Liu, Chi},
      booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
      pages={46--55},
      year={2022},
      organization={Springer}
    }


 ### Environment and Dependencies
 Requirements:
 * Python 3.6.10
 * Pytorch 1.2.0
 * numpy 1.19.2
 * scipy
 * scikit-image
 * h5py
 * tqdm
 
 Our code has been tested with Python 3.6.10, Pytorch 1.2.0, CUDA: 10.0.130 on Ubuntu 18.04.6.

 ### Dataset Setup
    .
    Data
    ├── train                # contain training files
    |   ├── data1.h5
    |       ├── Amap_Trans.mat  
    |       ├── Amap_CT.mat
    |       ├── SPECT_NC.mat
    |       ├── SPECT_SC.mat
    |       ├── Index_Trans.mat
    |   └── ...  
    |
    ├── valid                # contain validation files
    |   ├── data1.h5
    |       ├── Amap_Trans.mat  
    |       ├── Amap_CT.mat
    |       ├── SPECT_NC.mat
    |       ├── SPECT_SC.mat
    |       ├── Index_Trans.mat
    |   └── ... 
    |
    └── test                 # contain testing files
    |   ├── data1.h5
    |       ├── Amap_Trans.mat  
    |       ├── Amap_CT.mat
    |       ├── SPECT_NC.mat
    |       ├── SPECT_SC.mat
    |       ├── Index_Trans.mat
    |   └── ... 
    └── ...  

where \
`Amap_Trans`: rotated CT-based attenuation map with a size of H x W x D. \
`Amap_CT`: aligned CT-based attenuation maps with a size of H x W x D. \
`SPECT_NC`: reconstructed cardiac SPECT image in a photopeak window with a size of H x W x D. \
`SPECT_SC`: reconstructed cardiac SPECT image in a scatter window with a size of H x W x D (optional). \
`Index_Trans`: rigid transformation index with a size of 6 x 1 (3 translational index, 3 rotational index). 

### To Run the Code
Sample training/testing scripts are provided at the root folder as `train_register.sh` and `test_register.sh`.

- Train the model 
```bash
python train.py --experiment_name 'train_register' --model_type 'model_reg' --dataset 'CardiacSPECT_Reg' --data_root '../../Data/Dataset_filename/' --net_G 'DuRegister_DuSE' --net_filter 32 --lr 5e-5 --step_size 1 --gamma 0.99 --n_epochs 400 --batch_size 4 --n_patch_train 1 --patch_size_train 80 80 40 --n_patch_test 1 --patch_size_test 80 80 40 --n_patch_valid 1 --patch_size_valid 80 80 40 --eval_epochs 5 --snapshot_epochs 5 --gpu_ids 0
```

where \
`--experiment_name` experiment name for the code, and save all the training results in this under this "experiment_name" folder. \
`--model_type`: model type used (default convolutional neural networks -- "model_reg"). \
`--dataset`: dataset type. \
`--data_root`: the path of the dataset. \
`--net_G`: neural network model used (default: 'DuRegister_DuSE'). \
`--net_filter`: num of filters in the densely connected layers of DuSFE (default: 32). \
`--lr`: learning rate. \
`--step_size`: num of epoch for learning rate decay .\
`--gamma`: learning decay rate. \
`--n_epochs`: num of epochs of training. \
`--batch_size`: training batch size. \
`--n_patch_train`: number of training patches extracted from each image volume. \
`--patch_size_train`: training patch size. \
`--n_patch_test`: number of testing patches extracted from each image volume. \
`--patch_size_test`: testing patch size. \
`--n_patch_valid`: number of validation patches extracted from each image volume. \
`--patch_size_valid`: validation patch size. \
`--test_epochs`: number of epochs for periodic validation. \
`--save_epochs`: number of epochs for saving trained model. \
`--gpu_ids`: GPU configuration.

- Test the model 
```bash
python test.py -resume './outputs/train_register/checkpoints/model_199.pt' --experiment_name 'test_register_199' --model_type 'model_reg' --dataset 'CardiacSPECT_Reg' --data_root '../../Data/Dataset_filename/' --net_G 'DuRegister_DuSE' --net_filter 32 --batch_size 4 --n_patch_train 1 --patch_size_train 80 80 40 --n_patch_test 1 --patch_size_test 80 80 40 --n_patch_valid 1 --patch_size_valid 80 80 40 --gpu_ids 0
```
where \
`--resume`: the path of the model to be tested. \
`--resume_epoch`: training epoch of the model to be tested. \
`--experiment_name`: experiment name for the code, and save all the testing results in this under this "experiment_name" folder. 

### Data Availability
The original dataset in this study is available from the corresponding author (chi.liu@yale.edu) upon reasonable request and approval of Yale University. 


### Contact 
If you have any questions, please file an issue or directly contact the author:
```
Xiongchao Chen: xiongchao.chen@yale.edu, cxiongchao9587@gmail.com
```




