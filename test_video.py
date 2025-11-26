import sys
import os
sys.path.insert(0, os.getcwd())
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from utils.lr_scheduler import build_scheduler
import numpy as np
import os
from models import get_model
from utils.config_files_utils import read_yaml, copy_yaml, get_params_values
from utils.torch_utils import get_device, get_net_trainable_params, load_from_checkpoint
from data import get_dataloaders
from metrics.torch_metrics import get_mean_metrics
from metrics.numpy_metrics import get_classification_metrics, get_per_class_loss
from metrics.loss_functions import get_loss
from utils.summaries import write_mean_summaries, write_class_summaries
from data import get_loss_data_input
import logging
import tqdm
import matplotlib.pyplot as plt
import torch
from data.PASTIS24_410.dataloader import get_dataloader as get_pastis_dataloader
from data.PASTIS24_410.dataloader import SatImDataset
from data.PASTIS24_410.data_transforms import PASTIS_segmentation_transform
from utils.config_files_utils import get_params_values, read_yaml
from glob import glob
from models.ours.TSViTdense import *

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from utils.lr_scheduler import build_scheduler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
from models import get_model
from utils.config_files_utils import read_yaml, copy_yaml, get_params_values
from utils.torch_utils import get_device, get_net_trainable_params, load_from_checkpoint, load_new
from data import get_dataloaders
from metrics.torch_metrics import get_mean_metrics
from metrics.numpy_metrics import get_classification_metrics, get_per_class_loss
from metrics.loss_functions import get_loss
from utils.summaries import write_mean_summaries, write_class_summaries
from data import get_loss_data_input
import logging
from tqdm import tqdm
import random

config_file = './configs/PASTIS24-410/TSViT_fold1.yaml'
config = read_yaml(config_file)

# save_path = './pics/cam'
# if not os.path.exists(save_path):
#     os.makedirs(save_path)
device = torch.device("cpu")
ratio = 1.0
# config['save_path'] = save_path
config['ratio'] = ratio
config['device'] = device

# 修改config配置
batch_size = 4
config['DATASETS']['train']['batch_size'] = batch_size
config['DATASETS']['eval']['batch_size'] = batch_size
config['DATASETS']['test']['batch_size'] = batch_size
dataloaders = get_dataloaders(config)
# checkpoint = '/data/kangjuyuan/tsseg/ours/exps/baseline/best.pth'
net = get_model(config,device)


# fr = freekd_net(config['MODEL'])
# fr.to(device)
# load_new(fr, checkpoint,partial_restore=True, device=device)
# for name, param in fr.named_parameters():
#     print(f"parameter: {name},requires_grad: {param}")
#     break

# 训练

import torch
from torch.profiler import profile, record_function, ProfilerActivity

def profile_model(model, x, seq_len, ratio=0,from_begin=False):
    with profile(activities=[ProfilerActivity.CPU],
                 record_shapes=True) as prof:
        with record_function("model_inference"):
            model(x, seq_len, ratio=0,from_begin=False)
    
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))



def count_parameters(model):
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            params = param.numel()
            print(f"{name}: {params}")
            total_params += params
    print(f"Total Parameters: {total_params}")
    return total_params

# count_parameters(net)

for step, sample in enumerate(dataloaders['train']):
    # print(f"rgb_images_{sample['img_path'][0]} is showing gt")
    imgs = sample['inputs']
    label = sample['labels']

    print(imgs[:,:,0,0,-1])
    cut_len = int(imgs.shape[1] * random.choice(range(1,11)) * 0.1)
    start = random.choice(range(0, imgs.shape[1] - cut_len + 1))
    tensor = imgs[:,start:start+cut_len, :, :, :]

    print(tensor[:,:,0,0,-1])
    # for ratio in [0.1,0.2,0.4,0.8,1.0]:
    #     cut_len = int(80*ratio)
    #     stride = int(80*0.1)
    #     starts = range(0,80-cut_len+1,stride)
    #     for start in starts:
    #         print(cut_len,start,imgs[:,start:start+cut_len,0,0,-1])
    
    out = net(imgs.to(device))

    # profile_model(net, x=imgs.to(device), seq_len=seq_len.to(device), ratio=0,from_begin=False)
    # out = net(imgs.to(device), seq_len.to(device), ratio=0,from_begin=False)
    print(out[0].shape)
    
    break