# single model baseline
import sys
import os
sys.path.insert(0, os.getcwd())
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
import random

import logging
from tqdm import tqdm


def callog(message):
    # print(message)
    logger.info(message)

def train_and_evaluate(net, dataloaders, config, device, lin_cls=False):

    def train_step(net, sample, loss_fn, optimizer, device, loss_input_fn):
        optimizer.zero_grad()
        # cut_len = int(sample['inputs'].shape[1] * random.choice(range(1,11)) * 0.1)
        # start = random.choice(range(0, sample['inputs'].shape[1] - cut_len + 1))
        # input = sample['inputs'][:,start:start+cut_len, :, :, :]
        input = sample['inputs']
        outputs,_,_= net(input.to(device))
        outputs = outputs.permute(0, 2, 3, 1)
        ground_truth = loss_input_fn(sample, device)
        loss = loss_fn['mean'](outputs, ground_truth)
        loss.backward()
        optimizer.step()
        return outputs, ground_truth, loss
  
    def evaluate_test(net, evalloader, loss_fn, config):
        net.eval()
        ioussss = []
        max_seq_len = config['MODEL']['max_seq_len']
        for ratio in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
            cut_len = int(max_seq_len*ratio)
            start = 0
            # stride = int(max_seq_len*0.1)
            # starts = range(0,max_seq_len-cut_len+1,stride)
            # for start in starts:
            num_classes = config['MODEL']['num_classes']
            predicted_all = []
            labels_all = []
            losses_all = []
            with torch.no_grad():
                for step, sample in enumerate(evalloader):
                    input = sample['inputs'][:,start:start+cut_len,:,:,:]
                    logits,_,_ = net(input.to(device))
                    logits = logits.permute(0, 2, 3, 1)
                    _, predicted = torch.max(logits.data, -1)
                    ground_truth = loss_input_fn(sample, device)
                    loss = loss_fn['all'](logits, ground_truth)
                    target, mask = ground_truth
                    if mask is not None:
                        predicted_all.append(predicted.view(-1)[mask.view(-1)].cpu().numpy())
                        labels_all.append(target.view(-1)[mask.view(-1)].cpu().numpy())
                    else:
                        predicted_all.append(predicted.view(-1).cpu().numpy())
                        labels_all.append(target.view(-1).cpu().numpy())
                    losses_all.append(loss.view(-1).cpu().detach().numpy())
                    

            callog(f"finished iterating over dataset after step {step} , calculating metrics...")
            predicted_classes = np.concatenate(predicted_all)
            target_classes = np.concatenate(labels_all)
            losses = np.concatenate(losses_all)

            eval_metrics = get_classification_metrics(predicted=predicted_classes, labels=target_classes,
                                                    n_classes=num_classes, unk_masks=None)

            micro_acc, micro_precision, micro_recall, micro_F1, micro_IOU = eval_metrics['micro']
            macro_acc, macro_precision, macro_recall, macro_F1, macro_IOU = eval_metrics['macro']
            class_acc, class_precision, class_recall, class_F1, class_IOU = eval_metrics['class']

            ioussss.append(macro_IOU)

            un_labels, class_loss = get_per_class_loss(losses, target_classes, unk_masks=None)

            callog(f"ratio={str(ratio)},cut_len={str(cut_len)},start={str(start)}")
            callog("Mean (micro) Evaluation metrics (micro/macro), loss: %.7f, iou: %.4f/%.4f, accuracy: %.4f/%.4f, "
                "precision: %.4f/%.4f, recall: %.4f/%.4f, F1: %.4f/%.4f, unique pred labels: %s" %
                (losses.mean(), micro_IOU, macro_IOU, micro_acc, macro_acc, micro_precision, macro_precision,
                micro_recall, macro_recall, micro_F1, macro_F1, np.unique(predicted_classes)))
            callog("-------------------------------------------------------------------------------------")
        callog(f"mean_AUC:{str(sum(ioussss)/len(ioussss))}")
        with open(os.path.join(args.save_path,args.AUC_Metrics_Loss_path), "a", encoding="utf-8") as file:
            file.write(f"AUC:{str(ioussss[-1])}, mean_AUC:{str(sum(ioussss)/len(ioussss))}"+"Mean (micro) Evaluation metrics (micro/macro), loss: %.7f, iou: %.4f/%.4f, accuracy: %.4f/%.4f, "
                "precision: %.4f/%.4f, recall: %.4f/%.4f, F1: %.4f/%.4f, unique pred labels: %s \n" %
                (losses.mean(), micro_IOU, macro_IOU, micro_acc, macro_acc, micro_precision, macro_precision,
                micro_recall, macro_recall, micro_F1, macro_F1, np.unique(predicted_classes)))
        return sum(ioussss)/len(ioussss)

    #------------------------------------------------------------------------------------------------------------------#
    num_classes = config['MODEL']['num_classes']
    num_epochs = config['SOLVER']['num_epochs']
    lr = float(config['SOLVER']['lr_base'])
    train_metrics_steps = config['CHECKPOINT']['train_metrics_steps']
    eval_steps = config['CHECKPOINT']['eval_steps']
    save_steps = config['CHECKPOINT']["save_steps"]
    save_path = config['CHECKPOINT']["save_path"]
    checkpoint = config['CHECKPOINT']["load_from_checkpoint"]
    num_steps_train = len(dataloaders['train'])
    local_device_ids = config['local_device_ids']
    weight_decay = get_params_values(config['SOLVER'], "weight_decay", 0)

    start_global = 1
    start_epoch = 1

    if checkpoint:
        checkpoint= load_from_checkpoint(net, checkpoint, device=device)
        callog(f"successfully load from checkpoint {checkpoint}")

    # callog("current learn rate: %.4f" % lr)

    if len(local_device_ids) > 1:
        net = nn.DataParallel(net, device_ids=local_device_ids)
    net.to(device)

    if save_path and (not os.path.exists(save_path)):
        os.makedirs(save_path)

    copy_yaml(config)

    loss_input_fn = get_loss_data_input(config)
    
    loss_fn = {'all': get_loss(config, device, reduction=None),
               'mean': get_loss(config, device, reduction="mean")}

    trainable_params = get_net_trainable_params(net)
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    optimizer.zero_grad()

    scheduler = build_scheduler(config, optimizer, num_steps_train)

    # writer = SummaryWriter(save_path)

    BEST_AUC = 0
    net.train()

    # eval_metrics = evaluate(net, dataloaders['eval'], loss_fn, config)

    for epoch in range(start_epoch, start_epoch + num_epochs):  # loop over the dataset multiple times
        for step, sample in enumerate(dataloaders['train']):
            abs_step = start_global + (epoch - start_epoch) * num_steps_train + step
            logits, ground_truth, loss = train_step(net, sample, loss_fn, optimizer, device, loss_input_fn=loss_input_fn)
            if len(ground_truth) == 2:
                labels, unk_masks = ground_truth
            else:
                labels = ground_truth
                unk_masks = None
            # print batch statistics ----------------------------------------------------------------------------------#
            if abs_step % train_metrics_steps == 0:
                logits = logits.permute(0, 3, 1, 2)
                batch_metrics = get_mean_metrics(
                    logits=logits, labels=labels, unk_masks=unk_masks, n_classes=num_classes, loss=loss, epoch=epoch,
                    step=step)
                # write_mean_summaries(writer, batch_metrics, abs_step, mode="train", optimizer=optimizer)
                callog("abs_step: %d, epoch: %d, step: %5d, loss: %.7f, batch_iou: %.4f, batch accuracy: %.4f, batch precision: %.4f, "
                      "batch recall: %.4f, batch F1: %.4f, lr: %.7f" %
                      (abs_step, epoch, step + 1, loss, batch_metrics['IOU'], batch_metrics['Accuracy'], batch_metrics['Precision'],
                       batch_metrics['Recall'], batch_metrics['F1'], optimizer.param_groups[0]["lr"]))

            if abs_step % save_steps == 0:
                if len(local_device_ids) > 1:
                    torch.save(net.module.state_dict(), "%s/%depoch_%dstep.pth" % (save_path, epoch, abs_step))
                else:
                    torch.save(net.state_dict(), "%s/%depoch_%dstep.pth" % (save_path, epoch, abs_step))

            # evaluate model ------------------------------------------------------------------------------------------#
            if abs_step % eval_steps == 0:
                eval_metrics = evaluate_test(net, dataloaders['eval'], loss_fn, config)
                if eval_metrics > BEST_AUC:
                    if len(local_device_ids) > 1:
                        torch.save(net.module.state_dict(), "%s/best.pth" % (save_path))
                    else:
                        torch.save(net.state_dict(), "%s/best.pth" % (save_path))
                    BEST_AUC = eval_metrics
                net.train()

        scheduler.step_update(abs_step)
    
    callog(f"-----------Eval dataset BestIOU: {BEST_AUC}")

    # load the best checkpoint and evaluate on test dataset -------------------------------------------------------#
    checkpoint_new = os.path.join(save_path, 'best.pth')
    saved_net = torch.load(checkpoint_new, map_location=device)
    new_saved_net = {}
    for key, value in saved_net.items():
        new_key = f"module.{key}"  # 关键：添加module.前缀
        new_saved_net[new_key] = value
    net.load_state_dict(new_saved_net, strict=True)
    callog(f"successfully load from checkpoint {checkpoint_new}")
    test_metrics = evaluate_test(net, dataloaders['test'], loss_fn, config)
    callog(f"-----------Test dataset BestIOU: {test_metrics}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--config', help='configuration (.yaml) file to use')
    parser.add_argument('--device', default='0,1,2,3,4,5,6,7', type=str, help='gpu ids to use')
    parser.add_argument('--save_path', type=str, help='save_path')
    parser.add_argument('--log_file', default='new_log.log', type=str, help='log_file')
    parser.add_argument('--AUC_Metrics_Loss_path', default='AUC_Metrics_Loss.txt', type=str, help='AUC_Metrics_Loss_path')
    parser.add_argument('--lin', action='store_true', help='train linear classifier only')
    parser.add_argument('--describe', default='This is a tsvit teacher student net.', type=str, help='describe the training process')

    args = parser.parse_args()
    config_file = args.config
    if args.save_path and (not os.path.exists(args.save_path)):
        os.makedirs(args.save_path)
    # 创建一个logger
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)  # 设置日志级别
    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(os.path.join(args.save_path,args.log_file))
    file_handler.setLevel(logging.DEBUG)
    # 定义handler的输出格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    # 将handler添加到logger
    logger.addHandler(file_handler)
    # 使用logger记录信息
    # logger.info('这是一条info信息')

    # callog(args.device)
    device_ids = [int(d) for d in args.device.split(',')]
    lin_cls = args.lin

    device = get_device(device_ids, allow_cpu=False)

    # device = torch.device('cpu')
    # device_ids = [0]  # Force to use CPU for debugging

    config = read_yaml(config_file)
    config['local_device_ids'] = device_ids
    config['CHECKPOINT']['save_path'] = args.save_path
    config['About train'] = args.describe


    max_seq_len = int(config['MODEL']['max_seq_len'] )
    config['MODEL']['max_seq_len'] = max_seq_len
    config['DATASETS']['train']['max_seq_len'] = max_seq_len
    config['DATASETS']['eval']['max_seq_len'] = max_seq_len
    config['DATASETS']['test']['max_seq_len'] = max_seq_len

    dataloaders = get_dataloaders(config)

    net = get_model(config, device)

    train_and_evaluate(net, dataloaders, config, device)
