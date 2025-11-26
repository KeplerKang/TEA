# teacher student model
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
import logging
from tqdm import tqdm
import random


def train_and_evaluate(net, dataloaders, config, device, save_path, lin_cls=False):

    def train_step(net, sample, loss_fn, optimizer, device, loss_input_fn):
        optimizer.zero_grad()
        # print(sample['inputs'].shape)
        teacher_x, student_x, mse_loss, dice_loss= net(sample['inputs'].to(device),mode='train')

        teacher_x = teacher_x.permute(0, 2, 3, 1)
        student_x = student_x.permute(0, 2, 3, 1)
        ground_truth = loss_input_fn(sample, device)
        loss_teacher = loss_fn['mean'](teacher_x, ground_truth)
        loss_student = loss_fn['mean'](student_x, ground_truth)

        all_mse_loss = mse_loss.mean()
        mdice_loss = dice_loss.mean()
        loss = loss_teacher + loss_student + all_mse_loss + mdice_loss

        loss.backward()
        optimizer.step()
        return teacher_x, student_x, ground_truth, loss, loss_teacher, loss_student, all_mse_loss, mdice_loss
  
    def evaluate_test(net, evalloader, loss_fn, config, device):
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
            teacher_predicted_all = []
            student_predicted_all = []
            labels_all = []
            losses_teacher_all = []
            losses_student_all = []
            with torch.no_grad():
                for step, sample in enumerate(evalloader):
                    teacher_x, student_x, loss_x, dice_loss= net(sample['inputs'].to(device), mode='eval', cut_len=cut_len, start=start)

                    teacher_x = teacher_x.permute(0, 2, 3, 1)
                    student_x = student_x.permute(0, 2, 3, 1)
                    _, teacher_predicted = torch.max(teacher_x.data, -1)
                    _, student_predicted = torch.max(student_x.data, -1)
                    ground_truth = loss_input_fn(sample, device)
                    loss_teacher = loss_fn['all'](teacher_x, ground_truth)
                    loss_student = loss_fn['all'](student_x, ground_truth)

                    target, mask = ground_truth
                    if mask is not None:
                        teacher_predicted_all.append(teacher_predicted.view(-1)[mask.view(-1)].cpu().numpy())
                        student_predicted_all.append(student_predicted.view(-1)[mask.view(-1)].cpu().numpy())
                        labels_all.append(target.view(-1)[mask.view(-1)].cpu().numpy())
                    else:
                        teacher_predicted_all.append(teacher_predicted.view(-1).cpu().numpy())
                        student_predicted_all.append(student_predicted.view(-1).cpu().numpy())
                        labels_all.append(target.view(-1).cpu().numpy())
                    losses_teacher_all.append(loss_teacher.view(-1).cpu().detach().numpy())
                    losses_student_all.append(loss_student.view(-1).cpu().detach().numpy())


            callog(f"finished iterating over dataset after step {step} , calculating metrics...")
            callog(f"ratio={str(ratio)},cut_len={str(cut_len)},start={str(start)}")
            teacher_predicted_classes = np.concatenate(teacher_predicted_all)
            student_predicted_classes = np.concatenate(student_predicted_all)
            target_classes = np.concatenate(labels_all)
            losses_teacher = np.concatenate(losses_teacher_all)
            losses_student = np.concatenate(losses_student_all)

            teacher_eval_metrics = get_classification_metrics(predicted=teacher_predicted_classes, labels=target_classes,
                                                    n_classes=num_classes, unk_masks=None)
            student_eval_metrics = get_classification_metrics(predicted=student_predicted_classes, labels=target_classes,
                                                    n_classes=num_classes, unk_masks=None)
            un_labels, class_loss_teacher = get_per_class_loss(losses_teacher, target_classes, unk_masks=None)
            _, class_loss_student = get_per_class_loss(losses_student, target_classes, unk_masks=None)

            callog(
                "-----------------------------------------------------------------------------------------------------------------------------------------------------------------")
            callog("******  teacher Evaluation metrics  ******")
            callog("Mean (micro) Evaluation metrics (micro/macro), loss: %.7f, iou: %.4f/%.4f, accuracy: %.4f/%.4f, "
                "precision: %.4f/%.4f, recall: %.4f/%.4f, F1: %.4f/%.4f, unique pred labels: %s" %
                (losses_teacher.mean(),
                teacher_eval_metrics['micro'][4], teacher_eval_metrics['macro'][4], 
                teacher_eval_metrics['micro'][0], teacher_eval_metrics['macro'][0],
                teacher_eval_metrics['micro'][1], teacher_eval_metrics['macro'][1],
                teacher_eval_metrics['micro'][2], teacher_eval_metrics['macro'][2],
                teacher_eval_metrics['micro'][3], teacher_eval_metrics['macro'][3],
                np.unique(teacher_predicted_classes)))
            callog("******  student Evaluation metrics  ******")
            callog("Mean (micro) Evaluation metrics (micro/macro), loss: %.7f, iou: %.4f/%.4f, accuracy: %.4f/%.4f, "
                "precision: %.4f/%.4f, recall: %.4f/%.4f, F1: %.4f/%.4f, unique pred labels: %s" %
                (losses_student.mean(),
                student_eval_metrics['micro'][4], student_eval_metrics['macro'][4], 
                student_eval_metrics['micro'][0], student_eval_metrics['macro'][0],
                student_eval_metrics['micro'][1], student_eval_metrics['macro'][1],
                student_eval_metrics['micro'][2], student_eval_metrics['macro'][2],
                student_eval_metrics['micro'][3], student_eval_metrics['macro'][3],
                np.unique(student_predicted_classes)))
            callog(
                "-----------------------------------------------------------------------------------------------------------------------------------------------------------------")
            ioussss.append(student_eval_metrics['macro'][4])
        callog(f"AUC:{str(sum(ioussss)/len(ioussss))}")
        with open(os.path.join(save_path,"metrics.txt"), "a", encoding="utf-8") as file:
            file.write(f"AUC:{str(sum(ioussss)/len(ioussss))}"+"Mean (micro) Evaluation metrics (micro/macro), loss: %.7f, iou: %.4f/%.4f, accuracy: %.4f/%.4f, "
                "precision: %.4f/%.4f, recall: %.4f/%.4f, F1: %.4f/%.4f, unique pred labels: %s \n" %
                (losses_student.mean(),
                student_eval_metrics['micro'][4], student_eval_metrics['macro'][4], 
                student_eval_metrics['micro'][0], student_eval_metrics['macro'][0],
                student_eval_metrics['micro'][1], student_eval_metrics['macro'][1],
                student_eval_metrics['micro'][2], student_eval_metrics['macro'][2],
                student_eval_metrics['micro'][3], student_eval_metrics['macro'][3],
                np.unique(student_predicted_classes)))
        # return sum(ioussss)/len(ioussss)
        return ioussss[-1]
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
        checkpoint = load_from_checkpoint(net, checkpoint, device=device)
        callog(f"successfully load from checkpoint {checkpoint}")
        # checkpoint, train_params = load_new(net, checkpoint, partial_restore=False,teaorstu='teacher',frozen=True,device=device)
        # callog(f"successfully load from checkpoint {checkpoint}, trainable params: {train_params}")

    callog(f"current learn rate: {lr}")

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

    BEST_IOU = 0

    net.train()
    for epoch in range(start_epoch, start_epoch + num_epochs):  # loop over the dataset multiple times
        for step, sample in enumerate(dataloaders['train']):
            abs_step = start_global + (epoch - start_epoch) * num_steps_train + step
            teacher_x, student_x, ground_truth, loss, loss_teacher, loss_student, mse_loss, dice_loss = train_step(net, sample, loss_fn, optimizer, device, loss_input_fn=loss_input_fn)
            if len(ground_truth) == 2:
                labels, unk_masks = ground_truth
            else:
                labels = ground_truth
                unk_masks = None
            # print batch statistics ----------------------------------------------------------------------------------#
            if abs_step % train_metrics_steps == 0:
                teacher_x = teacher_x.permute(0, 3, 1, 2)
                student_x = student_x.permute(0, 3, 1, 2)
                teacher_batch_metrics = get_mean_metrics(
                    logits=teacher_x, labels=labels, unk_masks=unk_masks, n_classes=num_classes, loss=loss_teacher, epoch=epoch,
                    step=step)
                student_batch_metrics = get_mean_metrics(
                    logits=student_x, labels=labels, unk_masks=unk_masks, n_classes=num_classes, loss=loss_student, epoch=epoch,
                    step=step)
                batch_metrics = {
                    "teacher_Accuracy": teacher_batch_metrics["Accuracy"], "teacher_Precision": teacher_batch_metrics["Precision"],
                    "teacher_Recall": teacher_batch_metrics["Recall"], "teacher_F1": teacher_batch_metrics["F1"],
                    "teacher_IOU": teacher_batch_metrics["IOU"], "teacher_Loss": teacher_batch_metrics["Loss"],
                    "student_Accuracy": student_batch_metrics["Accuracy"], "student_Precision": student_batch_metrics["Precision"],
                    "student_Recall": student_batch_metrics["Recall"], "student_F1": student_batch_metrics["F1"],
                    "student_IOU": student_batch_metrics["IOU"], "student_Loss": student_batch_metrics["Loss"],
                    "total_loss": float(loss.detach().cpu().numpy()), "mse_loss": float(mse_loss.detach().cpu().numpy()),
                    "dice_loss": float(dice_loss.detach().cpu().numpy())
                }
                callog("abs_step: %d, epoch: %d, step: %5d, total_loss: %.7f, loss_teacher: %.7f, loss_student: %.7f, mse_loss: %.7f, dice_loss:%.7f, batch_teacher_iou: %.4f, "
                      "batch_student_iou: %.4f, lr: %.7f" %
                      (abs_step, epoch, step + 1, batch_metrics['total_loss'], batch_metrics['teacher_Loss'], batch_metrics['student_Loss'], 
                       batch_metrics['mse_loss'], batch_metrics['dice_loss'], batch_metrics['teacher_IOU'], batch_metrics['student_IOU'], optimizer.param_groups[0]["lr"]))

            if abs_step % save_steps == 0:
                if len(local_device_ids) > 1:
                    torch.save(net.module.state_dict(), "%s/%depoch_%dstep.pth" % (save_path, epoch, abs_step))
                else:
                    torch.save(net.state_dict(), "%s/%depoch_%dstep.pth" % (save_path, epoch, abs_step))
            
            # evaluate model ------------------------------------------------------------------------------------------#
            if abs_step % eval_steps == 0:
                eval_metrics = evaluate_test(net, dataloaders['eval'], loss_fn, config, device)
                if eval_metrics > BEST_IOU:
                    if len(local_device_ids) > 1:
                        torch.save(net.module.state_dict(), "%s/best.pth" % (save_path))
                    else:
                        torch.save(net.state_dict(), "%s/best.pth" % (save_path))
                    BEST_IOU = eval_metrics

                net.train()

        scheduler.step_update(abs_step)
    
    callog(f"-----------Eval dataset BestIOU: {BEST_IOU}")

    # load the best checkpoint and evaluate on test dataset -------------------------------------------------------#
    checkpoint_new = os.path.join(save_path, 'best.pth')
    saved_net = torch.load(checkpoint_new, map_location=device)
    new_saved_net = {}
    # {f"module.{k}": v for k, v in saved_net_student.items()}
    for key, value in saved_net.items():
        new_key = f"module.{key}"  # 关键：添加module.前缀
        new_saved_net[new_key] = value
    net.load_state_dict(new_saved_net, strict=True)
    callog(f"successfully load from checkpoint {checkpoint_new}")
    test_metrics = evaluate_test(net, dataloaders['test'], loss_fn, config, device)
    callog(f"-----------Test dataset BestIOU: {test_metrics}")

def creat_log(log_name):
    logger = logging.getLogger('training_logger')
    logger.setLevel(logging.DEBUG)  # 设置日志级别
    file_handler = logging.FileHandler(log_name)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

def callog(message):
    logger.info(message)
    # print(message)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--config', help='configuration (.yaml) file to use')
    parser.add_argument('--device', default='0,1,2,3,4,5,6,7', type=str,help='gpu ids to use')
    parser.add_argument('--save_path', type=str,help='save_path')
    parser.add_argument('--log_file', default='new_log.log', type=str, help='log_file')
    parser.add_argument('--lin', action='store_true',help='train linear classifier only')
    parser.add_argument('--describe', default='This is a teacher student net.', type=str, help='describe the training process')

    args = parser.parse_args()
    config_file = args.config
    
    device_ids = [int(d) for d in args.device.split(',')]
    lin_cls = args.lin

    device = get_device(device_ids, allow_cpu=False)

    config = read_yaml(config_file)
    config['local_device_ids'] = device_ids
    config['CHECKPOINT']['save_path'] = args.save_path
    config['About train'] = args.describe
    max_seq_len = int(config['MODEL']['max_seq_len'])
    config['MODEL']['max_seq_len'] = max_seq_len
    config['DATASETS']['train']['max_seq_len'] = max_seq_len
    config['DATASETS']['eval']['max_seq_len'] = max_seq_len
    config['DATASETS']['test']['max_seq_len'] = max_seq_len

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    log_name = os.path.join(args.save_path,'train.log')
    logger = creat_log(log_name=log_name)
    callog(args.device)
    
    

    dataloaders = get_dataloaders(config)
    # first_batch = next(iter(dataloaders['train']))
    # print(first_batch['inputs'].shape,first_batch['labels'].shape)

    net = get_model(config, device)

    train_and_evaluate(net, dataloaders, config, device,args.save_path)
    callog("Training finished successfully!")