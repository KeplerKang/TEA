import torch
import torch.nn.functional as F
import numpy as np

def cams_to_mask(cams, cls_label, fg_thre=0.35, bg_thre=0.2):
    tensor = torch.zeros_like(cams)
    cls_label_expanded = cls_label.unsqueeze(-1).unsqueeze(-1)

    mask = cls_label_expanded == 1

    tensor = torch.where(mask, cams, tensor)
    # tensor = F.softmax(tensor, dim=1)
    cams_value, pseudo_label = torch.max(tensor, dim=1)
    pseudo_label += 1

    pseudo_label[cams_value<=fg_thre] = 255 # ignore index
    pseudo_label[cams_value<=bg_thre] = 0

    return pseudo_label



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count

