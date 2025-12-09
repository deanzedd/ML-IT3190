from utils.utils import AverageMeter, accuracy
from utils.statistic import get_EER_states, get_HTER_at_thr, calculate, calculate_threshold
from sklearn.metrics import roc_auc_score
from torch.autograd import Variable
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

def eval(valid_dataloader, model, norm_flag):
    criterion = nn.CrossEntropyLoss()
    valid_losses = AverageMeter()
    valid_top1 = AverageMeter()
    prob_dict = {}
    label_dict = {}
    model.eval()
    output_dict_tmp = {}
    target_dict_tmp = {}
    with torch.no_grad():
        # infer device from model parameters
        model_device = next(model.parameters()).device
        for iter, (input, target, videoID) in enumerate(valid_dataloader):
            input = Variable(input).to(model_device)
            target = Variable(torch.from_numpy(np.array(target)).long()).to(model_device)
            cls_out, feature = model(input, norm_flag)
            prob = F.softmax(cls_out, dim=1).cpu().data.numpy()[:, 1]
            label = target.cpu().data.numpy()
            videoID = videoID.cpu().data.numpy()
            for i in range(len(prob)):
                if(videoID[i] in prob_dict.keys()):
                    prob_dict[videoID[i]].append(prob[i])
                    label_dict[videoID[i]].append(label[i])
                    output_dict_tmp[videoID[i]].append(cls_out[i].view(1, 2))
                    target_dict_tmp[videoID[i]].append(target[i].view(1))
                else:
                    prob_dict[videoID[i]] = []
                    label_dict[videoID[i]] = []
                    prob_dict[videoID[i]].append(prob[i])
                    label_dict[videoID[i]].append(label[i])
                    output_dict_tmp[videoID[i]] = []
                    target_dict_tmp[videoID[i]] = []
                    output_dict_tmp[videoID[i]].append(cls_out[i].view(1, 2))
                    target_dict_tmp[videoID[i]].append(target[i].view(1))
    prob_list = []
    label_list = []
    for key in prob_dict.keys():
        avg_single_video_prob = sum(prob_dict[key]) / len(prob_dict[key])
        avg_single_video_label = sum(label_dict[key]) / len(label_dict[key])
        prob_list = np.append(prob_list, avg_single_video_prob)
        label_list = np.append(label_list, avg_single_video_label)
        # compute loss and acc for every video
        avg_single_video_output = sum(output_dict_tmp[key]) / len(output_dict_tmp[key])
        # build a single integer target per video (most frequent frame label)
        targets = torch.cat(target_dict_tmp[key], dim=0).view(-1)
        if targets.numel() == 0:
            avg_single_video_target = torch.tensor(0, dtype=torch.long, device=avg_single_video_output.device)
        else:
            # mode returns (values, indices); we want the most common value
            mode_result = torch.mode(targets)
            avg_single_video_target = mode_result.values if hasattr(mode_result, 'values') else mode_result[0]
        # CrossEntropyLoss expects output [N, C] and target [N]
        # avg_single_video_output is [1, 2] (already has batch dim), target is scalar
        loss = criterion(avg_single_video_output, avg_single_video_target.unsqueeze(0))
        acc_valid = accuracy(avg_single_video_output, avg_single_video_target.unsqueeze(0), topk=(1,))
        valid_losses.update(loss.item())
        valid_top1.update(acc_valid[0])
    # roc_auc_score is undefined when `label_list` contains only one class
    if len(np.unique(label_list)) < 2:
        auc_score = float('nan')
    else:
        auc_score = roc_auc_score(label_list, prob_list)

    cur_EER_valid, threshold, _, _ = get_EER_states(prob_list, label_list)
    ACC_threshold = calculate_threshold(prob_list, label_list, threshold)
    cur_HTER_valid = get_HTER_at_thr(prob_list, label_list, threshold)
    return [valid_losses.avg, valid_top1.avg, cur_EER_valid, cur_HTER_valid, auc_score, threshold, ACC_threshold*100]

