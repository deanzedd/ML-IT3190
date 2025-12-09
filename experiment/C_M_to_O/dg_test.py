import sys
sys.path.append('../../')
import pandas as pd
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import functional as F
from config import config
from utils.utils import sample_frames
from utils.dataset import YunpeiDataset
from utils.utils import AverageMeter, accuracy, draw_roc
from utils.statistic import get_EER_states, get_HTER_at_thr, calculate, calculate_threshold
from sklearn.metrics import roc_auc_score, roc_curve, auc
from models.DGFAS import DG_model
import shutil
 
# Determine device: use GPU if configured and available, else CPU
gpus = getattr(config, 'gpus', '')
if gpus and torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    device = 'cuda'
else:
    device = 'cpu'
    print(f"Using device: {device}")

def test(test_dataloader, model, threshold):
    valid_top1 = AverageMeter()
    prob_dict = {}
    label_dict = {}
    model.eval()
    output_dict_tmp = {}
    target_dict_tmp = {}
    number = 0
    with torch.no_grad():
        for iter, (input, target, videoID) in enumerate(test_dataloader):
            input = Variable(input).to(device)
            target = Variable(torch.from_numpy(np.array(target)).long()).to(device)
            cls_out, _ = model(input, config.norm_flag)
            prob = F.softmax(cls_out, dim=1).cpu().data.numpy()[:, 1]
            label = target.cpu().data.numpy()
            videoID = videoID.cpu().data.numpy()
            for i in range(len(prob)):
                if (videoID[i] in prob_dict.keys()):
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
                    number += 1
                    if (number % 100 == 0):
                        print('**Testing** ', number, ' photos done!')
    print('**Testing** ', number, ' photos done!')
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
            mode_result = torch.mode(targets)
            avg_single_video_target = mode_result.values if hasattr(mode_result, 'values') else mode_result[0]
        acc_valid = accuracy(avg_single_video_output, avg_single_video_target.unsqueeze(0), topk=(1,))
        valid_top1.update(acc_valid[0])

    cur_EER_valid, threshold, FRR_list, FAR_list = get_EER_states(prob_list, label_list)
    ACC_threshold = calculate_threshold(prob_list, label_list, threshold)
    # roc_auc_score is undefined when only one class is present in labels
    if len(np.unique(label_list)) < 2:
        auc_score = float('nan')
    else:
        auc_score = roc_auc_score(label_list, prob_list)
        draw_roc(FRR_list, FAR_list, auc_score)
    cur_HTER_valid = get_HTER_at_thr(prob_list, label_list, threshold)
    return [valid_top1.avg, cur_EER_valid, cur_HTER_valid, auc_score, ACC_threshold, threshold]

def main():
    net = DG_model(config.model).to(device)
    test_data = sample_frames(flag=2, num_frames=config.tgt_test_num_frames, dataset_name=config.tgt_data)
    test_dataloader = DataLoader(YunpeiDataset(test_data, train=False), batch_size=1, shuffle=False)
    print('\n')
    print("**Testing** Get test files done!")
    # load model
    ckpt_path = os.path.join(config.best_model_path, config.tgt_best_model_name)
    if not os.path.exists(ckpt_path):
        print(f"\n[INFO] Checkpoint not found at {ckpt_path}")
        print("[INFO] Skipping test. Run training first to generate checkpoint.")
        return
    net_ = torch.load(ckpt_path, weights_only=False)
    net.load_state_dict(net_["state_dict"])
    threshold = net_["threshold"]
    # test model
    test_args = test(test_dataloader, net, threshold)
    print('\n===========Test Info===========\n')
    print(config.tgt_data, 'Test acc: %5.4f' %(test_args[0]))
    print(config.tgt_data, 'Test EER: %5.4f' %(test_args[1]))
    print(config.tgt_data, 'Test HTER: %5.4f' %(test_args[2]))
    print(config.tgt_data, 'Test AUC: %5.4f' % (test_args[3]))
    print(config.tgt_data, 'Test ACC_threshold: %5.4f' % (test_args[4]))
    print('\n===============================\n')

if __name__ == '__main__':
    main()