import torch

def _build_labels_from_shape_list(shape_list, device, start_label=0):
    labels = []
    for idx, size in enumerate(shape_list):
        lbl = start_label + idx
        labels.append(torch.full((size, 1), lbl, dtype=torch.long, device=device))
    if len(labels) == 0:
        return torch.tensor([], dtype=torch.long, device=device)
    return torch.cat(labels, dim=0).view(-1)

def Real_AdLoss(discriminator_out, criterion, shape_list):
    device = discriminator_out.device
    ad_label = _build_labels_from_shape_list(shape_list, device, start_label=0)
    real_adloss = criterion(discriminator_out, ad_label)
    return real_adloss

def Fake_AdLoss(discriminator_out, criterion, shape_list):
    device = discriminator_out.device
    ad_label = _build_labels_from_shape_list(shape_list, device, start_label=0)
    fake_adloss = criterion(discriminator_out, ad_label)
    return fake_adloss

def AdLoss_Limited(discriminator_out, criterion, shape_list):
    device = discriminator_out.device
    ad_label = _build_labels_from_shape_list(shape_list, device, start_label=0)
    real_adloss = criterion(discriminator_out, ad_label)
    return real_adloss
