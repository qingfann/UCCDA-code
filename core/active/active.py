import random
import math
import numpy as np
import torch


def RAND_active(tgt_unlabeled_ds, tgt_selected_ds, active_ratio, totality):
    length = len(tgt_unlabeled_ds.samples)
    index = random.sample(range(length), round(totality * active_ratio))

    tgt_selected_ds.add_item(tgt_unlabeled_ds.samples[index])
    tgt_unlabeled_ds.remove_item(index)


def UCCDA_active(tgt_unlabeled_loader_full, tgt_unlabeled_ds, tgt_selected_ds, active_ratio, totality, model, cfg):
    model.eval()
    first_stat = list()
    with torch.no_grad():
        for _, data in enumerate(tgt_unlabeled_loader_full):
            tgt_img, tgt_lbl = data['img'], data['label']
            tgt_path, tgt_index = data['path'], data['index']
            tgt_img, tgt_lbl = tgt_img.cuda(), tgt_lbl.cuda()

            tgt_out = model(tgt_img, return_feat=False)

            min2 = torch.topk(tgt_out, k=2, dim=1, largest=False).values
            mvsm_uncertainty = min2[:, 0] - min2[:, 1]

            output_div_t = -1.0 * tgt_out / cfg.TRAINER.ENERGY_BETA
            output_logsumexp = torch.logsumexp(output_div_t, dim=1, keepdim=False)
            free_energy = -1.0 * cfg.TRAINER.ENERGY_BETA * output_logsumexp

            for i in range(len(free_energy)):
                first_stat.append([tgt_path[i], tgt_lbl[i].item(), tgt_index[i].item(),
                                   mvsm_uncertainty[i].item(), free_energy[i].item()])

    first_sample_ratio = 0.5
    first_sample_num = math.ceil(totality * first_sample_ratio)
    second_sample_ratio = active_ratio / 0.5
    second_sample_num = math.ceil(first_sample_num * second_sample_ratio)

    first_stat = sorted(first_stat, key=lambda x: x[4], reverse=True)
    second_stat = first_stat[:first_sample_num]

    second_stat = sorted(second_stat, key=lambda x: x[3], reverse=True)
    second_stat = np.array(second_stat)

    active_samples = second_stat[:second_sample_num, 0:2, ...]
    candidate_ds_index = second_stat[:second_sample_num, 2, ...]
    candidate_ds_index = np.array(candidate_ds_index, dtype=np.int)

    tgt_selected_ds.add_item(active_samples)
    tgt_unlabeled_ds.remove_item(candidate_ds_index)

    return active_samples

def self_training(tgt_unlabeled_loader_full, tgt_unlabeled_ds, tgt_selected_ds, active_ratio, totality, model, cfg):
    model.eval()
    first_stat = list()
    with torch.no_grad():
        for _, data in enumerate(tgt_unlabeled_loader_full):
            tgt_img, tgt_lbl = data['img'], data['label']
            tgt_path, tgt_index = data['path'], data['index']
            tgt_img, tgt_lbl = tgt_img.cuda(), tgt_lbl.cuda()

            tgt_out = model(tgt_img, return_feat=False)
            pseudo_lbl=torch.argmin(tgt_out,dim=-1)

            min2 = torch.topk(tgt_out, k=2, dim=1, largest=False).values
            mvsm_uncertainty = min2[:, 0] - min2[:, 1]

            output_div_t = -1.0 * tgt_out / cfg.TRAINER.ENERGY_BETA
            output_logsumexp = torch.logsumexp(output_div_t, dim=1, keepdim=False)
            free_energy = -1.0 * cfg.TRAINER.ENERGY_BETA * output_logsumexp
            for i in range(len(free_energy)):
                first_stat.append([tgt_path[i], pseudo_lbl[i].item(), tgt_index[i].item(),
                                   mvsm_uncertainty[i].item(), free_energy[i].item()])

    first_sample_ratio = 0.2
    first_sample_num = math.ceil(totality * first_sample_ratio)
    second_sample_ratio = active_ratio / 0.2
    second_sample_num = math.ceil(first_sample_num * second_sample_ratio)

    first_stat = sorted(first_stat, key=lambda x: x[4], reverse=False)
    second_stat = first_stat[:first_sample_num]

    second_stat = sorted(second_stat, key=lambda x: x[3], reverse=False)
    second_stat = np.array(second_stat)

    active_samples = second_stat[:second_sample_num, 0:2, ...]
    candidate_ds_index = second_stat[:second_sample_num, 2, ...]
    candidate_ds_index = np.array(candidate_ds_index, dtype=np.int)

    tgt_selected_ds.add_item(active_samples)
    tgt_unlabeled_ds.remove_item(candidate_ds_index)

    return active_samples
