from __future__ import print_function
import argparse
import os.path
import os
import logging
import time
import datetime
import torch
import torch.optim as optim
import numpy as np
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchtoolbox.tools import mixup_data, mixup_criterion
from core.datasets.image_list import ImageList
from core.models.network import ResNetFc,EfficientNetFc
from core.active.active import UCCDA_active, RAND_active, self_training
from core.utils.utils import set_random_seed, mkdir, momentum_update
from core.datasets.transforms import build_transform
from core.active.loss import  EnergyLoss
from core.utils.metric_logger import MetricLogger
from core.utils.logger import setup_logger
from core.config import cfg
import torch.nn.functional as F
from torch.autograd import Variable
from core.models import Consensus_loss
scaler = torch.cuda.amp.GradScaler()
autocast = torch.cuda.amp.autocast
cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def test(model, test_loader):
    start_test = True
    model.eval()
    with torch.no_grad():
        for batch_idx, test_data in enumerate(test_loader):
            img, labels = test_data['img0'], test_data['label']
            img = img.cuda()
            outputs = model(img, return_feat=False)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    _, predict = torch.min(all_output, 1)
    acc = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0]) * 100

    return acc


def train(cfg, task):
    logger = logging.getLogger("UCCDA.trainer")

    use_cuda = True if torch.cuda.is_available() else False

    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}

    # prepare data
    source_transform = build_transform(cfg, is_train=True, choices=cfg.INPUT.SOURCE_TRANSFORMS)
    target_transform = build_transform(cfg, is_train=True, choices=cfg.INPUT.TARGET_TRANSFORMS)
    test_transform = build_transform(cfg, is_train=False, choices=cfg.INPUT.TEST_TRANSFORMS)

    src_train_ds = ImageList(os.path.join(cfg.DATASET.ROOT+cfg.DATASET.NAME, cfg.DATASET.SOURCE_TRAIN_DOMAIN),
                             transform=source_transform)
    src_train_loader = DataLoader(src_train_ds, batch_size=cfg.DATALOADER.SOURCE.BATCH_SIZE, shuffle=True,
                                  drop_last=True, **kwargs)

    tgt_unlabeled_ds = ImageList(os.path.join(cfg.DATASET.ROOT+cfg.DATASET.NAME, cfg.DATASET.TARGET_TRAIN_DOMAIN),
                                 transform=target_transform)
    tgt_unlabeled_loader = DataLoader(tgt_unlabeled_ds, batch_size=cfg.DATALOADER.TARGET.BATCH_SIZE, shuffle=True,
                                      drop_last=True, **kwargs)
    tgt_unlabeled_loader_full = DataLoader(tgt_unlabeled_ds, batch_size=cfg.DATALOADER.TARGET.BATCH_SIZE,
                                              shuffle=True, drop_last=False, **kwargs)

    tgt_test_ds = ImageList(os.path.join(cfg.DATASET.ROOT+cfg.DATASET.NAME, cfg.DATASET.TARGET_VAL_DOMAIN),
                            transform=test_transform)
    tgt_test_loader = DataLoader(tgt_test_ds, batch_size=cfg.DATALOADER.TEST.BATCH_SIZE, shuffle=False, **kwargs)

    # active target dataset & loader
    tgt_selected_ds = ImageList(empty=True,
                                transform=source_transform)
    tgt_selected_loader = DataLoader(tgt_selected_ds, batch_size=cfg.DATALOADER.SOURCE.BATCH_SIZE,
                                     shuffle=True, drop_last=False, **kwargs)

    # model
    model = ResNetFc(class_num=cfg.DATASET.NUM_CLASS, cfg=cfg).cuda()
    # model = EfficientNetFc(class_num=cfg.DATASET.NUM_CLASS, cfg=cfg).cuda()

    #Consensus_dictionaryLoss
    domain_loss = Consensus_loss.Consensus_dictionaryLoss(nav_t=1, beta=0, num_classes=cfg.DATASET.NUM_CLASS, device=device, s_par=0.5).to(device)
    # optimizer
    optimizer = optim.Adadelta(model.parameters_list(cfg.OPTIM.LR), lr=cfg.OPTIM.LR)

    # unsupervised energy alignment bound loss
    Assist_criterion = EnergyLoss(cfg)
    # total number of target samples
    totality = tgt_unlabeled_ds.__len__()

    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    start_training_time = time.time()
    end = time.time()

    final_acc = 0.
    final_model = None
    all_epoch_result = []
    all_selected_images = None
    best_acc = 0
    for epoch in range(1, cfg.TRAINER.MAX_EPOCHS + 1):

        model.train()
        iter_per_epoch = max(len(src_train_loader), len(tgt_unlabeled_loader))
        for batch_idx in range(iter_per_epoch):
            data_time = time.time() - end

            if batch_idx % len(src_train_loader) == 0:
                src_iter = iter(src_train_loader)
            if batch_idx % len(tgt_unlabeled_loader) == 0:
                tgt_unlabeled_iter = iter(tgt_unlabeled_loader)
            if not tgt_selected_ds.empty:
                if batch_idx % len(tgt_selected_loader) == 0:
                    tgt_selected_iter = iter(tgt_selected_loader)

            src_data = src_iter.next()
            tgt_unlabeled_data = tgt_unlabeled_iter.next()

            src_img, src_lbl = src_data['img0'], src_data['label']
            src_img, src_lbl = src_img.to(device,non_blocking=True), src_lbl.to(device,non_blocking=True)



            tgt_unlabeled_img = tgt_unlabeled_data['img']
            tgt_unlabeled_img = tgt_unlabeled_img.cuda()
            with autocast():
                optimizer.zero_grad()

                total_loss = 0

                # supervised loss on label source data
                src_out= model(src_img, return_feat=False)
                src_out_smoothing=(1-0.1)*src_out+0.1*(1.0/cfg.DATASET.NUM_CLASS)
                Bce_loss = 10*F.cross_entropy(1-src_out_smoothing,src_lbl)
                total_loss += Bce_loss
                meters.update(Bce_loss=Bce_loss.item())

                tgt_feature, tgt_unlabeled_out = model(tgt_unlabeled_img, return_feat=True)
                Consensus_dictionaryLoss_s = model.classifier.weight.data.clone()
                transfer_loss = domain_loss(Consensus_dictionaryLoss_s, tgt_feature)
                total_loss += transfer_loss
                meters.update(Consensus_dictionaryLoss=transfer_loss.item())

                if cfg.TRAINER.ENERGY_ALIGN_WEIGHT > 0:


                    # energy alignment loss on unlabeled target data
                    with torch.no_grad():
                        # free energy of samples
                        output_div_t = -1.0 * cfg.TRAINER.ENERGY_BETA * src_out
                        output_logsumexp = torch.logsumexp(output_div_t, dim=1, keepdim=False)
                        free_energy = -1.0 * output_logsumexp / cfg.TRAINER.ENERGY_BETA

                        src_batch_free_energy = free_energy.mean().detach()
                        # init global mean free energy
                        if epoch == 1 and batch_idx == 0:
                            global_mean = src_batch_free_energy


                        # update global mean free energy
                        global_mean = momentum_update(global_mean, src_batch_free_energy)

                    Assist_fealoss = Assist_criterion(inputs=tgt_unlabeled_out, bound=global_mean)
                    total_loss += cfg.TRAINER.ENERGY_ALIGN_WEIGHT * Assist_fealoss
                    meters.update(Assist_fealoss=(cfg.TRAINER.ENERGY_ALIGN_WEIGHT * Assist_fealoss).item())

                # supervised loss on selected target data
                if not tgt_selected_ds.empty:
                    tgt_selected_data = tgt_selected_iter.next()
                    tgt_selected_img, tgt_selected_lbl = tgt_selected_data['img0'], tgt_selected_data['label']
                    tgt_selected_img, tgt_selected_lbl = tgt_selected_img.cuda(), tgt_selected_lbl.cuda()
                    if tgt_selected_img.size(0) == 1:
                        # avoid bs=1, can't pass through BN layer
                        tgt_selected_img = torch.cat((tgt_selected_img, tgt_selected_img), dim=0)
                        tgt_selected_lbl = torch.cat((tgt_selected_lbl, tgt_selected_lbl), dim=0)
#

                    tgt_selected_out = model(tgt_selected_img, return_feat=False)
                    selected_nll_loss = 10*F.cross_entropy(1 - tgt_selected_out, tgt_selected_lbl)
                    total_loss += selected_nll_loss
                    meters.update(selected_nll_loss=selected_nll_loss.item())


                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()

                batch_time = time.time() - end
                end = time.time()
                meters.update(time=batch_time, data=data_time)
                eta_seconds = meters.time.global_avg * (iter_per_epoch * cfg.TRAINER.MAX_EPOCHS - batch_idx * epoch)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                if batch_idx % cfg.TRAIN.PRINT_FREQ == 0:
                    logger.info(
                        meters.delimiter.join(
                            [
                                "eta: {eta}",
                                "task: {task}",
                                "epoch: {epoch}",
                                f"[iter: {batch_idx}/{iter_per_epoch}]",
                                "{meters}",

                            ]
                        ).format(
                            task=task,
                            eta=eta_string,
                            epoch=epoch,
                            meters=str(meters),

                        )
                    )

        # test every 5 epoch
        if epoch % 5 == 0:
            testacc = test(model, tgt_test_loader)
            if testacc>best_acc:
                best_acc=testacc
            logger.info('Task: {} Test Epoch: {} testacc: {:.2f}'.format(task, epoch, testacc))
            logger.info('Task: {} Test Epoch: {} best_acc: {:.2f}'.format(task, epoch, best_acc))
            all_epoch_result.append({'epoch': epoch, 'acc': best_acc})
            if epoch == cfg.TRAINER.MAX_EPOCHS:
                final_model = model.state_dict()
                final_acc = best_acc

        # active selection rounds
        if epoch in cfg.TRAINER.ACTIVE_ROUND:
            logger.info('Task: {} Active Epoch: {}'.format(task, epoch))
            if cfg.TRAINER.NAME == 'RAND':
                active_samples = RAND_active(tgt_unlabeled_ds=tgt_unlabeled_ds,
                                             tgt_selected_ds=tgt_selected_ds,
                                             active_ratio=0.01,
                                             totality=totality)
            elif cfg.TRAINER.NAME == 'UCCDA':
                active_samples2 = UCCDA_active(tgt_unlabeled_loader_full=tgt_unlabeled_loader_full,
                                             tgt_unlabeled_ds=tgt_unlabeled_ds,
                                             tgt_selected_ds=tgt_selected_ds,
                                             active_ratio=0.01,
                                             totality=totality,
                                             model=model,
                                             cfg=cfg)

                active_samples = self_training(tgt_unlabeled_loader_full=tgt_unlabeled_loader_full,
                                             tgt_unlabeled_ds=tgt_unlabeled_ds,
                                             tgt_selected_ds=tgt_selected_ds,
                                             active_ratio=0.002,
                                             totality=totality,
                                             model=model,
                                             cfg=cfg)

            # record all selected target images
            if all_selected_images is None:
                all_selected_images = np.concatenate((active_samples2,active_samples),axis=0)
            else:
                all_active_samples=np.concatenate((active_samples2,active_samples),axis=0)
                all_selected_images = np.concatenate((all_selected_images, all_active_samples), axis=0)


    # record all selected images
    ckt_path = os.path.join(cfg.OUTPUT_DIR, cfg.DATASET.NAME, task)
    mkdir(ckt_path)
    torch.save(all_selected_images, os.path.join(ckt_path, "all_selected_images.pth"))
    torch.save(final_model, os.path.join(ckt_path, "final_model_{}.pth".format(task)))

    # record results for test epochs
    with open(os.path.join(ckt_path, 'all_epoch_result.csv'), 'w') as handle:
        for i, rec in enumerate(all_epoch_result):
            if i == 0:
                handle.write(','.join(list(rec.keys())) + '\n')
            line = [str(rec[key]) for key in rec.keys()]
            handle.write(','.join(line) + '\n')

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / ep)".format(
            total_time_str, total_training_time / cfg.TRAINER.MAX_EPOCHS
        )
    )

    return task, final_acc


def main():
    parser = argparse.ArgumentParser(description='PyTorch Activate Domain Adaptation')
    parser.add_argument('--cfg',
                        default='',
                        metavar='FILE',
                        help='path to config file',
                        type=str)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    output_dir = os.path.join(cfg.OUTPUT_DIR, cfg.DATASET.NAME)
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("UCCDA", output_dir, 0)
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.SEED >= 0:
        print('Setting fixed seed: {}'.format(cfg.SEED))
        set_random_seed(cfg.SEED)

    cudnn.deterministic = True

    all_task_result = []
    for source in cfg.DATASET.SOURCE_DOMAINS:
        for target in cfg.DATASET.TARGET_DOMAINS:
            if source != target:
                cfg.DATASET.SOURCE_TRAIN_DOMAIN = os.path.join(source + '_train.txt')
                cfg.DATASET.TARGET_TRAIN_DOMAIN = os.path.join(target + '_train.txt')
                cfg.DATASET.TARGET_VAL_DOMAIN = os.path.join(target + '_test.txt')

                cfg.freeze()
                task, final_acc = train(cfg, task=source + '2' + target)
                all_task_result.append({'task': task, 'final_acc': final_acc})
                cfg.defrost()

    # record all results for all tasks
    with open(os.path.join(output_dir, 'all_task_result.csv'), 'w') as handle:
        for i, rec in enumerate(all_task_result):
            if i == 0:
                handle.write(','.join(list(rec.keys())) + '\n')
            line = [str(rec[key]) for key in rec.keys()]
            handle.write(','.join(line) + '\n')


if __name__ == '__main__':
    main()
