#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File     :     train.py
@Time     :     2025/11/30 15:29:55
@Author   :     Louis Swift
@Desc     :     
'''

import os 
import hydra
import torch
import logging
import numpy as np 
from omegaconf import OmegaConf
from torch.optim import Adam,SGD
import torchvision.utils as vutils
from utils.dataset import CddDataset,VOCDataset
from torch.utils.data import DataLoader
from utils.metric import calc_psnr,calc_ssim
from core import build_enhancer,build_dip_loss 
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR,ExponentialLR
from utils.misc import set_random_seed,get_model_info,get_exp_info

os.environ['HYDRA_FULL_ERROR'] = '1'
logger = logging.getLogger(__name__)

@hydra.main(config_path='config', config_name='config',version_base=None)
def main(config:OmegaConf):
    # ----------------------
    # 1. 基础配置

    # 权重保存目录
    ckpt_dir = config.exp.exp_dir + os.sep + 'checkpoints'
    os.makedirs(ckpt_dir,exist_ok=True)

    # 日志文件目录
    # tensorboard 目录
    tb_dir = config.exp.exp_dir + os.sep + 'tb_logs'
    os.makedirs(tb_dir,exist_ok=True)
    writer = SummaryWriter(tb_dir)
    
    logger.info(f'Exp Directory : {config.exp.exp_dir}.')
    logger.info(f'Ckpt Directory : {ckpt_dir}.')
    logger.info(f'Tensorboard Directory : {tb_dir}.')

    # 固定随机种子
    set_random_seed(config.exp.seed)
    
    # 实验配置打印
    logger.info(f'Exp Config:\n' + get_exp_info(config))

    # ----------------------
    # 2. 模型配置
    net = build_enhancer(config.enhancer)

    net.to(config.exp.device)
    net.train()

    logger.info("Enhancer Info:\n" + get_model_info(net))
    
    loss = build_dip_loss(config.enhancer.loss)
    loss.to(config.exp.device)

    # ----------------------
    # 3. 数据集配置
    if config.dataset.dataset_name == 'cdd':
        train_dataset = CddDataset(data_dir=config.dataset.train_dir,phase='train',**config.dataset)
        val_dataset   = CddDataset(data_dir=config.dataset.val_dir,phase='val',**config.dataset)
    else:
        train_dataset = VOCDataset(txt_path=config.dataset.train_txt_path,phase='train',**config.dataset)
        val_dataset   = VOCDataset(txt_path=config.dataset.val_txt_path,phase='val',**config.dataset)
    
    train_dataloader  = DataLoader(
                train_dataset,
                batch_size=config.exp.batch_size,
                shuffle=True,
                num_workers=config.exp.num_workers
        )
    
    val_dataloader    = DataLoader(
                val_dataset,
                batch_size=config.exp.batch_size,
                shuffle=False,
                num_workers=config.exp.num_workers
        )
    
    # ----------------------
    # 4. 优化器与调度器
    optimizer = Adam(
        net.parameters(),
        lr=config.exp.lr,
        # momentum=config.exp.momentum,
        weight_decay=config.exp.lr_decay
    )

    scheduler = ExponentialLR(optimizer,gamma=config.exp.gamma)

    # ----------------------
    # 5. 训练
    global_cur_iter = 0 
    val_epoch_cnt   = 0 # 用于记录 val 的iteration
    total_iters_per_ep_train = len(train_dataset)  // config.exp.batch_size
    total_iters_per_ep_val   = len(val_dataset) // config.exp.batch_size

    for cur_ep in range(config.exp.max_epoch):

        for cur_iter,(damage,gt) in enumerate(train_dataloader):
            global_cur_iter = cur_ep * total_iters_per_ep_train + cur_iter + 1
            damage = damage.to(config.exp.device)
            gt = gt.to(config.exp.device)

            pred_lst,gate = net(damage)

            # 损失计算
            total_loss , loss_dict = loss(pred_lst,gt,damage)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            # 记录实验数据
            writer.add_scalar('Train/l1-loss',loss_dict['l1'].item(),global_cur_iter)
            writer.add_scalar('Train/msssim-loss',loss_dict['msssim'].item(),global_cur_iter)
            writer.add_scalar('Train/color-loss',loss_dict['color'].item(),global_cur_iter)
            writer.add_scalar('Train/texture-loss',loss_dict['texture'].item(),global_cur_iter)
            writer.add_scalar('Train/contrast-loss',loss_dict['contrast'].item(),global_cur_iter)
            writer.add_scalar('Train/total-loss',loss_dict['dip'].item(),global_cur_iter)

            logger.info(f'Epoch:[{cur_ep}][{cur_iter}/{total_iters_per_ep_train}], Loss:[{total_loss.item():.3f}].')

        # 保存模型权重
        if (cur_ep + 1) % config.exp.save_model_interval == 0 or (cur_ep + 1) == config.exp.max_epoch:
            state_dict = net.state_dict()
            torch.save(state_dict, ckpt_dir + os.sep + f'model_epoch_{cur_ep + 1:d}.pth')

            logger.info(f"Model checkpoints saved at {ckpt_dir + os.sep + f'model_epoch_{ global_cur_iter :d}.pth'}.")


        # 验证阶段
        if (cur_ep + 1) % config.exp.val_epoch == 0:

            logger.info('Start to evaluation.')
            psnr_lst = [] 
            ssim_lst = [] 
            loss_lst = [] 

            with torch.no_grad():
                for cur_iter,(damage,gt) in enumerate(val_dataloader):

                    global_cur_iter = val_epoch_cnt * total_iters_per_ep_val + cur_iter + 1

                    damage = damage.to(config.exp.device)
                    gt = gt.to(config.exp.device)

                    pred_lst,gate = net(damage)

                    # 损失计算 
                    total_loss , loss_dict = loss(pred_lst,gt,damage)

                    logger.info(f'Evaluation:[{cur_iter}/{total_iters_per_ep_val}], Loss:[{total_loss.item():.3f}].')

                    psnr = calc_psnr(pred_lst[-1],gt).item()
                    ssim = calc_ssim(pred_lst[-1],gt).item()
                    psnr_lst.append(psnr)
                    ssim_lst.append(ssim)
                    loss_lst.append(total_loss.item())

                    # 记录实验数据
                    writer.add_scalar('Eval/l1-loss',loss_dict['l1'].item(),global_cur_iter)
                    writer.add_scalar('Eval/msssim-loss',loss_dict['msssim'].item(),global_cur_iter)
                    writer.add_scalar('Eval/contrast-loss',loss_dict['contrast'].item(),global_cur_iter)
                    writer.add_scalar('Eval/texture-loss',loss_dict['texture'].item(),global_cur_iter)
                    writer.add_scalar('Eval/color-loss',loss_dict['color'].item(),global_cur_iter)
                    writer.add_scalar('Eval/total-loss',loss_dict['dip'].item(),global_cur_iter)
                            
                    writer.add_scalar('Eval/psnr',psnr,global_cur_iter)
                    writer.add_scalar('Eval/ssim',ssim,global_cur_iter)
                
                val_epoch_cnt += 1

            logger.info(f'Evalation:[{cur_ep}], Mean Loss:[{np.mean(loss_lst):.3f}], PSNR:[{np.mean(psnr_lst):.3f}], SSIM:[{np.mean(ssim_lst):.3f}].')

            writer.add_image('grid/gt',
                vutils.make_grid(gt, normalize=True, scale_each=True),
                cur_ep+1)
            
            writer.add_image('grid/damage',
                vutils.make_grid(damage, normalize=True, scale_each=True),
                cur_ep+1)

            writer.add_image('grid/pred',
                vutils.make_grid(pred_lst[-1], normalize=True, scale_each=True),
                cur_ep+1)

    writer.close()

if __name__ == '__main__':
    main()