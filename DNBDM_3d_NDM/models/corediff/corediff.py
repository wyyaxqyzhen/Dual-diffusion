import os.path as osp
from torch.nn import functional as F
import os
import torch
import torchvision
import argparse
import tqdm
import copy
from utils.measure import *
from utils.loss_function import PerceptualLoss
from utils.ema import EMA

from models.basic_template import TrainTask
from .corediff_wrapper import Network, WeightNet
from .diffusion_modules import Diffusion

import wandb


class corediff(TrainTask):
    @staticmethod
    def build_options():
        parser = argparse.ArgumentParser('Private arguments for training of different methods')
        parser.add_argument("--in_channels", default=1, type=int)
        parser.add_argument("--out_channels", default=1, type=int)
        parser.add_argument("--init_lr", default=2e-4, type=float)

        parser.add_argument('--update_ema_iter', default=10, type=int)
        parser.add_argument('--start_ema_iter', default=2000, type=int)
        parser.add_argument('--ema_decay', default=0.995, type=float)

        parser.add_argument('--T', default=5, type=int)

        parser.add_argument('--sampling_routine', default='ddim', type=str)
        parser.add_argument('--only_adjust_two_step', action='store_true')
        parser.add_argument('--start_adjust_iter', default=1, type=int)

        return parser

    # 实例化模型
    def set_model(self):
        opt = self.opt
        self.ema = EMA(opt.ema_decay)
        self.update_ema_iter = opt.update_ema_iter
        self.start_ema_iter = opt.start_ema_iter
        self.dose = opt.dose
        self.T = opt.T
        self.sampling_routine = opt.sampling_routine
        self.context = opt.context

        denoise_fn = Network(in_channels=opt.in_channels, context=opt.context)

        model = Diffusion(
            denoise_fn=denoise_fn,
            image_size=36,
            timesteps=opt.T,
            context=opt.context
        ).cuda()

        optimizer = torch.optim.Adam(model.parameters(), opt.init_lr)
        ema_model = copy.deepcopy(model)

        self.logger.modules = [model, ema_model, optimizer]
        self.model = model
        self.optimizer = optimizer
        self.ema_model = ema_model
        self.best_val_loss = float('inf')  # 初始化最佳验证损失为无穷大

        self.lossfn = nn.MSELoss()
        self.lossfn_sub1 = nn.MSELoss()

        self.reset_parameters()

    # 前期ema模型直接使用训练的模型，后期更新使用
    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    # 跟新ema模型
    def step_ema(self, n_iter):
        if n_iter < self.start_ema_iter:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    # 定义的训练方法
    def train(self, inputs, n_iter):
        opt = self.opt
        self.model.train()
        self.ema_model.train()
        low_dose, full_dose = inputs
        low_dose, full_dose = low_dose.cuda(), full_dose.cuda()

        ## training process of CoreDiff
        gen_full_dose, x_mix, gen_full_dose_sub1, x_mix_sub1 = self.model(
            low_dose, full_dose, n_iter,
            only_adjust_two_step=opt.only_adjust_two_step,
            start_adjust_iter=opt.start_adjust_iter
        )

        loss = 0.5 * self.lossfn(gen_full_dose, full_dose) + 0.5 * self.lossfn_sub1(gen_full_dose_sub1, full_dose)
        loss.backward()

        if opt.wandb:
            if n_iter == opt.resume_iter + 1:
                wandb.init(project="your wandb project name")

        self.optimizer.step()
        self.optimizer.zero_grad()

        lr = self.optimizer.param_groups[0]['lr']
        loss = loss.item()
        self.logger.train_msg([loss, lr], n_iter)

        if opt.wandb:
            wandb.log({'epoch': n_iter, 'tra_loss': loss})

        if n_iter % self.update_ema_iter == 0:
            self.step_ema(n_iter)
        return loss

    # 定义的验证方法
    @torch.no_grad()
    def val(self, n_iter):
        opt = self.opt
        self.ema_model.eval()
        loss = 0.
        for low_dose, full_dose in tqdm.tqdm(self.test_loader, desc='val'):
            low_dose, full_dose = low_dose.cuda(), full_dose.cuda()

            gen_full_dose, direct_recons, imstep_imgs = self.ema_model.sample(
                batch_size=low_dose.shape[0],
                img=low_dose,
                t=self.T,
                sampling_routine=self.sampling_routine,
                n_iter=n_iter,
                start_adjust_iter=opt.start_adjust_iter,
            )
            loss_core = self.lossfn(gen_full_dose, full_dose)
            loss += loss_core / len(self.test_loader)
        val_loss = loss.item()
        self.logger.val_msg([val_loss], n_iter)

        if opt.wandb:
            wandb.log({'epoch': n_iter, 'val_loss': val_loss})
        return val_loss

    # 测试生成结果图像
    @torch.no_grad()
    def generate_images(self, n_iter):
        opt = self.opt
        self.ema_model.eval()
        low_dose, full_dose = self.test_images

        gen_full_dose, direct_recons, imstep_imgs = self.ema_model.sample(
            batch_size=low_dose.shape[0],
            img=low_dose,
            t=self.T,
            sampling_routine=self.sampling_routine,
            n_iter=n_iter,
            start_adjust_iter=opt.start_adjust_iter,
        )
        # 解包时适配 5D 张量 (B, C, D, H, W)
        b, c, d, h, w = low_dose.size()

        # 拼接生成的 3D 图像，将 low_dose、full_dose 和 gen_full_dose 按维度 1 堆叠
        # 结果形状: (B, 3, D, H, W)
        fake_imgs = torch.stack([low_dose, full_dose, gen_full_dose], dim=1)

        # 调整窗口：确保 transfer_display_window 函数能处理 5D 张量 (B, 3, D, H, W)
        fake_imgs = self.transfer_display_window(fake_imgs)

        # 调整维度顺序并 reshape，适配保存或后续处理
        # 结果形状: (B * D, C, H, W)，将深度维度合并到批次维度
        fake_imgs = fake_imgs.transpose(1, 0).reshape((-1, c, h, w))

        """self.logger.save_image(torchvision.utils.make_grid(fake_imgs, nrow=3),
                               n_iter, 'test_{}_{}'.format(self.dose, self.sampling_routine) + '_' + opt.test_dataset)"""

        self.logger.save_tifimage(torchvision.utils.make_grid(fake_imgs, nrow=3),
                                  n_iter,
                                  'test_{}_{}'.format(self.dose, self.sampling_routine) + '_' + opt.test_dataset)
