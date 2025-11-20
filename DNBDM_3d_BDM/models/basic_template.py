# This part builds heavily on https://github.com/Hzzone/DU-GAN.
import torch
import os.path as osp
import tqdm
import argparse
import torch.distributed as dist
import os
import glob
from utils.dataset import dataset_dict
from utils.loggerx import LoggerX
from utils.sampler import RandomSampler
from utils.ops import load_network
import wandb

#主要参数和运行训练、测试
class TrainTask(object):

    def __init__(self, opt):
        self.opt = opt
        self.logger = LoggerX(save_root=osp.join(
            osp.dirname(osp.dirname(osp.abspath(__file__))), 'output', '{}_{}'.format(opt.model_name, opt.run_name)))
        self.rank = dist.get_rank() if dist.is_initialized() else 0

        self.set_loader()
        self.set_model()
        self.best_val_loss = float("inf")

    @staticmethod
    def build_default_options():
        parser = argparse.ArgumentParser('Default arguments for training of different methods')

        parser.add_argument('--save_freq', type=int, default=2500,
                            help='save frequency')
        parser.add_argument('--patience', type=int, default=20000,
                            help='early stopping patience')
        parser.add_argument('--min_delta', type=float, default=0.,
                            help='sub')
        parser.add_argument('--batch_size', type=int, default=4,
                            help='batch_size')
        parser.add_argument('--test_batch_size', type=int, default=1,
                            help='test_batch_size')
        parser.add_argument('--num_workers', type=int, default=4,
                            help='num of workers to use')
        parser.add_argument('--max_iter', type=int, default=150000,
                            help='number of training iterations')
        parser.add_argument('--resume_iter', type=int, default=0,
                            help='number of training epochs')
        parser.add_argument('--test_iter', type=int, default=150000,
                            help='number of epochs for test')
        parser.add_argument("--local_rank", type=int, default=0)
        parser.add_argument("--mode", type=str, default='train')
        parser.add_argument('--wandb', action="store_true")

        # run_name and model_name
        parser.add_argument('--run_name', type=str, default='default',
                            help='each run name')
        parser.add_argument('--model_name', type=str, default='corediff',
                            help='the type of method')

        # training parameters for one-shot learning framework
        parser.add_argument("--osl_max_iter", type=int, default=3001,
                            help='number of training iterations for one-shot learning framework training')
        parser.add_argument("--osl_batch_size", type=int, default=8,
                            help='batch size for one-shot learning framework training')
        parser.add_argument("--index", type=int, default=10,
                            help='slice index selected for one-shot learning framework training')
        parser.add_argument("--unpair", action="store_true",
                            help='use unpaired data for one-shot learning framework training')
        parser.add_argument("--patch_size", type=int, default=256,
                            help='patch size used to divide the image')

        # dataset
        parser.add_argument('--train_dataset', type=str, default='mayo_2016_sim')
        parser.add_argument('--test_dataset', type=str, default='mayo_2016_sim')   # mayo_2020, piglte, phantom, mayo_2016
        parser.add_argument('--test_ids', type=str, default='9',
                            help='test patient index for Mayo 2016')
        parser.add_argument('--context', action="store_true",
                            help='use contextual information')   #
        parser.add_argument('--image_size', type=str, default="36,36,36", help="Image size as Depth,Height,Width")
        parser.add_argument('--dose', type=str, default=5,
                            help='dose% data use for training and testing')
        parser.add_argument(
            '--checkpoint_dir',
            type=str,
            default="/hy-tmp/corediff_spect_val_3d/output/corediff_dose10s1s_spect/save_models",
            help='Directory to save model checkpoints'
        )
        parser.add_argument('--max_checkpoints_to_keep', type=int, default=5,
                            help='Maximum number of checkpoints to keep')

        return parser

    @staticmethod
    def build_options():
        pass

    def load_pretrained_dict(self, file_name: str):
        self.project_root = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
        return load_network(osp.join(self.project_root, 'pretrained', file_name))
    #加载数据，数据定义
    def set_loader(self):
        opt = self.opt

        if opt.mode == 'train':
            test_ids = list(map(int, opt.test_ids.split(',')))  # 解析test_ids为整数列表
            train_dataset = dataset_dict['train'](
                dataset=opt.train_dataset,
                test_ids=test_ids,
                dose=opt.dose,
                context=opt.context,
            )
            train_sampler = RandomSampler(dataset=train_dataset, batch_size=opt.batch_size,
                                          num_iter=opt.max_iter,
                                          restore_iter=opt.resume_iter)

            train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=opt.batch_size,
                sampler=train_sampler,
                shuffle=False,
                drop_last=False,
                num_workers=opt.num_workers,
                pin_memory=True
            )
            self.train_loader = train_loader

        test_dataset = dataset_dict[opt.test_dataset](
            dataset=opt.test_dataset,
            test_ids=test_ids,
            dose=opt.dose,
            context=opt.context
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=opt.test_batch_size,
            shuffle=False,
            num_workers=opt.num_workers,
            pin_memory=True
        )
        self.test_loader = test_loader

        test_images = [test_dataset[i] for i in range(len(test_dataset))]
        low_dose = torch.stack([torch.from_numpy(x[0]) for x in test_images], dim=0).cuda()
        full_dose = torch.stack([torch.from_numpy(x[1]) for x in test_images], dim=0).cuda()
        self.test_images = (low_dose, full_dose)

        self.test_dataset = test_dataset

    #训练、测试
    def fit(self):
        opt = self.opt
        no_improve_count = 0
        if opt.mode == 'train':
            # 加载断点
            if opt.resume_iter > 0:
                self.logger.load_checkpoints(opt.resume_iter)

            # 训练循环
            loader = iter(self.train_loader)
            for n_iter in tqdm.trange(opt.resume_iter + 1, opt.max_iter + 1, disable=(self.rank != 0)):
                inputs = next(loader)
                self.train(inputs, n_iter)

                # 保存模型和验证
                if n_iter % opt.save_freq == 0:
                    #self.logger.checkpoints(n_iter)

                    # 验证并生成图像
                    current_val_loss = self.val(n_iter)
                    #self.generate_images(n_iter)

                    # 检查是否为最佳模型
                    if current_val_loss < self.best_val_loss:
                        self.best_val_loss = current_val_loss
                        self.best_model_state = self.ema_model.state_dict()
                        self.logger.save_best_model(n_iter)
                        no_improve_count = 0
                    else:
                        no_improve_count += 1

                    # 删除旧的检查点文件（仅保留最近的 N 个）
                    self.cleanup_old_checkpoints(opt.checkpoint_dir, opt.max_checkpoints_to_keep)

                    # # 提前停止条件
                    # if no_improve_count == opt.patience:
                    #     print('No improvement in validation loss.')
                    #     self.logger.plot_metrics()
                    #     break

            # 完成训练后绘制指标
            if n_iter == opt.max_iter:
                self.logger.plot_metrics()


        elif opt.mode == 'test':
            self.logger.load_test_checkpoints(opt.test_iter)
            self.test(opt.test_iter)
            # self.generate_images(opt.test_iter)

    def cleanup_old_checkpoints(self, checkpoint_dir, max_to_keep=5):
        """
        删除旧的检查点文件，仅保留最近 max_to_keep 个。
        """
        # 查找所有检查点文件，包括 model-* 和 optimizer-*
        checkpoint_files = sorted(
            glob.glob(os.path.join(checkpoint_dir, "model-*")) +
            glob.glob(os.path.join(checkpoint_dir, "optimizer-*")),
            key=os.path.getmtime
        )

        # 删除多余的检查点
        if len(checkpoint_files) > max_to_keep:
            for file in checkpoint_files[:-max_to_keep]:
                os.remove(file)
                print(f"Deleted old checkpoint: {file}")

    def set_model(opt):
        pass

    def train(self, inputs, n_iter):
        pass

    @torch.no_grad()
    def val(self, n_iter):
        pass

    @torch.no_grad()
    def generate_images(self, n_iter):
        pass

    def adjust_learning_rate(self, n_iter):
        opt = self.opt
        pass

    def transfer_calculate_window(self, img, MIN_B=0, MAX_B=65535, cut_min=0, cut_max=65535):
        img = img * (MAX_B - MIN_B) + MIN_B  # 将归一化图像恢复到物理值范围
        img[img < cut_min] = cut_min  # 裁剪低于 cut_min 的像素
        img[img > cut_max] = cut_max  # 裁剪高于 cut_max 的像素
        img = 255 * (img - cut_min) / (cut_max - cut_min)  # 映射到 [0, 255] 用于显示
        return img

    def transfer_display_window(self, img, MIN_B=0, MAX_B=65535, cut_min=0, cut_max=65535):
        """
        调整图像的显示窗口，适配 5D 数据。

        :param img: 输入图像，形状为 (B, C, D, H, W) 或其他维度张量。
        :param MIN_B: 图像的最小物理值范围。
        :param MAX_B: 图像的最大物理值范围。
        :param cut_min: 裁剪的最小值。
        :param cut_max: 裁剪的最大值。
        :return: 经过窗口调整并归一化到 [0, 1] 范围的图像。
        """
        # 将归一化图像恢复到物理值范围 [MIN_B, MAX_B]
        img = img * (MAX_B - MIN_B) + MIN_B

        # 裁剪到 [cut_min, cut_max] 范围
        img = torch.clamp(img, min=cut_min, max=cut_max)

        # 重新归一化到 [0, 1] 范围
        img = (img - cut_min) / (cut_max - cut_min + 1e-8)  # 防止除以 0

        return img


