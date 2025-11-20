from torch import nn
import torch
import os.path as osp
import os
from torchvision.utils import save_image
import torch.distributed as dist

def turn_on_spectral_norm(module):
    module_output = module
    if isinstance(module, torch.nn.Conv2d):
        if module.out_channels != 1 and module.in_channels > 4:
            module_output = nn.utils.spectral_norm(module)
    # if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
    #     module_output = nn.utils.spectral_norm(module)
    for name, child in module.named_children():
        module_output.add_module(name, turn_on_spectral_norm(child))
    del module
    return module_output


def reduce_tensor(tensor, world_size=None):
    rt = tensor.clone()
    if dist.is_initialized():
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    else:
        world_size = 1
    if world_size is not None:
        rt /= world_size
    return rt

#load_network 的作用是加载模型的权重，并处理分布式训练中生成的权重文件，以便在非分布式或其他设备上正确加载。
def load_network(state_dict):
    if isinstance(state_dict, str):
        state_dict = torch.load(state_dict, map_location='cpu')
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    """问题背景：分布式训练使用 torch.nn.DataParallel 或 torch.nn.parallel.DistributedDataParallel 时，模型的权重名称前会自动添加 module. 前缀。
例如，原本的权重名是 layer1.weight，分布式训练保存的权重名可能是 module.layer1.weight。"""
    for k, v in state_dict.items():
        namekey = k.replace('module.', '')  # remove `module.`
        new_state_dict[namekey] = v
    return new_state_dict


#convert_to_ddp 的作用是将模型转换为分布式数据并行（DistributedDataParallel, DDP）模式，以便在多 GPU 和多进程环境下高效训练。
def convert_to_ddp(*modules):
    modules = [x.cuda() for x in modules]
    if dist.is_initialized():
        rank = dist.get_rank()
        modules = [torch.nn.parallel.DistributedDataParallel(x,
                                                             device_ids=[rank, ],
                                                             output_device=rank) for x in modules]

    return modules

