import torch

# 模型文件路径
model_path = r'E:\machine_learning\结果分析\ema_model-best'

# 加载模型文件
checkpoint = torch.load(model_path, map_location='cpu')

# 打印 checkpoint 的键（通常包含 'state_dict', 'epoch', 'optimizer', 等）
print("Keys in the checkpoint:")
for key in checkpoint.keys():
    print(f"  - {key}")

# 如果包含模型参数（如 'state_dict'），可以进一步查看具体的内容
if 'state_dict' in checkpoint:
    print("\nModel state_dict keys:")
    for param_tensor in checkpoint['state_dict']:
        print(f"  - {param_tensor} : {checkpoint['state_dict'][param_tensor].shape}")
else:
    print("\nCheckpoint is likely a plain state_dict.")
    for param_tensor in checkpoint:
        print(f"  - {param_tensor} : {checkpoint[param_tensor].shape}")
