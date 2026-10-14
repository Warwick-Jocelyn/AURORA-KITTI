import torch
from model import DepthCompletionNet
import argparse

# -------------------- 配置 --------------------
parser = argparse.ArgumentParser(description='Count model parameters')
parser.add_argument('--checkpoint', type=str, required=True,
                    help='path to pretrained model checkpoint (.pth.tar)')
parser.add_argument('--cpu', action='store_true', help='run on CPU')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

# -------------------- 加载模型 --------------------
# 创建模型实例
model = DepthCompletionNet(args).to(device)

# 加载checkpoint
checkpoint = torch.load(args.checkpoint, map_location=device)
model.load_state_dict(checkpoint['model'])

# -------------------- 统计参数 --------------------
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
