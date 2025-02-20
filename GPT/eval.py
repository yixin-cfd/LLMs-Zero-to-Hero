import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from dataclasses import dataclass

from model import GPT
from config import GPTConfig
##########################
checkpoint_path = "checkpoints/model_epoch_1.pt"
##########################
checkpoint = torch.load(checkpoint_path)

# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
# epoch = checkpoint['epoch']
# val_loss = checkpoint['val_loss']


model = GPT(GPTConfig()).to('xpu')
model.load_state_dict(checkpoint['model_state_dict'])  # 加载模型的权重



while True:
    s = input("请输入文本:\n")
    print()
    model.query(s)