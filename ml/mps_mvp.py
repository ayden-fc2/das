import torch
import torch.nn as nn
import torch.optim as optim
import time
import networkx as nx
import matplotlib.pyplot as plt

# 选择设备（Apple MPS 或 CPU）
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# 1. 生成训练数据
if __name__ == '__main__':
    print("Loading data...")