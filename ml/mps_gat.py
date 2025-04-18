import json
import copy
import torch
from sympy import false
from torch_geometric.data import Data
from torch_geometric.nn import GATConv  # 关键修改：GCNConv -> GATConv
import matplotlib.pyplot as plt

config_default = {
    "in_dim": 4,
    "hidden_dim": 64,
    "heads": 2,
    "dropout": 0.6,
    "lr": 0.0005,
    "weight_decay": 1e-4,
    "epochs": 150,
    "edge_drop_rate": 0.2,
    "feat_mask_rate": 0.2
}

configs_0 = [
    {"in_dim": 4, "hidden_dim": 32, "heads": 2, "dropout": 0.3, "lr": 0.001, "weight_decay": 1e-4, "epochs": 100, "edge_drop_rate": 0.2, "feat_mask_rate": 0.2},
    {"in_dim": 4, "hidden_dim": 32, "heads": 4, "dropout": 0.4, "lr": 0.0005, "weight_decay": 1e-4, "epochs": 100, "edge_drop_rate": 0.2, "feat_mask_rate": 0.2},
    {"in_dim": 4, "hidden_dim": 64, "heads": 2, "dropout": 0.6, "lr": 0.0005, "weight_decay": 1e-4, "epochs": 100, "edge_drop_rate": 0.2, "feat_mask_rate": 0.2},
    {"in_dim": 4, "hidden_dim": 64, "heads": 4, "dropout": 0.6, "lr": 0.0003, "weight_decay": 1e-4, "epochs": 100, "edge_drop_rate": 0.2, "feat_mask_rate": 0.2},
    {"in_dim": 4, "hidden_dim": 128, "heads": 2, "dropout": 0.5, "lr": 0.0003, "weight_decay": 1e-4, "epochs": 100, "edge_drop_rate": 0.2, "feat_mask_rate": 0.2},
    {"in_dim": 4, "hidden_dim": 128, "heads": 4, "dropout": 0.5, "lr": 0.0001, "weight_decay": 5e-4, "epochs": 100, "edge_drop_rate": 0.2, "feat_mask_rate": 0.2},

    # 控制变量：改变 dropout
    {"in_dim": 4, "hidden_dim": 64, "heads": 2, "dropout": 0.2, "lr": 0.0005, "weight_decay": 1e-4, "epochs": 100, "edge_drop_rate": 0.2, "feat_mask_rate": 0.2},
    {"in_dim": 4, "hidden_dim": 64, "heads": 2, "dropout": 0.8, "lr": 0.0005, "weight_decay": 1e-4, "epochs": 100, "edge_drop_rate": 0.2, "feat_mask_rate": 0.2},

    # 控制变量：改变 edge drop & feature mask
    {"in_dim": 4, "hidden_dim": 64, "heads": 2, "dropout": 0.6, "lr": 0.0005, "weight_decay": 1e-4, "epochs": 100, "edge_drop_rate": 0.4, "feat_mask_rate": 0.4},
    {"in_dim": 4, "hidden_dim": 64, "heads": 2, "dropout": 0.6, "lr": 0.0005, "weight_decay": 1e-4, "epochs": 100, "edge_drop_rate": 0.0, "feat_mask_rate": 0.0},

    # 控制变量：改变学习率
    {"in_dim": 4, "hidden_dim": 64, "heads": 2, "dropout": 0.6, "lr": 0.001, "weight_decay": 1e-4, "epochs": 100, "edge_drop_rate": 0.2, "feat_mask_rate": 0.2},
    {"in_dim": 4, "hidden_dim": 64, "heads": 2, "dropout": 0.6, "lr": 0.0001, "weight_decay": 1e-4, "epochs": 100, "edge_drop_rate": 0.2, "feat_mask_rate": 0.2}
]

configs = [
# {"in_dim":4, "hidden_dim":64, "heads":2, "dropout":0.2, "lr":0.0005,
#      "weight_decay":1e-4, "epochs":100, "edge_drop_rate":0.2, "feat_mask_rate":0.2},
# {"in_dim":4, "hidden_dim":64, "heads":2, "dropout":0.2, "lr":0.0007,
#      "weight_decay":1e-4, "epochs":100, "edge_drop_rate":0.2, "feat_mask_rate":0.2},
# {"in_dim":4, "hidden_dim":64, "heads":2, "dropout":0.2, "lr":0.0003,
#      "weight_decay":1e-4, "epochs":100, "edge_drop_rate":0.2, "feat_mask_rate":0.2},
{"in_dim":4, "hidden_dim":64, "heads":2, "dropout":0.15, "lr":0.0005,
     "weight_decay":1e-4, "epochs":100, "edge_drop_rate":0.2, "feat_mask_rate":0.2},
{"in_dim":4, "hidden_dim":64, "heads":2, "dropout":0.25, "lr":0.0005,
     "weight_decay":1e-4, "epochs":100, "edge_drop_rate":0.2, "feat_mask_rate":0.2},
{"in_dim":4, "hidden_dim":64, "heads":2, "dropout":0.2, "lr":0.0005,
     "weight_decay":5e-5, "epochs":100, "edge_drop_rate":0.2, "feat_mask_rate":0.2},
{"in_dim":4, "hidden_dim":64, "heads":2, "dropout":0.2, "lr":0.0005,
     "weight_decay":2e-4, "epochs":100, "edge_drop_rate":0.2, "feat_mask_rate":0.2},
{"in_dim":4, "hidden_dim":64, "heads":1, "dropout":0.2, "lr":0.0005,
     "weight_decay":1e-4, "epochs":100, "edge_drop_rate":0.2, "feat_mask_rate":0.2},
{"in_dim":4, "hidden_dim":48, "heads":2, "dropout":0.2, "lr":0.0005,
     "weight_decay":1e-4, "epochs":100, "edge_drop_rate":0.2, "feat_mask_rate":0.2},
]
config = {"in_dim":4, "hidden_dim":64, "heads":2, "dropout":0.15, "lr":0.0007,
     "weight_decay":1e-4, "epochs":100, "edge_drop_rate":0.2, "feat_mask_rate":0.2}


# ===========================================
# 数据预处理
# ===========================================
def load_data(file_path):
    with open(file_path) as f:
        graphs = json.load(f)

    dataset = []
    for graph in graphs:
        node_features = []
        labels = []
        for node in graph["nodes"]:
            features = [
                node["in_degree"],
                node["betweenness"],
                node["closeness"],
                1.0 if node["is_anomalous_node"] else 0.0
            ]
            node_features.append(features)
            labels.append(node["is_fault_confidence"])

        edge_index = [[edge["source"], edge["target"]] for edge in graph["edges"]]

        data = Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
            y=torch.tensor(labels, dtype=torch.float).view(-1, 1))
        dataset.append(data)
    return dataset


# ===========================================
# 数据增强
# ===========================================
def graph_augmentation(data, edge_drop_rate=0.2, feat_mask_rate=0.2):
    data_aug = copy.deepcopy(data)
    if edge_drop_rate > 0:
        num_edges = data_aug.edge_index.size(1)
        mask = torch.rand(num_edges) > edge_drop_rate
        data_aug.edge_index = data_aug.edge_index[:, mask]
    if feat_mask_rate > 0:
        feat_mask = torch.rand_like(data_aug.x) < feat_mask_rate
        data_aug.x[feat_mask] = 0.0
    return data_aug


# ===========================================
# 改进的GAT模型
# ===========================================
class FaultGAT(torch.nn.Module):
    def __init__(self, in_dim=4, hidden_dim=64, heads=2, dropout=0.6):
        super().__init__()
        # 正向传播分支：2头注意力
        self.conv_forward = GATConv(
            in_dim,
            hidden_dim // heads,  # 每个头输出维度为hidden_dim//heads
            heads=heads,
            dropout=dropout
        )
        # 反向传播分支（捕获上游信息）
        self.conv_upstream = GATConv(
            in_dim,
            hidden_dim // heads,
            heads=heads,
            dropout=dropout
        )
        # 特征融合层
        self.fc = torch.nn.Linear(2 * hidden_dim, hidden_dim)
        # 输出层：单头注意力
        self.conv_out = GATConv(
            hidden_dim,
            1,  # 输出维度为1
            heads=1,  # 单头注意力
            concat=False  # 不拼接头，直接输出
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.relu = torch.nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, GATConv):
                if hasattr(m, 'lin_l') and m.lin_l is not None:
                    torch.nn.init.xavier_uniform_(m.lin_l.weight)
                if hasattr(m, 'lin_r') and m.lin_r is not None:
                    torch.nn.init.xavier_uniform_(m.lin_r.weight)
                # 对于注意力参数，根据具体版本可能存储在 m.att 或 m.att_l / m.att_r 中，
                # 可根据你的PyG版本进行调整，例如：
                if hasattr(m, 'att') and m.att is not None:
                    torch.nn.init.xavier_uniform_(m.att)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        reverse_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)

        # 正向传播分支
        h_forward = self.conv_forward(x, edge_index)
        h_forward = self.relu(h_forward)
        h_forward = self.dropout(h_forward)

        # 上游传播分支
        h_upstream = self.conv_upstream(x, reverse_edge_index)
        h_upstream = self.relu(h_upstream)
        h_upstream = self.dropout(h_upstream)

        # 特征拼接与融合
        h = torch.cat([h_forward, h_upstream], dim=1)
        h = self.fc(h)
        h = self.relu(h)

        # 最终输出层
        out = self.conv_out(h, edge_index)
        return torch.sigmoid(out)


# ===========================================
# 训练流程（调整超参数）
# ===========================================
# 修改后的训练函数，接收config参数并返回最佳验证损失

def train(config):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # 数据加载
    full_dataset = load_data("./utils/train.json")
    # 数据集划分（保持原有逻辑）
    train_indices = list(range(0, 6)) + list(range(10, 16)) + list(range(20, 26)) + list(range(30, 36))
    train_dataset = [full_dataset[i] for i in train_indices]
    val_indices = list(range(6, 8)) + list(range(16, 18)) + list(range(26, 28)) + list(range(36, 38))
    val_dataset = [full_dataset[i] for i in val_indices]
    test_indices = list(range(8, 10)) + list(range(18, 20)) + list(range(28, 30)) + list(range(38, 40))
    test_dataset = [full_dataset[i] for i in test_indices]

    # 模型初始化
    model = FaultGAT(
        in_dim=config["in_dim"],
        hidden_dim=config["hidden_dim"],
        heads=config["heads"],
        dropout=config["dropout"]
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )
    criterion = torch.nn.BCELoss()

    best_val_loss = float("inf")
    best_epoch = 0

    # 使用config中的epochs参数
    for epoch in range(config["epochs"]):
        model.train()
        train_loss = 0.0
        for data in train_dataset:
            data_aug = graph_augmentation(
                data,
                edge_drop_rate=config["edge_drop_rate"],
                feat_mask_rate=config["feat_mask_rate"]
            ).to(device)
            optimizer.zero_grad()
            out = model(data_aug)
            loss = criterion(out, data_aug.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in val_dataset:
                data = data.to(device)
                out = model(data)
                val_loss += criterion(out, data.y).item()

        avg_val_loss = val_loss / len(val_dataset)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch

        print(f"Config {current_config} | Epoch {epoch + 1:03d}/{config['epochs']:03d} | "
              f"Train Loss: {train_loss / len(train_dataset):.4f} | Val Loss: {avg_val_loss:.4f}")

    return best_val_loss

def train_save_best():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # 数据加载
    full_dataset = load_data("./utils/train.json")
    # train_dataset = full_dataset[:30]
    # val_dataset = full_dataset[30:35]
    # test_dataset = full_dataset[35:40]
    # 训练集
    train_indices = list(range(2, 8)) + list(range(12, 18)) + list(range(22, 28)) + list(range(32, 38)) + list(range(42, 48))
    train_dataset = [full_dataset[i] for i in train_indices]

    # 验证集
    val_indices = list(range(0, 2)) + list(range(10, 12)) + list(range(20, 22)) + list(range(30, 32)) + list(range(40, 42))
    val_dataset = [full_dataset[i] for i in val_indices]

    # 测试集
    test_indices = list(range(8, 10)) + list(range(18, 20)) + list(range(28, 30)) + list(range(38, 40)) + list(range(48, 50))
    test_dataset = [full_dataset[i] for i in test_indices]

    # 模型初始化（关键修改）
    model = FaultGAT(
        in_dim=config["in_dim"],
        hidden_dim=config["hidden_dim"],
        heads=config["heads"],
        dropout=config["dropout"]
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )
    criterion = torch.nn.BCELoss()

    # 训练循环
    best_val_loss = float("inf")

    train_loss_list = []  # 用于记录每个 epoch 的训练损失
    val_loss_list = []  # 用于记录每个 epoch 的验证损失

    for epoch in range(config["epochs"]):
        model.train()
        train_loss = 0.0
        for data in train_dataset:
            data_aug = graph_augmentation(
                data,
                edge_drop_rate=config["edge_drop_rate"],
                feat_mask_rate=config["feat_mask_rate"]
            ).to(device)
            optimizer.zero_grad()
            out = model(data_aug)
            loss = criterion(out, data_aug.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 记录训练损失均值
        avg_train_loss = train_loss / len(train_dataset)
        train_loss_list.append(avg_train_loss)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in val_dataset:
                data = data.to(device)
                out = model(data)
                val_loss += criterion(out, data.y).item()
        avg_val_loss = val_loss / len(val_dataset)
        val_loss_list.append(avg_val_loss)

        # 早停逻辑
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_gat_model.pth")  # 修改保存名称

        print(
            f"Epoch {epoch + 1:03d} | Train Loss: {train_loss / len(train_dataset):.4f} | Val Loss: {val_loss / len(val_dataset):.4f}")

    # 绘制训练过程的损失曲线
    epochs_range = range(1, config["epochs"] + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, train_loss_list, label='Train Loss')
    plt.plot(epochs_range, val_loss_list, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig("training_loss_curve.png", dpi=300)
    plt.show()

    # 测试阶段
    model.load_state_dict(torch.load("best_gat_model.pth"))
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for data in test_dataset:
            data = data.to(device)
            out = model(data)
            test_loss += criterion(out, data.y).item()
    print(f"GAT Test Loss: {test_loss / len(test_dataset):.4f}")

configs_0 = [
    {"in_dim": 4, "hidden_dim": 32, "heads": 2, "dropout": 0.3, "lr": 0.001, "weight_decay": 1e-4, "epochs": 100, "edge_drop_rate": 0.2, "feat_mask_rate": 0.2},
    {"in_dim": 4, "hidden_dim": 32, "heads": 4, "dropout": 0.4, "lr": 0.0005, "weight_decay": 1e-4, "epochs": 100, "edge_drop_rate": 0.2, "feat_mask_rate": 0.2},
    {"in_dim": 4, "hidden_dim": 64, "heads": 2, "dropout": 0.6, "lr": 0.0005, "weight_decay": 1e-4, "epochs": 100, "edge_drop_rate": 0.2, "feat_mask_rate": 0.2},
    {"in_dim": 4, "hidden_dim": 64, "heads": 4, "dropout": 0.6, "lr": 0.0003, "weight_decay": 1e-4, "epochs": 100, "edge_drop_rate": 0.2, "feat_mask_rate": 0.2},
    {"in_dim": 4, "hidden_dim": 128, "heads": 2, "dropout": 0.5, "lr": 0.0003, "weight_decay": 1e-4, "epochs": 100, "edge_drop_rate": 0.2, "feat_mask_rate": 0.2},
    {"in_dim": 4, "hidden_dim": 128, "heads": 4, "dropout": 0.5, "lr": 0.0001, "weight_decay": 5e-4, "epochs": 100, "edge_drop_rate": 0.2, "feat_mask_rate": 0.2},

    # 控制变量：改变 dropout
    {"in_dim": 4, "hidden_dim": 64, "heads": 2, "dropout": 0.2, "lr": 0.0005, "weight_decay": 1e-4, "epochs": 100, "edge_drop_rate": 0.2, "feat_mask_rate": 0.2},
    {"in_dim": 4, "hidden_dim": 64, "heads": 2, "dropout": 0.8, "lr": 0.0005, "weight_decay": 1e-4, "epochs": 100, "edge_drop_rate": 0.2, "feat_mask_rate": 0.2},

    # 控制变量：改变 edge drop & feature mask
    {"in_dim": 4, "hidden_dim": 64, "heads": 2, "dropout": 0.6, "lr": 0.0005, "weight_decay": 1e-4, "epochs": 100, "edge_drop_rate": 0.4, "feat_mask_rate": 0.4},
    {"in_dim": 4, "hidden_dim": 64, "heads": 2, "dropout": 0.6, "lr": 0.0005, "weight_decay": 1e-4, "epochs": 100, "edge_drop_rate": 0.0, "feat_mask_rate": 0.0},

    # 控制变量：改变学习率
    {"in_dim": 4, "hidden_dim": 64, "heads": 2, "dropout": 0.6, "lr": 0.001, "weight_decay": 1e-4, "epochs": 100, "edge_drop_rate": 0.2, "feat_mask_rate": 0.2},
    {"in_dim": 4, "hidden_dim": 64, "heads": 2, "dropout": 0.6, "lr": 0.0001, "weight_decay": 1e-4, "epochs": 100, "edge_drop_rate": 0.2, "feat_mask_rate": 0.2}
]

configs = [
# {"in_dim":4, "hidden_dim":64, "heads":2, "dropout":0.2, "lr":0.0005,
#      "weight_decay":1e-4, "epochs":100, "edge_drop_rate":0.2, "feat_mask_rate":0.2},
# {"in_dim":4, "hidden_dim":64, "heads":2, "dropout":0.2, "lr":0.0007,
#      "weight_decay":1e-4, "epochs":100, "edge_drop_rate":0.2, "feat_mask_rate":0.2},
# {"in_dim":4, "hidden_dim":64, "heads":2, "dropout":0.2, "lr":0.0003,
#      "weight_decay":1e-4, "epochs":100, "edge_drop_rate":0.2, "feat_mask_rate":0.2},
{"in_dim":4, "hidden_dim":64, "heads":2, "dropout":0.15, "lr":0.0005,
     "weight_decay":1e-4, "epochs":100, "edge_drop_rate":0.2, "feat_mask_rate":0.2},
{"in_dim":4, "hidden_dim":64, "heads":2, "dropout":0.25, "lr":0.0005,
     "weight_decay":1e-4, "epochs":100, "edge_drop_rate":0.2, "feat_mask_rate":0.2},
{"in_dim":4, "hidden_dim":64, "heads":2, "dropout":0.2, "lr":0.0005,
     "weight_decay":5e-5, "epochs":100, "edge_drop_rate":0.2, "feat_mask_rate":0.2},
{"in_dim":4, "hidden_dim":64, "heads":2, "dropout":0.2, "lr":0.0005,
     "weight_decay":2e-4, "epochs":100, "edge_drop_rate":0.2, "feat_mask_rate":0.2},
{"in_dim":4, "hidden_dim":64, "heads":1, "dropout":0.2, "lr":0.0005,
     "weight_decay":1e-4, "epochs":100, "edge_drop_rate":0.2, "feat_mask_rate":0.2},
{"in_dim":4, "hidden_dim":48, "heads":2, "dropout":0.2, "lr":0.0005,
     "weight_decay":1e-4, "epochs":100, "edge_drop_rate":0.2, "feat_mask_rate":0.2},
]
config = {"in_dim":4, "hidden_dim":64, "heads":2, "dropout":0.15, "lr":0.0007,
     "weight_decay":1e-4, "epochs":150, "edge_drop_rate":0.2, "feat_mask_rate":0.2}



if __name__ == "__main__":
    if false:
        results = []
        for idx, config in enumerate(configs):
            print(f"\n{'=' * 40}")
            print(f"Training config {idx + 1}/{len(configs)}")
            print(json.dumps(config, indent=2))
            current_config = idx + 1

            best_val = train(config)
            results.append((config, best_val))

            print(f"\nConfig {current_config} Best Val Loss: {best_val:.4f}")
            print('=' * 40 + '\n')

        # 按验证损失排序
        sorted_results = sorted(results, key=lambda x: x[1])

        print("\n\n=== Final Results Ranking ===")
        for rank, (config, loss) in enumerate(sorted_results, 1):
            print(f"Rank {rank}:")
            print(f"Val Loss: {loss:.4f}")
            print(json.dumps(config, indent=2))
            print('-' * 40)
    else:
        train_save_best()