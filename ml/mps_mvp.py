import json
import copy
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# ===========================================
# 数据预处理（适配您的JSON格式）
# ===========================================
def load_data(file_path):
    with open(file_path) as f:
        graphs = json.load(f)

    dataset = []
    for graph in graphs:
        # 节点特征矩阵
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

        # 边索引（处理有向图）
        edge_index = []
        for edge in graph["edges"]:
            edge_index.append([edge["source"], edge["target"]])

        # 转换为PyG Data对象
        data = Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
            y=torch.tensor(labels, dtype=torch.float).view(-1, 1)
        )
        dataset.append(data)
    return dataset

# ===========================================
# 数据增强函数（深拷贝以避免原数据被破坏）
# ===========================================
def graph_augmentation(data, edge_drop_rate=0.2, feat_mask_rate=0.2):
    data_aug = copy.deepcopy(data)
    # 随机边丢弃
    if edge_drop_rate > 0:
        num_edges = data_aug.edge_index.size(1)
        mask = torch.rand(num_edges) > edge_drop_rate
        data_aug.edge_index = data_aug.edge_index[:, mask]
    # 特征遮罩
    if feat_mask_rate > 0:
        feat_mask = torch.rand_like(data_aug.x) < feat_mask_rate
        data_aug.x[feat_mask] = 0.0
    return data_aug

# ===========================================
# 改进的GNN模型（关注有向图中异常节点的上游信息）
# ===========================================
class FaultGNN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        # 正向分支：利用原始边捕获全局信息
        self.conv_forward = GCNConv(in_dim, hidden_dim)
        # 上游分支：利用反向边捕获上游信息
        self.conv_upstream = GCNConv(in_dim, hidden_dim)
        # 融合层
        self.fc = torch.nn.Linear(2 * hidden_dim, hidden_dim)
        # 最终输出层
        self.conv_out = GCNConv(hidden_dim, 1)  # 输出节点故障概率
        self.dropout = torch.nn.Dropout(0.5)
        self.relu = torch.nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        # 对所有GCNConv和Linear层进行Xavier初始化
        for m in self.modules():
            if isinstance(m, GCNConv):
                torch.nn.init.xavier_uniform_(m.lin.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # 构造反向边索引用于上游信息传播：将 (source, target) 翻转为 (target, source)
        reverse_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)

        # 正向分支（全局信息聚合）
        h_forward = self.conv_forward(x, edge_index)
        h_forward = self.relu(h_forward)
        h_forward = self.dropout(h_forward)

        # 上游分支（专注于上游节点信息）
        h_upstream = self.conv_upstream(x, reverse_edge_index)
        h_upstream = self.relu(h_upstream)
        h_upstream = self.dropout(h_upstream)

        # 融合两个分支的信息
        h = torch.cat([h_forward, h_upstream], dim=1)
        h = self.fc(h)
        h = self.relu(h)

        # 最终输出层，可以选择用正向边进一步聚合信息
        out = self.conv_out(h, edge_index)
        return torch.sigmoid(out)  # 输出概率值

# ===========================================
# 训练流程
# ===========================================
def train():
    # 优先使用CUDA，其次检查MPS
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    # 加载数据
    full_dataset = load_data("./utils/train.json")
    train_dataset = full_dataset[:30]
    val_dataset = full_dataset[30:35]
    test_dataset = full_dataset[35:40]

    # 初始化模型
    model = FaultGNN(in_dim=4, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = torch.nn.BCELoss()

    # 训练循环
    best_val_loss = float("inf")
    for epoch in range(300):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for data in train_dataset:
            data_aug = graph_augmentation(data).to(device)  # 应用数据增强
            optimizer.zero_grad()
            out = model(data_aug)
            loss = criterion(out, data_aug.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证阶段
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for data in val_dataset:
                data = data.to(device)
                out = model(data)
                val_loss += criterion(out, data.y).item()

        # 早停和模型保存逻辑
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")

        print(f"Epoch {epoch + 1:03d} | "
              f"Train Loss: {train_loss / len(train_dataset):.4f} | "
              f"Val Loss: {val_loss / len(val_dataset):.4f}")

    # 测试阶段
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for data in test_dataset:
            data = data.to(device)
            out = model(data)
            test_loss += criterion(out, data.y).item()
    print(f"Final Test Loss: {test_loss / len(test_dataset):.4f}")

if __name__ == "__main__":
    train()
