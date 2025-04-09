import json
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from torch_geometric.data import Data
from torch_geometric.nn import GATConv  # 修改：使用 GATConv 替换 GCNConv

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# ===========================================
# 设备选择：优先使用CUDA，其次MPS，最后CPU
# ===========================================
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

# ===========================================
# 改进的 FaultGAT 模型（双分支结构关注上游信息，使用GATConv）
# ===========================================
class FaultGAT(torch.nn.Module):
    def __init__(self, in_dim=4, hidden_dim=64, heads=2, dropout=0.6):
        super().__init__()
        # 正向分支：利用原始边捕获全局信息，使用多头注意力
        self.conv_forward = GATConv(
            in_dim,
            hidden_dim // heads,  # 每个头的输出维度
            heads=heads,
            dropout=dropout
        )
        # 上游分支：利用反向边捕获上游信息
        self.conv_upstream = GATConv(
            in_dim,
            hidden_dim // heads,
            heads=heads,
            dropout=dropout
        )
        # 融合层：将两个分支的输出拼接后映射到 hidden_dim
        self.fc = torch.nn.Linear(2 * hidden_dim, hidden_dim)
        # 最终输出层：使用正向边索引进一步聚合信息
        self.conv_out = GATConv(
            hidden_dim,
            1,             # 输出维度为1
            heads=1,       # 单头注意力
            concat=False,  # 不拼接头，直接输出
            dropout=dropout
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.relu = torch.nn.ReLU()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # 构造反向边索引，用于上游信息传播
        reverse_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)

        # 正向分支：全局信息聚合
        h_forward = self.conv_forward(x, edge_index)
        h_forward = self.relu(h_forward)
        h_forward = self.dropout(h_forward)

        # 上游分支：利用反向边捕获上游信息
        h_upstream = self.conv_upstream(x, reverse_edge_index)
        h_upstream = self.relu(h_upstream)
        h_upstream = self.dropout(h_upstream)

        # 融合两个分支的特征
        h = torch.cat([h_forward, h_upstream], dim=1)
        h = self.fc(h)
        h = self.relu(h)

        # 最终输出层，依然使用正向边进行信息聚合
        out = self.conv_out(h, edge_index)
        return torch.sigmoid(out)  # 输出每个节点的故障概率

# ===========================================
# 加载训练好的模型
# ===========================================
# 注意这里加载的模型文件应为基于GAT训练得到的最佳模型文件（例如 "best_gat_model.pth"）
model = FaultGAT().to(device)
model.load_state_dict(torch.load("best_gat_model.pth", map_location=device))
model.eval()

# ===========================================
# 数据预处理函数：将输入 JSON 转换为 PyG Data 对象
# ===========================================
def process_input(input_json):
    """将输入JSON转换为PyG Data对象，返回数据和节点id列表"""
    graph = input_json  # 假设每次只处理一个图

    # 收集原始节点 id（字符串）
    node_ids = [node["id"] for node in graph["nodes"]]
    # 建立从原始 id 到 0-based 索引的映射
    id2idx = {nid: idx for idx, nid in enumerate(node_ids)}

    node_features = []
    for node in graph["nodes"]:
        node_features.append([
            node["in_degree"],
            node["betweenness"],
            node["closeness"],
            1.0 if node["is_anomalous_node"] else 0.0
        ])

    # 构造 edge_index：将 source/target 的字符串 id 转换为数值索引
    edge_index = [
        [id2idx[edge["source"]], id2idx[edge["target"]]]
        for edge in graph["edges"]
    ]

    data = Data(
        x=torch.tensor(node_features, dtype=torch.float).to(device),
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)
    )
    return data, node_ids

# ===========================================
# API端点：/predict-gat
# ===========================================
@app.route('/predict-gat', methods=['POST'])
def predict():
    try:
        # 解析输入数据（JSON格式）
        input_data = request.json

        # 数据预处理
        data, node_ids = process_input(input_data)

        # 模型预测
        with torch.no_grad():
            pred = model(data)

        # 格式化输出，每个节点返回其预测的故障置信度
        results = [{
            "node_id": node_id,
            "predicted_confidence": round(float(confidence), 4)
        } for node_id, confidence in zip(node_ids, pred.cpu().numpy().flatten())]

        # 按置信度降序排序
        sorted_results = sorted(results, key=lambda x: x["predicted_confidence"], reverse=True)

        return jsonify({
            "status": "success",
            "predictions": sorted_results
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2081, debug=True)
