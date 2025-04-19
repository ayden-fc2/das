import json
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

app = Flask(__name__)
CORS(app)

# ===========================================
# 设备选择
# ===========================================
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))


# ===========================================
# 单向GAT模型（移除了反向传播分支）
# ===========================================
class SingleGAT(torch.nn.Module):  # 类名修改为SingleGAT
    def __init__(self, in_dim=4, hidden_dim=64, heads=2, dropout=0.6):
        super().__init__()
        # 仅保留正向传播层
        self.conv1 = GATConv(
            in_dim,
            hidden_dim // heads,
            heads=heads,
            dropout=dropout
        )
        # 输出层
        self.conv_out = GATConv(
            hidden_dim,
            1,
            heads=1,
            concat=False
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.relu = torch.nn.ReLU()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # 单向传播
        h = self.conv1(x, edge_index)
        h = self.relu(h)
        h = self.dropout(h)

        # 直接输出
        out = self.conv_out(h, edge_index)
        return torch.sigmoid(out)


# ===========================================
# 加载训练好的单向模型
# ===========================================
model = SingleGAT().to(device)  # 使用新的模型类
model.load_state_dict(torch.load("best_single_gat_model.pth", map_location=device))  # 加载单向模型
model.eval()


# ===========================================
# 数据预处理函数（保持不变）
# ===========================================
def process_input(input_json):
    """将输入JSON转换为PyG Data对象，返回数据和节点id列表"""
    graph = input_json

    node_ids = [node["id"] for node in graph["nodes"]]
    id2idx = {nid: idx for idx, nid in enumerate(node_ids)}

    node_features = []
    for node in graph["nodes"]:
        node_features.append([
            node["in_degree"],
            node["betweenness"],
            node["closeness"],
            1.0 if node["is_anomalous_node"] else 0.0
        ])

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
# API端点（保持路由不变）
# ===========================================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.json
        data, node_ids = process_input(input_data)

        with torch.no_grad():
            pred = model(data)

        results = [{
            "node_id": node_id,
            "predicted_confidence": round(float(confidence), 4)
        } for node_id, confidence in zip(node_ids, pred.cpu().numpy().flatten())]

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