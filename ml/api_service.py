import json
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# ===========================================
# 设备选择：优先使用CUDA，其次MPS，最后CPU
# ===========================================
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

# ===========================================
# 改进的 FaultGNN 模型：双分支结构关注上游信息
# ===========================================
class FaultGNN(torch.nn.Module):
    def __init__(self, in_dim=4, hidden_dim=64):
        super().__init__()
        # 正向分支：使用原始边索引获取全局信息
        self.conv_forward = GCNConv(in_dim, hidden_dim)
        # 上游分支：利用反向边索引获取上游节点信息
        self.conv_upstream = GCNConv(in_dim, hidden_dim)
        # 融合层，将两个分支的输出拼接后映射到 hidden_dim
        self.fc = torch.nn.Linear(2 * hidden_dim, hidden_dim)
        # 最终输出层：再次利用正向边索引进行信息聚合
        self.conv_out = GCNConv(hidden_dim, 1)
        self.dropout = torch.nn.Dropout(0.5)
        self.relu = torch.nn.ReLU()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # 构造反向边索引：用于上游信息传播
        reverse_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)

        # 正向分支：全局信息
        h_forward = self.conv_forward(x, edge_index)
        h_forward = self.relu(h_forward)
        h_forward = self.dropout(h_forward)

        # 上游分支：重点捕获上游节点信息
        h_upstream = self.conv_upstream(x, reverse_edge_index)
        h_upstream = self.relu(h_upstream)
        h_upstream = self.dropout(h_upstream)

        # 融合两个分支的特征
        h = torch.cat([h_forward, h_upstream], dim=1)
        h = self.fc(h)
        h = self.relu(h)

        # 最终输出层，依然使用正向边聚合信息
        out = self.conv_out(h, edge_index)
        return torch.sigmoid(out)

# ===========================================
# 加载训练好的模型
# ===========================================
model = FaultGNN().to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# ===========================================
# 数据预处理函数：将输入 JSON 转换为 PyG Data 对象
# ===========================================
def process_input(input_json):
    """将输入JSON转换为PyG Data对象，返回数据和节点id列表"""
    graph = input_json  # 假设每次只处理一个图

    node_features = []
    node_ids = []
    for node in graph["nodes"]:
        features = [
            node["in_degree"],
            node["betweenness"],
            node["closeness"],
            1.0 if node["is_anomalous_node"] else 0.0
        ]
        node_features.append(features)
        node_ids.append(node["id"])

    edge_index = []
    for edge in graph["edges"]:
        edge_index.append([edge["source"], edge["target"]])

    data = Data(
        x=torch.tensor(node_features, dtype=torch.float).to(device),
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)
    )
    return data, node_ids

# ===========================================
# API端点：/predict
# ===========================================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 解析输入数据（JSON 格式）
        input_data = request.json

        # 数据预处理
        data, node_ids = process_input(input_data)

        # 模型预测
        with torch.no_grad():
            pred = model(data)

        # 格式化输出，每个节点返回其预测的故障置信度
        results = [{
            "node_id": int(node_id),
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
    app.run(host='0.0.0.0', port=2080, debug=True)
