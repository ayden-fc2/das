import json
import networkx as nx
import matplotlib.pyplot as plt
from sympy import true, false

# 把你的 JSON 粘到这里，或者从文件/接口读取
data = {
    "nodes": [
        {
            "in_degree": 0,
            "closeness": 0.25806451612903225,
            "x": 87.3032909143769,
            "is_fault_confidence": 0,
            "y": 516.6549630745781,
            "is_anomalous_node": false,
            "id": "2-872",
            "betweenness": 0
        },
        {
            "in_degree": 0,
            "closeness": 0.25806451612903225,
            "x": 253.21671396050078,
            "is_fault_confidence": 0,
            "y": 582.1571400866422,
            "is_anomalous_node": false,
            "id": "2-874",
            "betweenness": 0
        },
        {
            "in_degree": 1,
            "closeness": 0,
            "x": 83.52943266436571,
            "is_fault_confidence": 0,
            "y": -155.03639874562705,
            "is_anomalous_node": false,
            "id": "2-895",
            "betweenness": 0
        },
        {
            "in_degree": 1,
            "closeness": 0,
            "x": 84.9489412010029,
            "is_fault_confidence": 0,
            "y": -295.5462822903275,
            "is_anomalous_node": false,
            "id": "2-909",
            "betweenness": 0
        },
        {
            "in_degree": 0,
            "closeness": 0.24444444444444444,
            "x": 87.74381591167472,
            "is_fault_confidence": 0,
            "y": 357.6692758760748,
            "is_anomalous_node": false,
            "id": "2-916",
            "betweenness": 0
        },
        {
            "in_degree": 0,
            "closeness": 0.24444444444444444,
            "x": 245.6198524974536,
            "is_fault_confidence": 0,
            "y": 424.62117714626464,
            "is_anomalous_node": false,
            "id": "2-923",
            "betweenness": 0
        },
        {
            "in_degree": 2,
            "closeness": 0.29411764705882354,
            "x": 392.51003260939467,
            "is_fault_confidence": 0,
            "y": 361.5993070864219,
            "is_anomalous_node": false,
            "id": "2-924",
            "betweenness": 0
        },
        {
            "in_degree": 2,
            "closeness": 0.30434782608695654,
            "x": 770.3899312889758,
            "is_fault_confidence": 0,
            "y": 287.94988407715283,
            "is_anomalous_node": false,
            "id": "2-956",
            "betweenness": 0
        },
        {
            "in_degree": 1,
            "closeness": 1,
            "x": 295.7576202525015,
            "is_fault_confidence": 0,
            "y": -303.6336221464484,
            "is_anomalous_node": false,
            "id": "2-974",
            "betweenness": 0
        },
        {
            "in_degree": 1,
            "closeness": 0.30434782608695654,
            "x": 669.2584931502406,
            "is_fault_confidence": 0,
            "y": 287.6822182157895,
            "is_anomalous_node": false,
            "id": "2-988",
            "betweenness": 0
        },
        {
            "in_degree": 1,
            "closeness": 0.375,
            "x": 772.5312675214793,
            "is_fault_confidence": 0,
            "y": 165.52550255554968,
            "is_anomalous_node": false,
            "id": "2-996",
            "betweenness": 0
        },
        {
            "in_degree": 1,
            "closeness": 0.375,
            "x": 672.6576676118744,
            "is_fault_confidence": 0,
            "y": 163.06855663593421,
            "is_anomalous_node": false,
            "id": "2-997",
            "betweenness": 0
        },
        {
            "in_degree": 1,
            "closeness": 0.5,
            "x": 667.7668946059308,
            "is_fault_confidence": 0,
            "y": -0.7552800232375176,
            "is_anomalous_node": false,
            "id": "2-1004",
            "betweenness": 0
        },
        {
            "in_degree": 4,
            "closeness": 0.8,
            "x": 722.4264513434847,
            "is_fault_confidence": 0,
            "y": -305.1547045008017,
            "is_anomalous_node": true,
            "id": "2-1018",
            "betweenness": 0
        },
        {
            "in_degree": 1,
            "closeness": 0.5,
            "x": 774.18372796212,
            "is_fault_confidence": 0,
            "y": -2.2624243669123913,
            "is_anomalous_node": false,
            "id": "2-1025",
            "betweenness": 0
        },
        {
            "in_degree": 1,
            "closeness": 0,
            "x": 726.6736624252453,
            "is_fault_confidence": 0,
            "y": -651.1721727567883,
            "is_anomalous_node": false,
            "id": "2-1032",
            "betweenness": 0
        },
        {
            "in_degree": 0,
            "closeness": 0.5,
            "x": 1008.0757787039109,
            "is_fault_confidence": 0,
            "y": -404.07028183718904,
            "is_anomalous_node": false,
            "id": "2-1046",
            "betweenness": 0
        },
        {
            "in_degree": 2,
            "closeness": 1,
            "x": 507.9649369267684,
            "is_fault_confidence": 0,
            "y": -1.5166115759666283,
            "is_anomalous_node": false,
            "id": "2-1053",
            "betweenness": 0
        },
        {
            "in_degree": 0,
            "closeness": 0.6666666666666666,
            "x": 388.00444650370207,
            "is_fault_confidence": 0,
            "y": 129.1080909165712,
            "is_anomalous_node": false,
            "id": "2-1060",
            "betweenness": 0
        },
        {
            "in_degree": 0,
            "closeness": 0.5,
            "x": 929.2024032476478,
            "is_fault_confidence": 0,
            "y": -184.16931026467418,
            "is_anomalous_node": true,
            "id": "2-1067",
            "betweenness": 0
        },
        {
            "in_degree": 1,
            "closeness": 0,
            "x": 929.2024032476478,
            "is_fault_confidence": 0,
            "y": -264.550922422434,
            "is_anomalous_node": false,
            "id": "2-1074",
            "betweenness": 0
        }
    ],
    "graph_name": "demo",
    "edges": [
        {
            "source": "2-872",
            "target": "2-956"
        },
        {
            "source": "2-874",
            "target": "2-956"
        },
        {
            "source": "2-1053",
            "target": "2-895"
        },
        {
            "source": "2-974",
            "target": "2-909"
        },
        {
            "source": "2-916",
            "target": "2-924"
        },
        {
            "source": "2-923",
            "target": "2-924"
        },
        {
            "source": "2-924",
            "target": "2-988"
        },
        {
            "source": "2-924",
            "target": "2-1053"
        },
        {
            "source": "2-956",
            "target": "2-996"
        },
        {
            "source": "2-1018",
            "target": "2-974"
        },
        {
            "source": "2-988",
            "target": "2-997"
        },
        {
            "source": "2-996",
            "target": "2-1025"
        },
        {
            "source": "2-997",
            "target": "2-1004"
        },
        {
            "source": "2-1004",
            "target": "2-1018"
        },
        {
            "source": "2-1025",
            "target": "2-1018"
        },
        {
            "source": "2-1046",
            "target": "2-1018"
        },
        {
            "source": "2-1067",
            "target": "2-1018"
        },
        {
            "source": "2-1018",
            "target": "2-1074"
        },
        {
            "source": "2-1018",
            "target": "2-1032"
        },
        {
            "source": "2-1060",
            "target": "2-1053"
        }
    ]
}

# 构造 DiGraph 并读取节点属性
G = nx.DiGraph()
pos = {}
node_colors = []
node_sizes = []

for n in data['nodes']:
    nid = n['id']
    G.add_node(nid)
    # networkx 要的 pos 是 dict：{node: (x,y)}
    pos[nid] = (n['x'], n['y'])
    # 异常节点用红色，大号；其余用蓝色、小号
    if n['is_anomalous_node']:
        node_colors.append('red')
        node_sizes.append(400)
    else:
        node_colors.append('skyblue')
        node_sizes.append(200)

# 添加有向边
for e in data['edges']:
    G.add_edge(e['source'], e['target'])

# 绘图
plt.figure(figsize=(12, 8))
nx.draw_networkx_nodes(G, pos,
                       node_color=node_colors,
                       node_size=node_sizes,
                       edgecolors='black')
nx.draw_networkx_labels(G, pos, font_size=8)
nx.draw_networkx_edges(G, pos,
                       arrowstyle='->',
                       arrowsize=12,
                       edge_color='gray',
                       connectionstyle='arc3,rad=0.1')

plt.axis('off')
plt.title(data.get('graph_name', 'Graph'))
plt.show()
