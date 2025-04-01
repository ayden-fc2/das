import networkx as nx
import matplotlib.pyplot as plt
import random
import json
import os

def generate_graph(graph_idx):
    # 创建空的有向图
    forest = nx.DiGraph()

    # 定义树的数量
    num_trees = random.randint(2, 3)

    # 为每棵树生成节点和边
    for i in range(num_trees):
        tree = nx.DiGraph()
        num_nodes = random.randint(4, 12)
        main_trunk_length = random.randint(num_nodes // 2, num_nodes - 1)
        main_trunk = list(range(main_trunk_length + 1))

        # 添加主干边
        for j in range(main_trunk_length):
            tree.add_edge(main_trunk[j], main_trunk[j + 1])

        # 添加枝节
        for j in range(random.randint(1, 3)):
            parent_node = random.choice(main_trunk)
            num_branches = random.randint(1, 2)
            for _ in range(num_branches):
                child_node = max(tree.nodes) + 1
                tree.add_edge(parent_node, child_node)
                parent_node = child_node

        # 添加环
        if random.random() > 0.5:
            u, v = random.sample(main_trunk, 2)
            tree.add_edge(u, v)

        # 合并到森林
        forest = nx.compose(forest, tree)


    # 计算拓扑特征
    in_degree = dict(forest.in_degree())
    betweenness = nx.betweenness_centrality(forest)
    closeness = nx.closeness_centrality(forest)

    # 生成布局坐标
    pos = nx.spring_layout(forest, seed=42)

    # 准备JSON数据结构
    graph_data = {
        "graph_name": f"graph_{graph_idx}",
        "nodes": [],
        "edges": []
    }

    # 填充节点信息
    for node in forest.nodes():
        x, y = pos[node]
        graph_data["nodes"].append({
            "id": node,
            "x": x,
            "y": y,
            "is_anomalous_node": False,
            "is_fault_confidence": 0.0,
            "in_degree": in_degree[node], # 入度
            "betweenness": round(betweenness[node], 4), # 介数
            "closeness": round(closeness[node], 4) # 接近度
        })

    # 填充边信息
    for u, v in forest.edges():
        graph_data["edges"].append({
            "source": u,
            "target": v
        })

    # 保存图片
    plt.figure(figsize=(8, 6))
    nx.draw(forest, pos, with_labels=True, node_color='lightblue',
            font_weight='bold', node_size=700, arrowsize=20)
    plt.title(f"Graph {graph_idx}")
    plt.savefig(f"imgs/graph_{graph_idx}.png")
    plt.close()

    return graph_data

# 主程序
def main():
    # 创建目录
    os.makedirs("imgs", exist_ok=True)

    # 设置要生成的图数量
    num_graphs = 100  # 可以修改为你需要的数量

    # 生成所有图数据
    all_graphs = []
    for i in range(1, num_graphs + 1):
        graph_data = generate_graph(i)
        all_graphs.append(graph_data)
        print(f"已生成图 {i}/{num_graphs}")

    # 保存到JSON文件
    with open("train.json", "w") as f:
        json.dump(all_graphs, f, indent=2)

    print(f"\n已完成！共生成 {num_graphs} 张图，数据已保存到 train.json")

if __name__ == "__main__":
    main()