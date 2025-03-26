import networkx as nx
import matplotlib.pyplot as plt
import random

# 创建一个空的有向图
forest = nx.DiGraph()

# 定义树的数量
num_trees = 2

# 为每棵树生成节点和边
for i in range(num_trees):
    # 创建一棵有向树
    tree = nx.DiGraph()

    # 随机生成节点数量，设置范围在4到12个节点之间
    num_nodes = random.randint(4, 12)

    # 生成一条主干（较长的路径），主干长度为节点数的一半左右
    main_trunk_length = random.randint(num_nodes // 2, num_nodes - 1)
    main_trunk = list(range(main_trunk_length + 1))

    # 添加主干的边
    for j in range(main_trunk_length):
        tree.add_edge(main_trunk[j], main_trunk[j + 1])

    # 为主干上的某些节点添加短的枝节（只添加少量节点形成枝节）
    for j in range(random.randint(1, 3)):  # 随机选择1到3个节点作为枝节
        parent_node = random.choice(main_trunk)  # 从主干上选择一个节点
        num_branches = random.randint(1, 2)  # 每个枝节最多添加1到2个节点
        for _ in range(num_branches):
            child_node = max(tree.nodes) + 1  # 新节点
            tree.add_edge(parent_node, child_node)
            parent_node = child_node  # 将当前节点作为下一个节点的父节点

    # 在树中随机加入环，形成带环的树（添加少量环以保持树的形态）
    if random.random() > 0.5:  # 50% 概率带环
        u, v = random.sample(main_trunk, 2)  # 随机选择两个主干上的节点
        tree.add_edge(u, v)

    # 将树添加到森林图中
    forest = nx.compose(forest, tree)

# 绘制森林图
plt.figure(figsize=(8, 6))
pos = nx.spring_layout(forest, seed=42)  # 使用 spring layout 进行布局
nx.draw(forest, pos, with_labels=True, node_color='lightblue', font_weight='bold', node_size=700, arrowsize=20)
plt.title("Directed Forest with Main Trunk and Short Branches")
plt.show()
