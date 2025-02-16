import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 示例数据（替换为你实际的 JSON 数据）
lines = [
    {
        "start": [247079.4469294449, -4004.319230611727],
        "end": [245481.3856039633, -2406.257905130158],
        "entity": "LINE"
    },
    {
        "start": [249499.4469294449, -4004.319230611727],
        "end": [247901.3856039633, -2406.257905130158],
        "entity": "LINE"
    },
    {
        "start": [245481.3856039633, -2406.257905130158],
        "end": [245481.3856039633, -4056.257905130158],
        "entity": "LINE"
    },
    {
        "start": [247901.3856039633, -4056.257905130158],
        "end": [247901.3856039633, -2406.257905130158],
        "entity": "LINE"
    },
    {
        "start": [247079.4469294449, -5654.319230611727],
        "end": [245481.3856039633, -4056.257905130158],
        "entity": "LINE"
    },
    {
        "start": [249499.4469294449, -5654.319230611727],
        "end": [247901.3856039633, -4056.257905130158],
        "entity": "LINE"
    },
    {
        "start": [245481.3856039633, -4056.257905130158],
        "end": [247901.3856039633, -4056.257905130158],
        "entity": "LINE"
    },
    {
        "start": [247901.3856039633, -2406.257905130158],
        "end": [245481.3856039633, -2406.257905130158],
        "entity": "LINE"
    }
]

# 创建画布和坐标轴
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title("DWG Line Structure Visualization")
ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")

# 绘制所有线段
for line in lines:
    start = line["start"]
    end = line["end"]
    ax.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        color='black',  # 可根据 line["color"]["rgb"] 提取颜色
        linewidth=1,
        linestyle='-'
    )

# 设置等比例坐标轴（保持图形不变形）
ax.set_aspect('equal', adjustable='datalim')

# 自动调整坐标范围
ax.autoscale()

# 显示网格
ax.grid(True, linestyle='--', alpha=0.7)

# 保存为图片（可选）
# plt.savefig('dwg_structure.png', dpi=300, bbox_inches='tight')

# 显示图形
plt.show()