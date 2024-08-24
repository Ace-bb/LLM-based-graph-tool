import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches

# 定义流程图节点
nodes = {
    "Start": (0.1, 0.8),
    "Process 1": (0.1, 0.6),
    "Decision": (0.3, 0.6),
    "Process 2": (0.5, 0.6),
    "End": (0.5, 0.4)
}

# 定义节点之间的连接
edges = [
    ("Start", "Process 1"),
    ("Process 1", "Decision"),
    ("Decision", "Process 2"),
    ("Process 2", "End")
]

# 创建一个空白的绘图区域
fig, ax = plt.subplots(figsize=(4.48, 4.48))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# 保存节点的bounding box
bounding_boxes = {}

# 绘制节点
for node, (x, y) in nodes.items():
    bbox = dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightblue")
    text = ax.text(x, y, node, ha="center", va="center", fontsize=12, bbox=bbox)
    
    # 获取文本框的bounding box
    renderer = fig.canvas.get_renderer()
    bbox_data = text.get_window_extent(renderer=renderer)
    bounding_boxes[node] = bbox_data.bounds  # 保存bounding box，格式为(x0, y0, width, height)

# 绘制连接
for start, end in edges:
    start_pos = nodes[start]
    end_pos = nodes[end]
    ax.annotate("",
                xy=end_pos, xycoords='data',
                xytext=start_pos, textcoords='data',
                arrowprops=dict(arrowstyle="->", lw=2.0))

# 隐藏坐标轴
ax.axis('off')

# 保存图片
plt.savefig('flowchart.png', dpi=100, bbox_inches='tight')

# 保存bounding boxes
with open('bounding_boxes.txt', 'w') as f:
    for node, bbox in bounding_boxes.items():
        f.write(f"{node}: {bbox}\n")

# 显示图片
plt.show()

# 打开图片并调整大小
image = Image.open('flowchart.png')
image = image.resize((448, 448), Image.ANTIALIAS)
image.save('flowchart_448x448.png')
