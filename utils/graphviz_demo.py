import graph_tool.all as gt
import numpy as np

# 创建一个新的图
g = gt.Graph()

# 添加顶点
v1 = g.add_vertex()
v2 = g.add_vertex()
v3 = g.add_vertex()

# 添加边
e1 = g.add_edge(v1, v2)
e2 = g.add_edge(v2, v3)

# 设置位置，这里使用简单的布局算法
pos = gt.sfdp_layout(g)

# 绘制图
gt.graph_draw(g, pos=pos, output_size=(448, 448), output="flowchart.png", fit_view=True, fit_view_ink=True)

# 计算并保存每个节点的bounding box
vertex_boxes = {}
for v in g.vertices():
    print(pos[v])
    x, y = pos[v].a
    # 假设每个节点的标签和边框是一个正方形，边长30
    vertex_boxes[v] = (x-15, y-15, x+15, y+15)

# 输出每个节点的bounding box
print(vertex_boxes)
