Rule ='多特征策略'
FeaA = '连续特征'
FeaB = '类别特征'
FeaC = '文本特征'
FeaD = '图片特征'
FeaE = '音频特征'
method1 = '等宽分箱'
method2 = '等频分箱'
method3 = '决策树分箱'
method4 = '卡方分箱'
method5 = '聚类分箱'
method6 = '分词等'
method7 = '目标检测'
method8 = '音转文等'
stat = '统计计算'
rest = '风控策略'
Combine = '特征组合'
 
nodes = [FeaA, FeaB,FeaC,FeaD,FeaE,stat, rest, method1, method2, method3, method4,method5,method6,method7,method8,Combine]
 
import pygraphviz as pgv
 
# 创建图
G = pgv.AGraph(directed=True, 
    strict=False, 
    ranksep=0.46,#不同级节点最小间距 
    nodesep=0.42,#同级节点最小间距 
    splines="ortho", #可选 ortho (直角), polyline (折线), spline (曲线), line (线条), none (无)
    rankdir='LR',
    concentrate=True
)
 
# 添加节点 - 根节点
G.add_node(Rule,
    color="#DDA0DD", 
    style="filled",
    fontname="times bold italic" ,
    shape='egg'
)
 
# 添加节点 - 其他节点 #ffffff
G.add_nodes_from(nodes, color="#87CEEB",style="filled",fontname="times bold italic")
 
# 添加边
G.add_edges_from([
    [Rule, FeaA],
    [Rule, FeaB], 
    [Rule, FeaC],
    [Rule, FeaD], 
    [Rule, FeaE], 
    [FeaA, method1], 
    [FeaA, method2],
    [FeaA, method3], 
    [FeaA, method4], 
    [FeaA, method5], 
    [FeaC, method6], 
    [FeaD, method7], 
    [FeaE, method8],
    [method8, method6],
    [FeaB,Combine],  
    [method1, Combine],[method2, Combine], [method3, Combine], [method4, Combine],
    [method5,Combine],[method6,Combine],[method7,Combine],[method8,Combine],
    [Combine,stat],[stat,rest]
], color="#7F01FF", arrowsize=0.8)
 
 
# 设置图的尺寸 分辨率
G.graph_attr.update(dpi='800')  # 设置分辨率
 
# 导出图形
G.layout()
G.draw("多特征策略.png", prog="dot")
 
# 图片图形
from IPython.display import Image
display(Image(filename="多特征策略.png"))