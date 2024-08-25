import pydot
import json

def read_dot_file(file_path):
    # 读取DOT文件内容
    with open(file_path, 'r', encoding="utf-8") as file:
        dot_string = file.read()
    return dot_string

def dot_to_json(dot_file_path):
    # 解析DOT格式字符串
    graphs = pydot.graph_from_dot_data(read_dot_file(dot_file_path))
    
    if not graphs:
        return None

    # 我们假设只有一个graph在DOT文件中
    if len(graphs)>1: print(f"### {dot_file_path}")
    graph = graphs[0]
    
    # 初始化JSON结构
    graph_json = {
        'directed': graph.get_type() == 'digraph',
        'nodes': [],
        'edges': []
    }
    
    # 添加节点信息
    for node in graph.get_nodes():
        graph_json['nodes'].append({
            'id': node.get_name(),
            'label': node.get_label() or node.get_name(),
            'attributes': node.get_attributes()
        })

    # 添加边信息
    for edge in graph.get_edges():
        graph_json['edges'].append({
            'source': edge.get_source(),
            'target': edge.get_destination(),
            'attributes': edge.get_attributes()
        })
    return graph_json

    return json.dumps(graph_json, indent=2)

# def 
# 示例DOT格式字符串
dot_string = """
digraph G {
    A [label="Node A"];
    B [label="Node B"];
    A -> B [label="Edge from A to B"];
}
"""

# 转换为JSON格式
json_output = dot_to_json(dot_string)
print(json_output)
