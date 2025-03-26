from graphviz import Digraph
import pydot
import json, os

def read_dot_file(file_path):
    # 读取DOT文件内容
    with open(file_path, 'r', encoding="utf-8") as file:
        dot_string = file.read()
    return dot_string

def dot2image(dot_string: str, file_name, output_file: str, format: str = 'png'):
    os.makedirs(os.path.dirname(output_file+f".{format}"), exist_ok=True)
    dot_string = dot_string[dot_string.find('{'):]
    # print("===========\ndot2image: ", dot_string, "===========\n")
    try:
        dot = Digraph(comment=file_name, body=dot_string)
        dot.attr(fontname='SimSun')
        dot.attr(dpi='150')
        dot.render(output_file, format=format, cleanup=True)
    except Exception as e:
        print("===========\ndot2image: ", dot_string, "===========\n")
        
    return output_file+".png"

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

def json_2_dot(flowchart_title, json_data):
    """
    将JSON格式的流程图数据转换成Dot格式
    Args:
        json_data: JSON格式的流程图数据
        output_type: 输出graphviz格式数据
    Returns:
        dot_data: Dot格式的流程图数据
    """
    nodes = json_data["nodes"]
    edges = json_data["edges"]
    dot_data = ""
    dot = Digraph()
    for node in nodes:
        dot.node(name=node["id"], **node['attributes'])
    for edge in edges:
        dot.edge(edge["source"], edge["target"], **edge['attributes'])
    return dot.source

def trans2rect(dot_path):
    json_data = dot_to_json(dot_path)
    if json_data is None: return None
    for nid in range(len(json_data['nodes'])):
        json_data['nodes'][nid]['attributes']['shape'] = 'rect'
    return json_2_dot('rect', json_data).replace("\\\"", '')