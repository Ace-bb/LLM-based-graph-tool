"""
将JSON格式的流程图数据转换成Dot格式和mermaid格式
"""
from graphviz import Digraph
from python_mermaid.diagram import (
    MermaidDiagram,
    Node,
    Link
)

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
    dot = Digraph(comment=flowchart_title)
    for node in nodes:
        dot.node(str(node["id"]), node["Name"])
    for edge in edges:
        dot.edge(edge["sourceNode"], edge["targetNode"], label=edge["label"])
    return dot.source

def json_2_mermaid(flowchart_title, json_data):
    """
    将JSON格式的流程图数据转换成mermaid格式
    Args:
        json_data: JSON格式的流程图数据
        output_type: 输出格式，支持mermaid格式
    Returns:
        dot_data: Mermaid格式的流程图数据
    """
    nodes = json_data["nodes"]
    edges = json_data["edges"]
    dot_data = ""
    mermaid_nodes, mermaid_edges = [], []
    for node in nodes:
        mermaid_nodes.append(Node(id=str(node["id"]), content=node["Name"]))
    for edge in edges:
        mermaid_edges.append(Link(origin=edge["sourceNode"], end=edge["targetNode"], message=edge["label"]))
    diagram = MermaidDiagram(flowchart_title, nodes=mermaid_nodes, links=mermaid_edges)
    return str(diagram)