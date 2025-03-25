import os, json, copy,re
from graphviz import Source
import networkx as nx
from networkx.readwrite import json_graph
from conf.Tools import Tools
from lmdeploy import pipeline
from lmdeploy.vl import load_image
from tqdm import tqdm
from openai import OpenAI
import base64
import pydot
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from rich.console import Console
print = Console().log

def transform_dot_2_json(dot_content):
    P_list = pydot.graph_from_dot_data(dot_content)

    if P_list!=None and len(P_list)>0:
        dot_c = nx.nx_pydot.from_pydot(P_list[0])
        dot_graph = nx.Graph(dot_c)
        # dot_graph = nx.Graph(dot_content)
        json_graph = nx.node_link_data(dot_graph)
        return {"nodes": json_graph["nodes"], "edges": json_graph["links"]}
    else:
        print("AXSAXAXSF")
        print(dot_content)
        return {"nodes": [], "edges": []}

def eval_dot_format(dot_content):
    try:
        graph = Source(dot_content)
        return True
    except:
        return False

def eval_dot_render_success_rate(model_results):
    """评估dot文件的渲染成功率

    Args:
        dot_contents (list): 全部dot内容

    Returns:
        _type_: float
    """
    success_num = 0
    for f_name in model_results.keys():
        P_list = pydot.graph_from_dot_data(model_results[f_name]['llm'])
        if P_list!=None and len(P_list)>0: 
            graph = P_list[0]
            os.makedirs(os.path.dirname(f"output/tmp_imgs/{f_name}"), exist_ok=True)
            try:
                graph.write_png(f"output/tmp_imgs/{f_name}.png")
                success_num +=1
            except Exception as e:
                print(e)
                print(f"Error in {f_name}")
        
    return success_num/len(model_results.keys())
    
def eval_flowchart_nodes(model_output_nodes, golden_output_nodes):
    """评估流程图中的节点的准确率，精确率和召回率

    Args:
        model_output_nodes (list): 模型预测的流程图节点
        golden_output_nodes (list): 流程图节点的真实值
    """
    model_output_nodes = [node.strip().replace("\n", "").lower() for node in model_output_nodes]
    golden_output_nodes = [node.strip().replace("\n", "").lower() for node in golden_output_nodes]
    right_num = 0
    for node in model_output_nodes:
        if node in golden_output_nodes: right_num +=1
    accuracy = right_num/len(model_output_nodes)
    precision = right_num/len(golden_output_nodes)
    recall = right_num/len(model_output_nodes)
    return accuracy, precision, recall
    
    model_output_nodes_map = {k:i for i,k in enumerate(list(set(model_output_nodes)))}
    golden_output_nodes_map = {k:i for i,k in enumerate(list(set(golden_output_nodes)))}
    model_pred = [model_output_nodes_map[node] for node in model_output_nodes]
    golden_pred = [golden_output_nodes_map[node] for node in golden_output_nodes]
    accuracy = accuracy_score(golden_pred, model_pred)
    precision = precision_score(golden_pred, model_pred, average='macro')
    recall = recall_score(golden_pred, model_pred, average='macro')
    
    return accuracy, precision, recall

def extract_str(s):
    """提取字符串中的全部英文字母和中文汉字，去除其他字符"""
    return ''.join(re.findall(r'[\u4e00-\u9fa5a-zA-Z]', s))

def eval_flowchart_edges(model_output_edges, golden_output_edges):
    """评估流程图中的连接线的准确率，精确率和召回率

    Args:
        model_output_edges (list): 模型预测的流程图连接线
        golden_output_edges (list): 流程图连接线的真实值
    """
    model_output_edges = [f"{edge['source']}->{edge['target']}".lower() for edge in model_output_edges]
    golden_output_edges = [f"{edge['source']}->{edge['target']}".lower() for edge in golden_output_edges]
    right_num = 0
    for edge in model_output_edges:
        if edge in golden_output_edges: right_num +=1
    accuracy = right_num/len(model_output_edges)
    precision = right_num/len(golden_output_edges)
    recall = right_num/len(model_output_edges)
    return accuracy, precision, recall
    
    
    model_output_edges_map = {k:i for i,k in enumerate(list(set(model_output_edges)))}
    golden_output_edges_map = {k:i for i,k in enumerate(list(set(golden_output_edges)))}
    model_pred = [model_output_edges_map[edge] for edge in model_output_edges]
    golden_pred = [golden_output_edges_map[edge] for edge in golden_output_edges]
    accuracy = accuracy_score(golden_pred, model_pred)
    precision = precision_score(golden_pred, model_pred, average='macro')
    recall = recall_score(golden_pred, model_pred, average='macro')
    
    return accuracy, precision, recall

def eval_llm_result(model_result_path):
    """评估模型将流程图转换为Dot格式的结果

    Args:
        model_result_path (str): 模型结果存储的地址
    """
    tools = Tools()
    model_results = tools.read_json(model_result_path)
    all_eval_results = []
    model_output_nodes, model_output_edges = [], []
    golden_output_nodes, golden_output_edges = [], []
    model_output_dots = []
    for key, result in model_results.items():
        if "DifferentialDiagnosisEnglish" not in key: continue
        print(key)
        model_output_dots.append(result["llm"])
        model_output = transform_dot_2_json(result["llm"])
        
        model_id2node = {}
        for node in model_output['nodes']:
            if "label" in node.keys():
                label = extract_str(node['label']) # node['label'].strip().replace("\n", "").replace("\"", "")
                model_output_nodes.append(label)
                model_id2node[node['id']] = label
            elif "id" in node.keys():
                label = extract_str(node['id']) # node['id'].strip().replace("\n", "").replace("\"", "")
                model_output_nodes.append(label)
                model_id2node[node['id']] = label
            else:    print(f"===={node}====")
        # tools.write_2_json([k for k in model_id2node.keys()], f"output/test/{key}/model.json")
        for edge in model_output['edges']:
            model_output_edges.append({
                "source": model_id2node[edge['source']],
                "target": model_id2node[edge['target']]
            })
            # model_output_edges.append(f"{edge['source']}->{edge['target']}")
        
        golden_output = result["json"]
        # tools.write_2_json([n['Name'] for n in golden_output['nodes']], f"output/test/{key}/golden.json")
        id2node = {}
        for node in golden_output['nodes']:
            label = extract_str(node['Name']) # node['Name'].strip().replace("\n", "").replace("\"", "")
            golden_output_nodes.append(label)
            id2node[str(node['id'])] = label
        for edge in golden_output['edges']:
            # golden_output_edges.append(f"{id2node[edge['sourceNode']]}->{id2node[edge['targetNode']]}")
            if edge['sourceNode'] not in id2node.keys() or edge['targetNode'] not in id2node.keys(): continue
            golden_output_edges.append({
                "source": id2node[edge['sourceNode']],
                "target": id2node[edge['targetNode']]
            })
    # 排序
    model_output_nodes = sorted(model_output_nodes)
    golden_output_nodes = sorted(golden_output_nodes)
    tools.write_2_json(model_output_nodes, model_result_path.replace('.json', '_model_output_nodes.json'))
    tools.write_2_json(golden_output_nodes, model_result_path.replace('.json', '_golden_output_nodes.json'))
    node_accuracy, node_precision, node_recall = eval_flowchart_nodes(model_output_nodes, golden_output_nodes)
    edge_accuracy, edge_precision, edge_recall = eval_flowchart_edges(model_output_edges, golden_output_edges)
    
    eval_result = {
        "dot_format": eval_dot_render_success_rate(model_results),
        "node_accuracy": node_accuracy,
        "node_precision": node_precision,
        "node_recall": node_recall,
        "edge_accuracy": edge_accuracy,
        "edge_precision": edge_precision,
        "edge_recall": edge_recall
    }
    
    tools.write_2_json(eval_result, model_result_path.replace('.json', '_eval_result.json'))
    
def eval_json_format(model_output, golden_output):
    eval_result = {
        "dot_format": False,
        "node_number_equal": False,
        "equal_nodes": 0,
        "all_nodes": 0,
        "equal_edges": 0,
        "all_edges": 0
    }

    if eval_dot_format(model_output):
        eval_result["dot_format"] = True
    else: return eval_result
    
    golden_output = transform_dot_2_json(golden_output)
    eval_result["all_nodes"] = len(golden_output["nodes"])
    eval_result["all_edges"] = len(golden_output["edges"])

    model_output = transform_dot_2_json(model_output)
    # print(golden_output)
    nodes, edges = model_output["nodes"], model_output["edges"]
    if len(nodes)<=0 or len(edges)<=0:
        eval_result["dot_format"] = False
        return eval_result
    # 判断节点数量
    if len(nodes) != len(golden_output["nodes"]): eval_result["node_number_equal"] = False
    else: eval_result["node_number_equal"] = True

    # 判断节点内容是否符合
    golden_node_contents = {}
    id2node, node2id = {}, {}
    for node in golden_output["nodes"]:
        node_content = node["label"].strip().replace("\n", "")
        golden_node_contents[node_content] = node
        id2node[node['id']] = node_content
        node2id[node_content]=node['id']
        
    equal_nodes_num = 0
    for node in nodes:
        # print(node)
        if "label" not in node.keys(): continue
        node_content = node["label"].strip().replace("\n", "")
        if node_content in golden_node_contents.keys(): 
            equal_nodes_num +=1
        # 有节点
        # target_node = golden_node_contents[node_content]
        # if node['shape'] != target_node['shape']: return False, 2
    # print("equal_nodes", equal_nodes_num, len(golden_node_contents.keys()))
    eval_result["equal_nodes"] = equal_nodes_num
    # eval_result["all_nodes"] = len(nodes)

    # 判断连接线
    target_edges = []
    equal_edges_num = 0
    for edge in golden_output["edges"]:
        target_edges.append(f"{id2node[edge['source']]}->{id2node[edge['target']]}")
    for edge in edges:
        if edge['source'] in id2node.keys() and edge['target'] in id2node.keys(): 
            if f"{id2node[edge['source']]}->{id2node[edge['target']]}" in target_edges: equal_edges_num+=1

    eval_result["equal_edges"] = equal_edges_num
    # eval_result["all_edges"] = len(edges)

    return eval_result


def eval_model_output(inferenced_data):
    tools = Tools()
    total_num = len(inferenced_data)
    right_num = 0
    all_eval_result = []
    right_nodes, all_nodes, right_edges, all_edges = 0,0,0,0
    for item in tqdm(inferenced_data):
        model_output = re.findall(r'```dot(.*?)```', item['output'], re.DOTALL)
        if len(model_output)>0: model_output = model_output[0]
        else: continue
        golden_output = item["conversations"][-1]['value']
        if model_output==golden_output:
            right_num +=1
            continue
        eval_results = eval_json_format(model_output, golden_output)
        right_nodes+= eval_results["equal_nodes"]
        all_nodes += eval_results["all_nodes"]
        right_edges += eval_results["equal_edges"]
        all_edges += eval_results["all_edges"]
        if eval_results["dot_format"]: right_num +=1
        all_eval_result.append({"item": item, "eval_results": eval_results})

    print(right_num, total_num)
    tools.write_2_json(all_eval_result, "data/evaluation/Dot/dot_eval_result.json")

    print(right_nodes, all_nodes, right_edges, all_edges)
    print(right_nodes/all_nodes)
    print(right_edges/all_edges)


def generate_llm_output(input_data_path, llm, output_savepath):
    tools = Tools()
    input_data = tools.read_json(input_data_path)
    output_data = []
    for item in tqdm(input_data):
        img_path = item["image"]
        conversations = item["conversations"]
        human_input = conversations[0]['value'].split('\n')[-1]
        # image = load_image(img_path)
        response = llm((human_input, img_path))
        output_item = copy.deepcopy(item)
        if response!=None:
            output_item["output"] = response.text
            output_data.append(output_item)
    
    tools.write_2_json(output_data, output_savepath)

os.environ['OPENAI_API_BASE'] = "https://fast.xeduapi.com/v1"
# os.environ['OPENAI_API_BASE'] = "https://api.xeduapi.com"
OPENAI_API_KEY = "sk-jz0shLgMJY9HBVnLC3Fe3dCaA5204a418e67003f637f1eFf"

def run_gpt(args):
    human_input, img_path = args[0], args[1]
    with open(img_path, 'rb') as f:
        base64_img = base64.b64encode(f.read()).decode('utf-8')
    client = OpenAI(
        # This is the default and can be omitted
        api_key= OPENAI_API_KEY,
        base_url="https://fast.xeduapi.com/v1"
    )
    img_type = 'image/png'
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": human_input},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{img_type};base64,{base64_img}"}, # {"url": f"{img_url}"},
                    },
                ],
            }
        ]
    )
    
    dot_contents = re.findall(r'```dot(.*?)```', response.choices[0].message.content, re.DOTALL)
    print(dot_contents)
    if len(dot_contents)>0:
        return dot_contents[0]

# def generate_gpt_output(input_data_path, llm, output_savepath):

def main():
    pipe = pipeline('/root/LLM-based-graph-tool/models/InternVL2-8B-flow2json_v1')
    generate_llm_output("../data/datasets/Flowchart2DotDatasets/dotV1/datasets/flowchart2dot_eval.json", pipe, "../data/evaluation/Dot/internvl2-8B-dot-evaluation.json")

if __name__ == "__main__":
    main()

