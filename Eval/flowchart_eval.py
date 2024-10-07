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

def transform_dot_2_json(dot_content):
    # tools = Tools()
    # dot_c = tools.read_file("data/datasets/Flowchart2DotDatasets/dotV1/Dot/儿科/Apgar 评分/0_all.dot")
    # dot_c = "data/datasets/Flowchart2DotDatasets/dotV1/Dot/儿科/Apgar 评分/0_all.dot".read()
    P_list = pydot.graph_from_dot_data(dot_content)

    # dot_c = nx.nx_pydot.read_dot("data/datasets/Flowchart2DotDatasets/dotV1/Dot/儿科/Apgar 评分/0_all.dot")
    # return
    # Convert only the first such instance into a NetworkX graph.
    if P_list!=None and len(P_list)>0:
        dot_c = nx.nx_pydot.from_pydot(P_list[0])
        dot_graph = nx.Graph(dot_c)
        # dot_graph = nx.Graph(dot_content)
        json_graph = nx.node_link_data(dot_graph)
        return {"nodes": json_graph["nodes"], "edges": json_graph["links"]}
    else:
        print(dot_content)
        return {"nodes": [], "edges": []}

def eval_dot_format(dot_content):
    try:
        graph = Source(dot_content)
        return True
    except:
        return False

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

