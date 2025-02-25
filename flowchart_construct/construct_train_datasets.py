import os, json, cv2
from conf.Tools import Tools
import copy
from graphviz import Source
from graphviz import Digraph
from tqdm import tqdm
import numpy as np
DxDiseaseTreeDots = "data/datasets/DxDiseaseTreeDots"
DxDiseaseTreeImg = "data/datasets/DotImages"
version = "dotV2"
Image_save_path = "data/datasets/Flowchart2DotDatasets/v3/images"
# DxDiseaseTreeImg = Image_save_path
Anont_save_path = "data/datasets/Flowchart2DotDatasets/v3/dataset"

tools = Tools()
class FLowchartLLM:
    def __init__(self) -> None:
        pass

    def tansform_json_2_dot(self, json_path, dot_save_path, img_save_path):
        tools = Tools()
        all_json_files = tools.get_all_dirs_sub_files(json_path)
        for jf in tqdm(all_json_files):
            print(jf['filename'])
            if not jf['filename'].endswith(".json"): continue
            json_data = tools.read_json(jf['filepath'])
            nodes_content, edges_content = "", ""
            # node['content'] = node['content'].replace("\n", "\\\n")
            nodes = []
            for node in json_data["nodes"]:
                node['content'] = node['content'].replace('\n', '')

                # nodes.append()
            nodes_content = "\n    ".join([f"{node['content']} [shape=box, label=\"{node['content']}\"];" for node in json_data["nodes"]])
            id2node = {}
            for node in json_data["nodes"]: id2node[node["id"]] = node["content"]
            edges_content = "\n    ".join([f"{id2node[edge['source']]} -> {id2node[edge['target']]};" for edge in json_data["edges"]])

            dot_content = """digraph """ + jf['dirs'].split("/")[-1] + " {" +f""" 
    fontname="conf/SIMSUN.TTC"
    fontcolor="red
    {nodes_content}

    {edges_content}            
"""+ "}"
            graph = Source(dot_content, encoding='utf-8')
            # 保存 .dot 源文件
            graph.save(f"{dot_save_path}/{jf['dirs']}/{jf['filename'].replace('.json', '.dot')}")
            graph.render(f"{img_save_path}/{jf['dirs']}/{jf['filename'].replace('.json', '')}", format='png', cleanup=True) 
    
    def find_node_sub_tree(self, root, flowchart_data, visited_node = []):
        # if root in visited_node.keys(): return visited_node[root]["nodes"], visited_node[root]["edges"]
        if root in visited_node: return None, None
        visited_node.append(root)
        nodes, edges = flowchart_data['nodes'], flowchart_data['edges']
        id2node, id2edge = {}, {}
        for node in nodes: id2node[node['id']] = node
        # for node in edges: id2node[node['id']] = node

        sub_nodes, sub_edges = [], []
        root_sons = [edge['target'] for edge in edges if edge['source']==root]
        
        for sub_id in root_sons:
            sub_nodes.append(id2node[sub_id])
            sub_edges.append({"source": root, "target": sub_id})
            tmp_sub_nodes, tmp_sub_edges = self.find_node_sub_tree(sub_id, flowchart_data, visited_node)
            # visited_node[sub_id] = {"nodes": tmp_sub_nodes, "edges": tmp_sub_edges}
            if tmp_sub_nodes!=None and tmp_sub_edges!=None:
                sub_nodes.extend(tmp_sub_nodes)
                sub_edges.extend(tmp_sub_edges)
        return sub_nodes, sub_edges



    def check_flowchart_binary(self, flowchart_data):
        # return [flowchart_data]
        nodes, edges = flowchart_data['nodes'], flowchart_data['edges']
        source_node = [edge['source'] for edge in edges]
        target_node = [edge['target'] for edge in edges]
        id2node = {}
        for node in nodes: id2node[node['id']] = node
        roots = []
        for sn in source_node:
            if sn not in target_node:
                # root = sn
                roots.append(sn)
        res_flowchats = []
        for root in set(roots):
            root_sons = [edge['target'] for edge in edges if edge['source']==root]
            if len(root_sons) > 2:
                # 多于2分支
                sub_tree_num = len(root_sons)//2
                for i in range(sub_tree_num):
                    tmp_flow = {"nodes":[id2node[root]], "edges":[]}
                    for sub_node in root_sons[2*i:2*(i+1)]:
                        tmp_flow['nodes'].append(id2node[sub_node])
                        tmp_flow['edges'].append({"source": root, "target": sub_node})
                        sub_nodes, sub_edges = self.find_node_sub_tree(sub_node, flowchart_data, [root])
                        if sub_nodes!=None and sub_edges!=None:
                            tmp_flow['nodes'].extend(sub_nodes)
                            tmp_flow['edges'].extend(sub_edges)
                    res_flowchats.append(copy.deepcopy(tmp_flow))

        return res_flowchats
    
    def start_transform_json_2_dot(self, jf, img_save_path, dot_save_path, json_save_path):
        json_data = tools.read_json(jf['filepath'])
        flowcharts = self.check_flowchart_binary(copy.deepcopy(json_data))
        all_flowcharts = [json_data]
        if len(flowcharts)>1:
            all_flowcharts.extend(flowcharts)
        # print(len(all_flowcharts))
        for fid, flow_data in enumerate(all_flowcharts):
            if fid==0:
                flowchart_img_savepath = f"{img_save_path}/{jf['dirs']}/{jf['filename'].replace('.json', '')}_all"
                flowchart_dot_savepath = f"{dot_save_path}/{jf['dirs']}/{jf['filename'].replace('.json', '')}_all.dot"
                flowchart_json_savepath = f"{json_save_path}/{jf['dirs']}/{jf['filename'].replace('.json', '')}_all.json"
            else:
                flowchart_img_savepath = f"{img_save_path}/{jf['dirs']}/{jf['filename'].replace('.json', '')}_{fid}"
                flowchart_dot_savepath = f"{dot_save_path}/{jf['dirs']}/{jf['filename'].replace('.json', '')}_{fid}.dot"
                flowchart_json_savepath = f"{json_save_path}/{jf['dirs']}/{jf['filename'].replace('.json', '')}_{fid}.json"

            tools.write_2_json(flow_data, flowchart_json_savepath)
            continue

            graph = Digraph(name="pic", comment=jf['dirs'].split("/")[-1], format="png", encoding="utf-8")
            graph.attr('graph', rankdir='TB', showboxes='2', size='540')

            shape_list = ['box', 'ellipse', 'diamond', 'parallelogram']
            SHAPE_CLASSES_RANDOM_RATE = [0.7, 0.1, 0.1, 0.1]
            all_fonts = ["Microsoft YaHei","STXihei","STKaiti","FangSong","FZShuTi","DengXian","SimHei","LiSu","NSimSun","STZhongsong"]
            font_use = np.random.choice(all_fonts)

            for i in range(len(flow_data['nodes'])):
                flow_data['nodes'][i]['shape'] = np.random.choice(shape_list, 1, p=SHAPE_CLASSES_RANDOM_RATE)[0]

            for node in flow_data['nodes']:
                graph.node(name=str(node['id']), label=node['content'], color='black', shape=node['shape'], fontname=font_use)
            
            for edge in flow_data['edges']:
                graph.edge(tail_name=str(edge['source']), head_name=str(edge['target']), color='black', fontname=font_use)
            # 保存 .dot 源文件
            graph.render(flowchart_img_savepath, format='png', cleanup=True) 
            
            from PIL import Image
            import os

            img_path = f"{flowchart_img_savepath}.png"
            img = Image.open(img_path)
            width, height = img.size
            if width / height <3:
                new_size = max(width, height)
                new_img = Image.new("RGB", (new_size, new_size), (255, 255, 255))
                new_img.paste(img, ((new_size - width) // 2, (new_size - height) // 2))
                new_img.save(img_path)
            else:
                os.remove(img_path)

            dot = Digraph(name="pic", format="png", encoding="utf-8")
            # dot.attr('graph', rankdir='TB', showboxes='2', size='540')
            
            for node in flow_data['nodes']:
                dot.node(name=str(node['id']), label=node['content'].replace('\n', ''), color='black', shape=node['shape'])
            
            for edge in flow_data['edges']:
                dot.edge(tail_name=str(edge['source']), head_name=str(edge['target']), color='black')

            dot.save(flowchart_dot_savepath)

    def tansform_json_2_dot_v2(self, json_path, dot_save_path, img_save_path, json_save_path):
        tools = Tools()
        all_json_files = tools.get_all_dirs_sub_files(json_path)
        n = 0
        run_pras = list()
        for jf in tqdm(all_json_files):
            if not jf['filename'].endswith(".json"): continue
            # if "Bartter 综合征" not in jf['dirs']: continue
            run_pras.append((jf, img_save_path, dot_save_path, json_save_path))
            continue
            # if n>20: break
        
        tools.multi_thread_run(100, self.start_transform_json_2_dot, run_pras, description="Transform")
        # print(n)

def construct_dot_img():
    flowchart_llm = FLowchartLLM()
    flowchart_llm.tansform_json_2_dot_v2(json_path="data/datasets/Flowchart2DotDatasets/DxDiseaseTreeJSON", dot_save_path=f"data/datasets/Flowchart2DotDatasets/{version}/Dot", img_save_path=f"data/datasets/Flowchart2DotDatasets/{version}/FlowchartImages", json_save_path=f"data/datasets/Flowchart2DotDatasets/{version}/JSON")

a={
    "id": 0,
    "image": "/root/LLM-based-graph-tool/data/datasets/InternVL2_flowchart_Dataset/vbase1000/Images/0.png",
    "width": 630,
    "height": 640,
    "conversations": [
        {
            "from": "human",
            "value": "<image>\nPlease provide the bounding box coordinate of the region this sentence describes: <ref>circle</ref>"
        },
        {
            "from": "gpt",
            "value": "<ref>circle</ref><box>[[278,24,354,100],[515,10,611,106],[174,299,228,353],[407,415,489,497]]</box>"
        }
    ]
}

def construct_dot_img_train_datasets():
    flowchart_img_path = f"data/datasets/Flowchart2DotDatasets/{version}/FlowchartImages"
    dot_path = f"data/datasets/Flowchart2DotDatasets/{version}/Dot"
    Anont_save_path = f"data/datasets/Flowchart2DotDatasets/{version}/datasets"
    all_imgs = tools.get_all_dirs_sub_files(flowchart_img_path)
    annot_datasets = []
    import random
    random.shuffle(all_imgs)
    train_datasets, eval_datasets = [], []
    all_imgs = all_imgs[:2200]
    all_num = len(all_imgs)
    all_questions = tools.read_json("data/DOT_Questions.json")
    
    for _id, item in tqdm(enumerate(all_imgs), total=all_num):
        if not item["filename"].endswith(".png"): continue
        img = cv2.imread(item["filepath"])
        if img is None:
            print(f"Failed to load image {item['filepath']}")
            continue
        
        height, width = img.shape[:2]
        dot_content = tools.read_file(f"data/datasets/Flowchart2DotDatasets/{version}/Dot/{item['dirs']}/{item['filename'].replace('.png', '')}.dot")
        
        data_item = {
            "id": _id,
            "image": f"/root/LLM-based-graph-tool/{item['filepath']}",
            "width": width,
            "height": height,
            "conversations": [
                {
                    "from": "human",
                    "value": """<image>\n""" + random.sample(all_questions, 1)[0]
#                     """
# {
#     "nodes": [],
#     "edges": []
# }""" # 请将图中的流程图转换成DOT格式，使用DOT语言来描述流程图。
                },
                {
                    "from": "gpt",
                    "value": f"{dot_content}"
                }
            ]
        }
        if _id < (all_num-200):
            train_datasets.append(copy.deepcopy(data_item))
        else:
            eval_datasets.append(copy.deepcopy(data_item))

    tools.write_2_json(train_datasets, f"{Anont_save_path}/flowchart2dot_train.json")
    tools.write_2_json(eval_datasets, f"{Anont_save_path}/flowchart2dot_eval.json")

def dot_to_mermaid(dot_content):
    """
    将dot格式的流程图转换成mermaid格式
    :param dot_content: str, dot格式的内容
    :return: str, mermaid格式的内容
    """
    mermaid_content = "flowchart TD\n"
    lines = dot_content.splitlines()
    for line in lines:
        line = line.strip()
        if '->' in line:
            parts = line.split('->')
            start_node = parts[0].strip()
            end_node = parts[1].strip().strip(';')
            mermaid_content += f"    {start_node} --> {end_node}\n"
        elif '[' in line and ']' in line:
            node = line.split('[')[0].strip()
            mermaid_content += f"    {node}\n"
    return mermaid_content

def main():
    tools = Tools()
    all_img_files = tools.get_all_dirs_sub_files(DxDiseaseTreeImg)
    data_num = 0
    annot_datasets = []
    print(len(all_img_files))
    for item in all_img_files:
        # print(data_num)
        if not item["filename"].endswith(".png"): continue
        img = cv2.imread(item["filepath"])
        if img is None:
            print(f"Failed to load image {item['filepath']}")
            continue
        
        height, width = img.shape[:2]
        aspect_ratio = width / height

        if aspect_ratio > 2:
            # print(aspect_ratio)
            continue
            # print(f"Image {item['filename']} has an aspect ratio greater than 2.")
        
        new_image_path = os.path.join(Image_save_path, f"{data_num}.png")
        if not os.path.exists(os.path.dirname(new_image_path)):
            os.makedirs(os.path.dirname(new_image_path))
        
        os.rename(item["filepath"], new_image_path)
        print(f"Moved image {item['filename']} to {new_image_path}")

        img_dot_path = f"{DxDiseaseTreeDots}/{item['dirs']}/{item['filename'].replace('.png', '.dot')}"
        # import graphviz
        print("11111")
        try:
            with open(img_dot_path, 'r', encoding='utf-8') as f:
                dot_content = f.read()
                mermaid_content = dot_to_mermaid(dot_content)
                print(f"mermaid_content:{mermaid_content}")
                print("\n\n")
        except Exception as e:
            print(f"Failed to process DOT file {img_dot_path}: {e}")
            continue
        annot_datasets.append({
            "id": data_num,
            "image": f"/root/LLM-based-graph-tool/{new_image_path}",
            "width": width,
            "height": height,
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\n请将图中的流程图转换成DOT格式，使用DOT语言来描述流程图。"
                },
                {
                    "from": "gpt",
                    "value": f"{dot_content}"
                }
            ]
        })

        data_num+=1
    
    tools.write_2_json(annot_datasets, f"{Anont_save_path}/flowchart2dot.json")

def draw_all_dot_img():
    tools = Tools()
    dot_path = "data/datasets/DxDiseaseTreeDots"
    all_dot_files = tools.get_all_dirs_sub_files(dot_path)
    for dot_f in all_dot_files:
        print(dot_f['filepath'])
        dot_content = tools.read_file(dot_f["filepath"])
        dot_content = dot_content.replace("[", "[shape=box,")
        try:
            tools.save_dot_2_img(dot_content, f"data/datasets/DotImages/{dot_f['dirs']}/{dot_f['filename'].split('/')[-1].replace('.dot', '')}")
        except:
            continue


def convert_dot_2_json():
    import networkx as nx
    from networkx.readwrite import json_graph

    dot_graph = nx.nx_pydot.read_dot(f"data/datasets/Flowchart2DotDatasets/{version}/Dot/儿科/Apgar 评分/0_all.dot")
    print(json_graph.dumps(dot_graph))

def merge_json_dot_train_datasets():
    json_train_dataset_path = f"data/datasets/Flowchart2DotDatasets/{version}/datasets/flowchart2json_train.json"
    dot_train_dataset_path = f"data/datasets/Flowchart2DotDatasets/{version}/datasets/flowchart2dot_train.json"
    merged_data_savepath = f"data/datasets/Flowchart2DotDatasets/{version}/datasets/flowchart2dotjson_train.json"
    tools = Tools()
    json_train_data = tools.read_json(json_train_dataset_path)
    print(len(json_train_data))
    dot_train_data = tools.read_json(dot_train_dataset_path)
    print(len(dot_train_data))
    merged_train_data = []
    _id = 0
    for item in json_train_data:
        tmp = copy.deepcopy(item)
        tmp["id"] = _id
        merged_train_data.append(tmp)
        _id +=1
    for item in dot_train_data:
        tmp = copy.deepcopy(item)
        tmp["id"] = _id
        merged_train_data.append(tmp)
        _id +=1
        
    import random
    random.shuffle(merged_train_data)
    print(len(merged_train_data))
    tools.write_2_json(merged_train_data, merged_data_savepath)

if __name__=="__main__":
    # construct_dot_img()
    construct_dot_img_train_datasets()
    # merge_json_dot_train_datasets()