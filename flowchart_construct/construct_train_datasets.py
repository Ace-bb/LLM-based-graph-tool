import os, json, cv2, re, random, glob
from conf.settings import *
from conf.Tools import Tools
import copy
from graphviz import Source
from graphviz import Digraph
from tqdm import tqdm
import numpy as np
from flowchart_construct.data_format import create_internvl_data
from flowchart_construct.utils import dot2image, trans2rect

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


class FlowchartDatasetConstructor:
    def __init__(self, config, model_name):
        self.config = config
        self.logger = logger
        self.r1 = 0
        self.model_name = model_name
        pass
    
    def rewrite_dot_content(self, dot_content):
        """给dot_content添加样式
        随机添加不同类型的样式，可以添加的流程图样式有：
        1. 修改节点的形状、背景颜色、边框颜色、字体颜色、字体大小、节点大小等
        2. 修改连接线的颜色、粗细、箭头样式等

        Args:
            dot_content (_type_): _description_
        """
        from prompts.rewrite_dot import get_rewrite_dot_prompt
        from LLMs.llm import LLM
        llm = LLM(self.config['api_key'], self.config['base_url'], self.model_name)
        rewrite_dot = llm.generate(get_rewrite_dot_prompt(dot_content=dot_content))
        if rewrite_dot is None: return None
        if "```dot" in rewrite_dot:
            # 正则表达式匹配```dot和```之间的内容
            rewrite_dot = rewrite_dot.strip().strip("\n")
            try:
                rewrite_dot = re.search(r'```dot(.*?)```', rewrite_dot, re.S).group(1).strip()
            except Exception as e:
                print(e)
                print(rewrite_dot)
                tools.write_2_txt(rewrite_dot, f"output/cache/{self.r1}.md")
                self.r1+=1
                return None
            return rewrite_dot
        return None

    def construct_img2dot_datasets(self, input_path, save_path):
        print("Start to construct img2dot datasets.")
        dot_path = input_path
        res_save_folder = f"data/FlowchartDatasets/TrainDatasets/{save_path}"
        all_dot_files = glob.glob(f"{dot_path}/**/*.dot", recursive=True)
        all_num = len(all_dot_files)
        random.shuffle(all_dot_files)
        
        train_datasets, eval_datasets = [], []
        def run_llm(_id, dot_content, flowchart_name):
            rewrite_dot = self.rewrite_dot_content(dot_content)
            if rewrite_dot is None: return None
            img_path = dot2image(rewrite_dot, flowchart_name, f"{res_save_folder}/Images/{flowchart_name}", "png")
            if img_path is None: return None
            tools.write_2_txt(rewrite_dot, f"{res_save_folder}/Dot/{flowchart_name}.dot")
            data = create_internvl_data(_id, img_path, rewrite_dot)
            return data
        
        run_paras = []
        num = 0
        num_map = {}
        for _id, dot_f in tqdm(enumerate(all_dot_files), total=all_num):
            flowchart_name = dot_f.replace(dot_path, '').replace('.dot', '')
            dot_content = tools.read_file(dot_f)
            # 统计dot_content中label=的数量
            node_num = dot_content.count("label=")
            if node_num < 10: continue
            dot_content = "digraph " + dot_content[dot_content.find("{"):].strip()
            # 在0-1之间随机取数
            rate = random.random()
            if rate > 0.8:
                # 保持样式
                img_path = dot2image(dot_content, flowchart_name, f"{res_save_folder}/Images/{flowchart_name}", "png")
                if img_path is None: continue
                tools.write_2_txt(dot_content, f"{res_save_folder}/Dot/{flowchart_name}.dot")
                data = create_internvl_data(_id, img_path, dot_content)
                train_datasets.append(data)
            elif rate <=0.8 and rate >=0.3:
                # 修改样式
                run_paras.append(((_id, dot_content, flowchart_name)))
                continue
                rewrite_dot = self.rewrite_dot_content(dot_content)
                if rewrite_dot is None: continue
                img_path = dot2image(rewrite_dot, flowchart_name, f"{res_save_folder}/Images/{flowchart_name}", "png")
                if img_path is None: continue
                tools.write_2_txt(rewrite_dot, f"{res_save_folder}/Dot/{flowchart_name}.dot")
                data = create_internvl_data(_id, img_path, rewrite_dot)
                train_datasets.append(data)
            else:
                # 全部节点样式修改为矩形
                rect_dot_content = trans2rect(dot_f)
                if rect_dot_content is None: continue
                img_path = dot2image(rect_dot_content, flowchart_name, f"{res_save_folder}/Images/{flowchart_name}", "png")
                if img_path is None: continue
                tools.write_2_txt(rect_dot_content, f"{res_save_folder}/Dot/{flowchart_name}.dot")
                data = create_internvl_data(_id, img_path, rect_dot_content)
                train_datasets.append(data)
            if len(train_datasets)%100==0:
                tools.write_2_json(train_datasets, f"{res_save_folder}/datasets/flowchart2dot_train.json")
        rewrited_dot_contents = tools.multi_thread_run(24, run_llm, run_paras, description="Rewrite img2dot datasets")
        for rtc in rewrited_dot_contents:
            if rtc is not None:
                train_datasets.append(rtc)
                if len(train_datasets)%100==0:
                    tools.write_2_json(train_datasets, f"{res_save_folder}/datasets/flowchart2dot_train.json")
        # train_datasets.extend(rewrited_dot_contents)
        os.makedirs(f"{res_save_folder}/datasets", exist_ok=True)
        tools.write_2_json(train_datasets, f"{res_save_folder}/datasets/flowchart2dot_train.json")
        # tools.write_2_json(eval_datasets, f"data/datasets/Flowchart2DotDatasets/{version}/datasets/flowchart2dot_eval.json")
        
        
    def construct_dot_img_train_datasets(self, input_path, ):
        flowchart_img_path = input_path
        dot_path = f"data/datasets/Flowchart2DotDatasets/{version}/Dot"
        Anont_save_path = f"data/datasets/Flowchart2DotDatasets/{version}/datasets"
        all_imgs = tools.get_all_dirs_sub_files(flowchart_img_path)
        annot_datasets = []
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


import networkx as nx
import pydot
import shutil
from tqdm import tqdm

class FlowchartFilter:
    def __init__(self):
        pass
    
    def transform_dot_2_json(self, dot_content):
        P_list = pydot.graph_from_dot_data(dot_content)

        if P_list!=None and len(P_list)>0:
            dot_c = nx.nx_pydot.from_pydot(P_list[0])
            dot_graph = nx.Graph(dot_c)
            # dot_graph = nx.Graph(dot_content)
            json_graph = nx.node_link_data(dot_graph)
            return {"nodes": json_graph["nodes"], "edges": json_graph["links"]}
        else:
            print(dot_content)
            return {"nodes": [], "edges": []}
    
    def get_tree_max_depth(self, flowchart_data):
        """获取流程图的最大深度
        flowchart_data是由nodes和edges组成的字典，nodes是节点列表，edges是边列表
        """
        visited_node_id = []
        def dfs(node_id, depth, id2node, id2edge):
            max_depth = depth
            if node_id in visited_node_id: return max_depth
            visited_node_id.append(node_id)
            for edge in id2edge.get(node_id, []):
                max_depth = max(max_depth, dfs(edge['target'], depth + 1, id2node, id2edge))
            return max_depth
        
        nodes, edges = flowchart_data['nodes'], flowchart_data['edges']
        id2node = {node['id']: node for node in nodes}
        id2edge = {}
        for edge in edges:
            if edge['source'] not in id2edge:
                id2edge[edge['source']] = []
            id2edge[edge['source']].append(edge)
        
        roots = [node['id'] for node in nodes if node['id'] not in [edge['target'] for edge in edges]]
        max_depth = 0
        for root in roots:
            max_depth = max(max_depth, dfs(root, 1, id2node, id2edge))
        
        return max_depth

    def check_tree_max_children(self, flowchart_data):
        """检查流程图的最大子节点数
        flowchart_data是由nodes和edges组成的字典，nodes是节点列表，edges是边列表

        Args:
            flowchart_data (_type_): _description_
        """
        nodes, edges = flowchart_data['nodes'], flowchart_data['edges']
        source_node = [edge['source'] for edge in edges]
        target_node = [edge['target'] for edge in edges]
        id2node = {}
        for node in nodes: id2node[node['id']] = node
        roots = []
        for sn in source_node:
            if sn not in target_node:
                roots.append(sn)
        if len(roots)>1: return False
        for root in source_node:
            root_sons = [edge['target'] for edge in edges if edge['source']==root]
            if len(root_sons) > 5: return False
        return True
    
                
    def filter(self, flowchart_save_path):
        """过滤单个流程图

        Args:
            flowchart_save_path (_type_): _description_
        """
        tools = Tools()
        dot_content = tools.read_file(flowchart_save_path)
        dot_json = self.transform_dot_2_json(dot_content)
        max_depth = self.get_tree_max_depth(dot_json)
        if max_depth <5: return False
        if self.check_tree_max_children(dot_json): return False
        return True
        
    def filter_all_dots(self, dot_flowchart_folder, save_folder):
        tools = Tools()
        all_dot_files = glob.glob(f"{dot_flowchart_folder}/Dot/**/*.dot", recursive=True)
        
        for dot_f in tqdm(all_dot_files[5840:], desc="Filtering"):
            if self.filter(dot_f):
                print(f"Filter {dot_f} successfully.")
                # 将dot文件复制到save_folder下
                dot_relate_path = dot_f.replace(dot_flowchart_folder+"/Dot/", '')
                img_relate_path = dot_relate_path.replace(".dot", ".png")
                sourcec_img_path = f"data/FlowchartDatasets/TrainDatasets/Image2DotV1/Images/{img_relate_path}"
                target_img_path = f"data/FlowchartDatasets/TrainDatasets/Image2DotV2/Images/{img_relate_path}"
                save_path = f"{save_folder}/Dot/{dot_relate_path}"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                os.makedirs(os.path.dirname(target_img_path), exist_ok=True)
                shutil.copy(dot_f, save_path)
                shutil.copy(sourcec_img_path, target_img_path)
            
            
def flowchart_operate(args):
    input_file = args.input_file
    output_file = args.output_file
    print(f"input_file:{input_file},\n output_file:{output_file}")
    if args.op_type == "construct":
        flowchart_constructor = FlowchartDatasetConstructor(load_config("qwen-plus"), "qwen-plus-2025-01-25")
        flowchart_constructor.construct_img2dot_datasets(input_path=input_file, save_path=output_file)
    elif args.op_type == "filter":
        flowchart_filter = FlowchartFilter()
        flowchart_filter.filter_all_dots(input_file, output_file)
        pass

def main():
    from dot2mermaid import dot_to_mermaid
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

        if (aspect_ratio > 2):
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