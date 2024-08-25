import json

from tqdm import tqdm
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import os
import math
class Tools:
    def __init__(self) -> None:
        self.file_lock = threading.Lock()
        pass
    
    def llm_check_yes_no(self, llm):
        ...
    
    def read_json(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def read_jsonl(self, file_path):
        json_data = list()
        with open(file_path, 'r', encoding="utf-8") as f:
            all_data = f.readlines()
            for line in tqdm(all_data, total=len(all_data), desc=f"Load {file_path}"):
                json_data.append(json.loads(line))
        return json_data
    
    # data/TreeRelateDialogs/v1/LLMChecked
    def write_2_json(self, data, save_path = 'default'):
        self.file_lock.acquire()
        if not os.path.exists('/'.join(save_path.split('/')[:-1])): os.makedirs('/'.join(save_path.split('/')[:-1]))
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        self.file_lock.release()
    
    def write_2_txt(self, data, file_path):
        if not os.path.exists('/'.join(file_path.split('/')[:-1])): os.makedirs('/'.join(file_path.split('/')[:-1]))
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(data) 
        
    def multi_thread_run(self, thread_num, func, run_data, description, *args, **kwargs):
        thread_pool = ThreadPoolExecutor(max_workers=thread_num)
        job_list=list()
        for data in tqdm(run_data, total=len(run_data), desc=f"Add {description}"):
        # for data in run_data:
            job = thread_pool.submit(func, *data)
            job_list.append(job)
        total = len(job_list)
        fi=1
        res = list()
        # with open("./utils/temp.jsonl", 'w', encoding='utf-8') as f:
        for job in tqdm(as_completed(job_list), total=total, desc=f"Finish {description}"):
        # for job in as_completed(job_list):
            r = job.result()
            try:
                if job.done() and r != None:
                    res.append(r)
                    # f.write(json.dumps(r, ensure_ascii=False)+'\n')
            except Exception as e:
                print(e)
                ...
            fi+=1
        return res
    
    def draw_tree(self, tree_name, save_path, tree_nodes):
        nodes, edges = list(), list()
        for n in tree_nodes:
            if n['id']==-1:
                continue
            nodes.append({
                "id": str(n['id']),
                "name": f"{n['id']}-p:{n['parent']}  {self.str_split_by_length(n['content'], 12)}",
                "pass": False
            })
            if n['parent']==-1:
                continue
            if n['id'] != -1 or n['parent'] != -2 or n['parent'] != -1:
                edges.append({
                    "tail": str(n["parent"]),
                    "head": str(n["id"]),
                    "label": n["label"],
                    "pass": False
                })
        self.draw_decision_tree(tree_name, nodes, edges, save_path)
        
    def draw_decision_tree(self, tree_name, tree_nodes, tree_edges, save_path = "."):
        from graphviz import Digraph
        dot = Digraph(name="pic", comment=tree_name, format="png", encoding="utf-8")

        # 绘制方向。默认自顶向下TB，BT自底向上，LR:左到右
        dot.attr('graph', rankdir='TB')

        # 定义图中的节点
        for node in tree_nodes:
            if node['pass']:
                dot.node(name=node['id'], label=node['name'], color='red', shape="box", fontname="NSimSun")
            elif 'unable' in node.keys() and node['unable']:
                dot.node(name=node['id'], label=node['name'], color='red', shape="box", fontname="NSimSun")
            else:
                dot.node(name=node['id'], label=node['name'], color='black', shape="box", fontname="NSimSun")
        # 定义节点与节点之间的边线
        # tail_name开始，head_name结束
        for edge in tree_edges:
            if edge['pass']:
                dot.edge(tail_name=edge['tail'], head_name=edge['head'], label=edge['label'], color='green', fontname="NSimSun")
            else:
                dot.edge(tail_name=edge['tail'], head_name=edge['head'], label=edge['label'], color='black', dir='none', fontname="NSimSun")    # dir=none,边线不带箭头
        # 边线标签文本

        # 存放在当前目录
        if not os.path.exists(save_path): os.makedirs(save_path)
        dot.render(filename=tree_name, directory=save_path, view=False)
        # t="dot -Tpng ttttsssstt.dot -o tttoutput.png"
        # dot -Tpng digraph.dot -o tttoutput.png
        return f"static/decision_tree_path/{tree_name}.png"
    
    def str_split_by_length(self, string, length):
        n = len(string)//length
        if n < 1: return string
        for i in range(1, n+1):
            tmp_string = list(string)
            tmp_string.insert(i*length, "\n")
            string = "".join(tmp_string)
        return string
    
    def calculate_distance(self, p1, p2):
        return math.sqrt(pow(p1[0]-p2[0], 2) + pow(p1[1]-p2[1], 2))
    
    def get_all_dirs_sub_files(self, dir_path):
        all_files = list()
        all_dirs = list()
        all_file_names = list()
        for root,dirs,files in os.walk(dir_path):
            if len(files)!=0:
                for f in files:
                    # all_files.append(os.path.join(root, f))
                    all_files.append({
                        "filename": f,
                        "dirs": root.replace(dir_path+os.sep, ''),
                        "filepath": os.path.join(root, f)
                    })
        return all_files
