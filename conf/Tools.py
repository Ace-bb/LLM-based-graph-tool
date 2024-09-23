
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
    
    def read_json(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def read_jsonl(self, jsonl_path):
        json_data = list()
        with open(jsonl_path, 'r', encoding="utf-8") as f:
            all_data = f.readlines()
            for line in tqdm(all_data, total=len(all_data), desc=f"Load {jsonl_path}"):
                json_data.append(json.loads(line))
        return json_data
    
    def read_file(self, file_path):
        with open(file_path, 'r', encoding="utf-8") as f:
            return f.read()
        
    # data/TreeRelateDialogs/v1/LLMChecked
    def write_2_json(self, data, file_path = 'default'):
        self.file_lock.acquire()
        if not os.path.exists('/'.join(file_path.split('/')[:-1])): os.makedirs('/'.join(file_path.split('/')[:-1]))
        with open(file_path, 'w', encoding='utf-8') as f:
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
            # print(f"dirs:{root}----files:{files}")
            if len(files)!=0:
                for f in files:
                    # all_files.append(os.path.join(root, f))
                    all_files.append({
                        "filename": f,
                        "dirs": root.replace(dir_path+os.sep, ''),
                        "filepath": os.path.join(root, f)
                    })
        self.write_2_json(all_files, "./alf.json")
        return all_files

    def draw_image_annotation(self, img_path, annots):
        import cv2

        # 读取图片
        img = cv2.imread(img_path)

        # 遍历注释并绘制图形
        for annot in annots:
            if annot['type'] == 'rectangle':
                # 绘制矩形
                cv2.rectangle(img, (annot['x1'], annot['y1']), (annot['x2'], annot['y2']), annot['color'], annot['thickness'])
            elif annot['type'] == 'circle':
                # 绘制圆形
                cv2.circle(img, (annot['center_x'], annot['center_y']), annot['radius'], annot['color'], annot['thickness'])
            elif annot['type'] == 'line':
                # 绘制直线
                cv2.line(img, (annot['x1'], annot['y1']), (annot['x2'], annot['y2']), annot['color'], annot['thickness'])
            elif annot['type'] == 'text':
                # 绘制文本
                cv2.putText(img, annot['text'], (annot['x'], annot['y']), cv2.FONT_HERSHEY_SIMPLEX, annot['font_scale'], annot['color'], annot['thickness'])

        # 保存绘制后的图片
        output_path = img_path.replace('.jpg', '_annotated.jpg')
        cv2.imwrite(output_path, img)
        return output_path
    
    def save_dot_2_img(self, dot_content, img_path):
        from graphviz import Source
        graph = Source(dot_content)
        # 保存 .dot 源文件
        if not os.path.exists(os.path.dirname(img_path)): os.makedirs(os.path.dirname(img_path))
        # print(f"img_path:{img_path}")
        graph.render(img_path, format='png', cleanup=True) 