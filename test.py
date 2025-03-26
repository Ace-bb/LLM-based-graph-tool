from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from glob import glob
from conf.Tools import Tools
import os, json
def run_langchin():
    os.environ['OPENAI_API_BASE'] = "https://fast.xeduapi.com/v1"
    # os.environ['OPENAI_API_BASE'] = "https://api.xeduapi.com"
    OPENAI_API_KEY = "sk-jz0shLgMJY9HBVnLC3Fe3dCaA5204a418e67003f637f1eFf"
    MODEL_NAME = 'gpt-4o'


    llm = ChatOpenAI(model=MODEL_NAME, api_key=OPENAI_API_KEY)

    # response = llm.invoke("What can you do?")

    human_message = HumanMessage(content=[
        { "type": "text", "text": "请将图中的流程图转换成DOT格式，使用DOT语言来描述流程图。" },
        { "type": "image_url", "image_url": { "url": "https://github.com/Ace-bb/Image_server/blob/main/flowchart/%E5%86%85%E5%88%86%E6%B3%8C%E7%A7%91/POEMS%20%E7%BB%BC%E5%90%88%E5%BE%81/1_1.png?raw=true" }}
    ])
    response = llm.invoke([ human_message ])

    print(response.content)

def copy_images():
    from conf.Tools import Tools
    import shutil
    tools = Tools()
    datas = tools.read_json("/root/LLM-based-graph-tool/data/datasets/Flowchart2DotDatasets/dotV1/datasets/flowchart2dot_eval.json")
    for item in datas:
        copy_save_path = "/root/LLM-based-graph-tool/repo/Image_server/flowchart/" + item["image"].replace("/root/LLM-based-graph-tool/data/datasets/Flowchart2DotDatasets/dotV1/FlowchartImages/", '')
        base_dir = os.path.dirname(copy_save_path)
        if not os.path.exists(base_dir): os.makedirs(base_dir)
        shutil.copyfile(item["image"], copy_save_path)
    
def run_openai():
    from openai import OpenAI
    import base64
    os.environ['OPENAI_API_BASE'] = "https://fast.xeduapi.com/v1"
    # os.environ['OPENAI_API_BASE'] = "https://api.xeduapi.com"
    OPENAI_API_KEY = "sk-jz0shLgMJY9HBVnLC3Fe3dCaA5204a418e67003f637f1eFf"

    with open("/root/LLM-based-graph-tool/data/datasets/Flowchart2DotDatasets/dotV1/FlowchartImages/妇产科/产后关节响/0_1.png", 'rb') as f:
        base64_img = base64.b64encode(f.read()).decode('utf-8')
    client = OpenAI(
        # This is the default and can be omitted
        api_key= OPENAI_API_KEY,
        base_url="https://fast.xeduapi.com/v1"
    )
    img_type = 'image/png'
    img_url = "https://github.com/Ace-bb/Image_server/blob/main/flowchart/%E5%86%85%E5%88%86%E6%B3%8C%E7%A7%91/POEMS%20%E7%BB%BC%E5%90%88%E5%BE%81/1_1.png?raw=true"
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "请将图中的流程图转换成DOT格式，使用DOT语言来描述流程图。"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{img_type};base64,{base64_img}"}, # {"url": f"{img_url}"},
                    },
                ],
            }
        ]
    )

    print(response.choices[0].message.content)

def test():
    import re
    s = '''啊沙发沙发大事发生
    ```dot
digraph G {
    node [shape=box];
    节点1[label="产后关节有 \n 何痛怎么办?"];
    节点2[label="调整站姿,\n保护关节"];
    节点3[shape=oval,label="停止哺乳并\n 休息"];

    节点1 -> 节点2;
    节点1 -> 节点3;
}
```阿迪斯发啊实打实打算
'''
    dot_contents = re.findall(r'```dot(.*?)```', s, re.DOTALL)
    print(dot_contents[0])

def mermaid():
    # Creating a simple flowchart diagram
    from python_mermaid.diagram import (
        MermaidDiagram,
        Node,
        Link
    )

    # Family members
    meg = Node("Meg")
    jo = Node("Jo")
    beth = Node("Beth")
    amy = Node("Amy")
    robert = Node("Robert March")

    the_march_family = [meg, jo, beth, amy, robert]

    # Create links
    family_links = [
        Link(robert, meg),
        Link(robert, jo),
        Link(robert, beth),
        Link(robert, amy),
    ]

    chart = MermaidDiagram(
        title="Little Women",
        nodes=the_march_family,
        links=family_links
    )

    print(chart)

def log():
    from rich.console import Console
    # 设置Console将日志输出到文件中保存，同时在控制台输出
    console = Console(record=True)
    console.print("Hello, World!")

def save_jsonl():
    import json                           
    with open("./data/datasets/Flowchart2DotDatasets/dotV2/datasets/flowchart2dot_train.json", "r", encoding="utf-8") as f:
        base_graph_data = json.load(f)
    
    with open("./data/datasets/Flowchart2DotDatasets/dotV2/datasets/flowchart2dot_train.jsonl", "a", encoding="utf-8") as f:
        for item in base_graph_data:
            f.write(json.dumps(item)+"\n")

def merge():
    from conf.Tools import Tools
    tools = Tools()
    llm_output = tools.read_json("output/flowchart/graphviz/gpt-4o.json")
    ground_data = tools.read_json("data/FlowchartDatasets/FlowchartDatasets/test.json")
    for key, response in llm_output.items():
        # Extract text representation (mermaid, graphviz, or plantuml) from response
        ground_data[key]["llm"] = response
    tools.write_2_json(ground_data, "output/flowchart/graphviz/gpt-4o.json")

def tansform_dot2json():
    from Evaluation.flowchart_eval import transform_dot_2_json
    from conf.Tools import Tools
    tools = Tools()
    res = transform_dot_2_json("digraph G {\n    rankdir=TB;\n    node [shape=box style=rounded];\n\n    A [label=\"Upper extremity pain\" shape=box style=\"rounded,filled\" fillcolor=lightblue];\n    B [label=\"Where is the point of maximal pain?\" shape=box style=\"rounded,filled\" fillcolor=gold];\n    C [label=\"Forearm or elbow\"];\n    D [label=\"Wrist\"];\n    E [label=\"Hand\"];\n    F [label=\"Is the pain over the epicondyles?\"];\n    G [label=\"Lateral or medial epicondylitis\" shape=box style=\"rounded,filled\" fillcolor=plum];\n    H [label=\"Is there erythema and pain over the olecranon process?\"];\n    I [label=\"Olecranon bursitis\" shape=box style=\"rounded,filled\" fillcolor=plum];\n    J [label=\"Is the pain like 'hitting your funny bone'?\"];\n    K [label=\"Cubital tunnel syndrome\" shape=box style=\"rounded,filled\" fillcolor=plum];\n    L [label=\"Consider alternative diagnoses such as RSD/CRPS or referred pain\" shape=box style=\"rounded,filled\" fillcolor=orange];\n    M [label=\"Is there pain and loss of function of the first 2 or 3 digits of the hand?\"];\n    N [label=\"Carpal tunnel syndrome\" shape=box style=\"rounded,filled\" fillcolor=plum];\n    O [label=\"Did the pain start on the thumb side of the forearm and then spread?\"];\n    P [label=\"de Quervain tenosynovitis\" shape=box style=\"rounded,filled\" fillcolor=plum];\n    Q [label=\"Consider alternative diagnoses such as intersection syndrome\" shape=box style=\"rounded,filled\" fillcolor=orange];\n    R [label=\"Do the fingers change color with exposure to cold?\"];\n    S [label=\"Raynaud's disorder\" shape=box style=\"rounded,filled\" fillcolor=plum];\n    T [label=\"Is there clicking and pain when the digit is bent or straightened?\"];\n    U [label=\"Trigger finger\" shape=box style=\"rounded,filled\" fillcolor=plum];\n    V [label=\"Is the pain associated with repetitive motion of the arms held overhead?\"];\n    W [label=\"Thoracic outlet syndrome\" shape=box style=\"rounded,filled\" fillcolor=orange];\n    X [label=\"Consider osteoarthritis versus rheumatoid arthritis\" shape=box style=\"rounded,filled\" fillcolor=orange];\n\n    A -> B;\n    B -> C;\n    B -> D;\n    B -> E;\n    C -> F;\n    F -> G [label=\"Yes\"];\n    F -> H [label=\"No\"];\n    H -> I [label=\"Yes\"];\n    H -> J [label=\"No\"];\n    J -> K [label=\"Yes\"];\n    J -> L [label=\"No\"];\n    D -> M;\n    M -> N [label=\"Yes\"];\n    M -> O [label=\"No\"];\n    O -> P [label=\"Yes\"];\n    O -> Q [label=\"No\"];\n    E -> R;\n    R -> S [label=\"Yes\"];\n    R -> T [label=\"No\"];\n    T -> U [label=\"Yes\"];\n    T -> V [label=\"No\"];\n    V -> W [label=\"Yes\"];\n    V -> X [label=\"No\"];\n}")
    tools.write_2_json(res, "output/test/graphviz.json")

def trans_flowchart2dot():
    import glob
    from flowchart_construct.utils import json_2_dot
    from conf.Tools import Tools
    from tqdm import tqdm
    tools = Tools()
    dir_path = "data/FlowchartDatasets/FlowchartDatasets/Json"
    flowchart_files = glob.glob(f"{dir_path}/**/*.json", recursive=True)
    for flow_file in tqdm(flowchart_files):
        flowchart_json = tools.read_json(flow_file)
        data = {"nodes":[], "edges":[]}
        for node in flowchart_json["nodes"]:
            data["nodes"].append({
                "id": str(node["id"]),
                "label": node["Name"],
                "attributes": {
                    "label": node["Name"].replace('\n', '')
                }
            })
        
        for edge in flowchart_json["edges"]:
            data["edges"].append({
                "source": edge["sourceNode"],
                "target": edge["targetNode"],
                "attributes": {
                    "label": edge["label"]
                }
            })

        flowchart_dot = json_2_dot(None, data)
        tools.write_2_txt(flowchart_dot, flow_file.replace("Json", "Dot").replace(".json", ".dot"))

def write2jsonl():
    
    tools = Tools()
    data = tools.read_json("data/FlowchartDatasets/TrainDatasets/Image2DotV1/datasets/flowchart2dot_train.json")
    # 写jsonl文件
    with open("data/FlowchartDatasets/TrainDatasets/Image2DotV1/datasets/flowchart2dot_train.jsonl", "w", encoding="utf-8") as f:
        for item in data:
            item['image'] = item['image'].replace("data/FlowchartDatasets/TrainDatasets/Image2DotV1/Images//", "")
            f.write(json.dumps(item, ensure_ascii=False)+"\n")

def test_lmdeploy():
    from lmdeploy_test import pipeline, TurbomindEngineConfig
    from lmdeploy.vl import load_image

    model = 'work_dirs/internvl_chat_v2_5/internvl2_5_8b_dynamic_res_2nd_finetune_lora_unfreeze_llm_backbone_mlp_v2'
    image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
    pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192))
    response = pipe(('describe this image', image))
    print(response.text)

if __name__=="__main__":
    # tansform_dot2json()
    test_lmdeploy()
    # files = glob("data/datasets/Flowchart2DotDatasets/dotV2/Dot/**/*.dot", recursive=True)
    # print(len(files))