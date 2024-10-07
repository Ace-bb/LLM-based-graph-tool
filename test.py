from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

import os
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
test()