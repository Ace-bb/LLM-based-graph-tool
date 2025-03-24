from PIL import Image
import json, random

"""InternVL系列模型的训练数据格式"""
InternVL_data_format={
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

img2dot_introductions = [
    
]
read_json = lambda path: json.load(open(path, 'r', encoding='utf-8'))

def create_internvl_data(_id, image_path, dot_content):
    # 获取图片的宽高
    image = Image.open(image_path)
    width, height = image.size
    image.close()
    instructions = read_json("data/DOT_Questions.json")
    return {
        "id": _id,
        "image": image_path,
        "width": width,
        "height": height,
        "conversations": [
            {
                "from": "human",
                "value": "<image>\n" + random.choice(instructions) # 从instruction种随机选择一条
            },
            {
                "from": "gpt",
                "value": f"```dot\n{dot_content}\n```" if random.random() > 0.5 else dot_content
            }
        ]
    }