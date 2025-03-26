"""
    构建流程图数据集，数据来源有：
    1. 从医书中的流程图转换
    2. 从开源的流程图数据集中转换
        - FlowLearn
        - FlowVQA
        - FRDETR
    3. 自动化构造
"""
import os, json
from tqdm import tqdm
from glob import glob
from json2dot_mermaid import json_2_dot, json_2_mermaid

def construct_flowcharts():
    flowchart_img_path = "../data/FlowchartDatasets/FlowchartDatasets/images"
    flowchart_data_path = "../data/FlowchartDatasets/FlowchartDatasets/Json"
    flowchart_datasets = {}
    flowchart_images = glob(f"{flowchart_img_path}/**/*.png")
    for img_file in tqdm(flowchart_images):
        img_name = img_file.replace(flowchart_img_path+'/', '')
        data_file = os.path.join(flowchart_data_path, img_name.replace(".png", ".json"))
        with open(data_file, 'r', encoding="utf-8") as f:
            data = json.load(f)
        flowchart_datasets[img_name] = {
            "json": data,
            # "mermaid": json_2_mermaid(img_name, data),
            "dot": json_2_dot(img_name, data),
        }
    with open("../data/FlowchartDatasets/FlowchartDatasets/test.json", 'w', encoding="utf-8") as f:
        json.dump(flowchart_datasets, f, ensure_ascii=False)

if __name__=="__main__":
    construct_flowcharts()