import os, json, cv2
from conf.Tools import Tools



DxDiseaseTreeDots = "data/datasets/DxDiseaseTreeDots"
DxDiseaseTreeImg = "data/datasets/DotImages"

Image_save_path = "data/datasets/Flowchart2DotDatasets/v3/images"
# DxDiseaseTreeImg = Image_save_path
Anont_save_path = "data/datasets/Flowchart2DotDatasets/v3/dataset"

{
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



if __name__=="__main__":
    main()