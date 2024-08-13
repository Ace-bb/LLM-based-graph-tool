import json, cv2


class Tools:
    
    def read_json(self, file_path):
        with open(file_path, 'r', encoding="utf-8") as f:
            return json.load(f)
    
    def write_json(self, data, save_path):
        with open(save_path, 'w', encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

# cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,0,255), 2)
tools = Tools()
FRDETR_PATH = "/root/LLM-based-graph-tool/data/datasets/frdetr_dataset"
FRDETR_TRAIN_PATH = "/root/LLM-based-graph-tool/data/datasets/frdetr_dataset/train"
FRDETR_VAL_PATH = "/root/LLM-based-graph-tool/data/datasets/frdetr_dataset/val"

def load_frdetr_dataset(annotation_path, img_path):
    frdetr_train_annotation = tools.read_json(annotation_path)
    id2imgpath = {}
    for img in frdetr_train_annotation["images"]:
        id2imgpath[img["id"]] = f"{img_path}/{img['file_name']}"
    id2categorie = {}
    for cate in frdetr_train_annotation["categories"]:
        id2categorie[cate["id"]] = cate["name"]
    print(id2categorie)
    id2annots = {}
    for annot in frdetr_train_annotation["annotations"]:
        if annot["id"] not in id2annots.keys(): id2annots[annot["id"]] = list()
        annot["category_name"] = id2categorie[str(annot["category_id"])]
        annot["image_path"] = id2imgpath[annot["image_id"]]
        id2annots[annot["image_id"]].append(annot)

    return id2annots

processed_data_path = "/root/LLM-based-graph-tool/data/datasets/frdetr_dataset/tain_dataset.json"
def construct_finetune_datasets(file_path):
    processed_data = tools.read_json(processed_data_path)
    finetune_datasets = list()
    for img_id, img_annots  in enumerate(processed_data):
        finetune_datasets.append({
            "id": f"img_{img_id}",
            "conversations": list()
        })
        finetune_datasets["conversations"].append({
            "from": "user",
            "value": f"Picture {img_id}: <img>{img_annots[0]['image_path']}</img>\n这个流程图中都包含有哪些信息？"
        })
        finetune_datasets["conversations"].append({
            "from": "assistant",
            "value": ""
        })
        annot_str = ""
        for annot in img_annots:
            annot_str += f"<ref>{annot['category_name']}</ref><box>({annot['bbox'][0]},{annot['bbox'][1]}),({annot['bbox'][0] + annot['bbox'][2]},{annot['bbox'][1] + annot['bbox'][3]})</box>"
        finetune_datasets["conversations"].append({
            "from": "user",
            "value": "框出这张图流程图中的全部基础图形"
        })
        finetune_datasets["conversations"].append({
            "from": "assistant",
            "value": annot_str
        })
        flowchart_json_object = {"nodes":[],"edges":[]}
        finetune_datasets["conversations"].append({
            "from": "user",
            "value": "将这张流程图转换成Json格式"
        })
        finetune_datasets["conversations"].append({
            "from": "assistant",
            "value": annot_str
        })

def construct_finetune_datasets_v1():
    processed_data = tools.read_json(processed_data_path)
    finetune_datasets = list()
    for img_id, img_annots  in processed_data.items():
        if len(img_annots)==0:
            print(img_id)
            continue
        current_img_item = {
            "id": f"img_{img_id}",
            "conversations": list()
        }
        current_img_item["conversations"].append({
            "from": "user",
            "value": f"Picture {img_id}: <img>{img_annots[0]['image_path']}</img>\n框出这张图流程图中的全部基础图形."
        })
        annot_str = ""
        for annot in img_annots:
            if annot["category_name"]=="line":
                annot_str += f"<ref>{annot['category_name']}</ref><box>({int(annot['line'][0])},{int(annot['line'][1])}),({int(annot['line'][0] + annot['line'][2])},{int(annot['line'][1] + annot['line'][3])})</box>"
            else:
                annot_str += f"<ref>{annot['category_name']}</ref><box>({int(annot['bbox'][0])},{int(annot['bbox'][1])}),({int(annot['bbox'][0] + annot['bbox'][2])},{int(annot['bbox'][1] + annot['bbox'][3])})</box>"
        
        current_img_item["conversations"].append({
            "from": "assistant",
            "value": annot_str
        })
        finetune_datasets.append(current_img_item)
    tools.write_json(finetune_datasets, "/root/LLM-based-graph-tool/data/datasets/v1_frdetr/v1_frdetr_train.json")


def main():
    annotation_path = f"{FRDETR_PATH}/annotations/frdetr_train2017.json"
    tools.write_json(load_frdetr_dataset(annotation_path, FRDETR_TRAIN_PATH), "/root/LLM-based-graph-tool/data/datasets/frdetr_dataset/tain_dataset.json")


if __name__== "__main__":
    construct_finetune_datasets_v1()