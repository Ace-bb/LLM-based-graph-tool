import json, cv2
import copy

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
    img_obj_mapping = {
        "arrow": "箭头",
        "rec": "矩形",
        "diamond": "菱形",
        "roundrec": "圆角矩形",
        "hex": "六边形",
        "ellipse": "椭圆",
        "circle": "圆形",
        "parallel": "平行四边形",
        "line": "线"
    }
    for img_id, img_annots  in processed_data.items():
        if len(img_annots)==0:
            print(img_id)
            continue
        current_img_item = {
            "id": f"img_{img_id}",
            "conversations": list()
        }
        conversations = list()
        conversations.append({
            "from": "user",
            "value": f"Picture {img_id}: <img>{img_annots[0]['image_path']}</img>\n图中都包含有哪些基本图形？."
        })
        img_name_zh = set([img_obj_mapping[obj['category_name']] for obj in img_annots])
        conversations.append({
            "from": "assistant",
            "value": f"图中共包含有下面这几种：{'、'.join(list(img_name_zh))}。"
        })
        ######################
        img_name_set = set([obj["category_name"] for obj in img_annots])
        for n in img_name_set:
            conversations.append({
                "from": "user",
                "value": f"框出图中的{img_obj_mapping[n]}"
            })
            annot_str = ""
            for annot in img_annots:
                if annot['category_name']!=n: continue
                if annot["category_name"]=="line":
                    annot_str += f"<ref>{img_obj_mapping[n]}</ref><box>({int(annot['line'][0])},{int(annot['line'][1])}),({int(annot['line'][0] + annot['line'][2])},{int(annot['line'][1] + annot['line'][3])})</box>\n"
                else:
                    annot_str += f"<ref>{img_obj_mapping[n]}</ref><box>({int(annot['bbox'][0])},{int(annot['bbox'][1])}),({int(annot['bbox'][0] + annot['bbox'][2])},{int(annot['bbox'][1] + annot['bbox'][3])})</box>\n"
                    # <ref>{img_obj_mapping[n]}</ref><box>({int(annot['bbox'][0])},{int(annot['bbox'][1])}),({int(annot['bbox'][0] + annot['bbox'][2])},{int(annot['bbox'][1] + annot['bbox'][3])})</box>
            
            conversations.append({"from": "assistant", "value": annot_str})

        #######################
        conversations.append({"from": "user", "value": f"框出图中的全部基础图形"})
        annot_str = ""
        for annot in img_annots:
            if annot["category_name"]=="line":
                annot_str += f"<ref>{annot['category_name']}</ref><box>({int(annot['line'][0])},{int(annot['line'][1])}),({int(annot['line'][0] + annot['line'][2])},{int(annot['line'][1] + annot['line'][3])})</box>\n"
            else:
                annot_str += f"<ref>{annot['category_name']}</ref><box>({int(annot['bbox'][0])},{int(annot['bbox'][1])}),({int(annot['bbox'][0] + annot['bbox'][2])},{int(annot['bbox'][1] + annot['bbox'][3])})</box>\n"
        conversations.append({"from": "assistant", "value": annot_str})
        
        current_img_item["conversations"] = conversations

        finetune_datasets.append(current_img_item)
    tools.write_json(finetune_datasets, "/root/LLM-based-graph-tool/data/datasets/v1_frdetr/v2_frdetr_train.json")

def construct_finetune_datasets_v2():
    processed_data = tools.read_json(processed_data_path)
    finetune_datasets = list()
    img_obj_mapping = {
        "arrow": "箭头",
        "rec": "矩形",
        "diamond": "菱形",
        "roundrec": "圆角矩形",
        "hex": "六边形",
        "ellipse": "椭圆",
        "circle": "圆形",
        "parallel": "平行四边形",
        "line": "线"
    }
    data_num=0
    for img_id, img_annots  in processed_data.items():
        if len(img_annots)==0:
            print(img_id)
            continue
        current_img_item = {
            "id": f"img_{data_num}",
            "conversations": list()
        }
        img_path = img_annots[0]['image_path']
        current_img_item["conversations"].append({
            "from": "user",
            "value": f"Picture {data_num}: <img>{img_path}</img>\n图中都包含有哪些基本图形？."
        })
        img_name_zh = set([img_obj_mapping[obj['category_name']] for obj in img_annots])
        current_img_item["conversations"].append({
            "from": "assistant",
            "value": f"图中共包含有下面这几种：{'、'.join(list(img_name_zh))}。"
        })
        finetune_datasets.append(copy.deepcopy(current_img_item))
        data_num+=1
        ######################
        img_name_set = set([obj["category_name"] for obj in img_annots])
        for n in img_name_set:
            current_img_item = {
                "id": f"img_{data_num}",
                "conversations": list()
            }
            current_img_item["conversations"].append({
                "from": "user",
                "value": f"Picture {data_num}: <img>{img_path}</img>\n\n框出图中的{img_obj_mapping[n]}"
            })
            annot_str = ""
            for annot in img_annots:
                if annot['category_name']!=n: continue
                if annot["category_name"]=="line":
                    annot_str += f"<ref>{img_obj_mapping[n]}</ref><box>({int(annot['line'][0])},{int(annot['line'][1])}),({int(annot['line'][0] + annot['line'][2])},{int(annot['line'][1] + annot['line'][3])})</box>\n"
                else:
                    annot_str += f"<ref>{img_obj_mapping[n]}</ref><box>({int(annot['bbox'][0])},{int(annot['bbox'][1])}),({int(annot['bbox'][0] + annot['bbox'][2])},{int(annot['bbox'][1] + annot['bbox'][3])})</box>\n"
                    # <ref>{img_obj_mapping[n]}</ref><box>({int(annot['bbox'][0])},{int(annot['bbox'][1])}),({int(annot['bbox'][0] + annot['bbox'][2])},{int(annot['bbox'][1] + annot['bbox'][3])})</box>
            
            current_img_item["conversations"].append({"from": "assistant", "value": annot_str})
            finetune_datasets.append(copy.deepcopy(current_img_item))
            data_num+=1

        #######################
        current_img_item = {
            "id": f"img_{data_num}",
            "conversations": list()
        }
        current_img_item["conversations"].append({"from": "user", "value": f"Picture {data_num}: <img>{img_path}</img>\n\n框出图中的全部基础图形"})
        annot_str = ""
        for annot in img_annots:
            if annot["category_name"]=="line":
                annot_str += f"<ref>{annot['category_name']}</ref><box>({int(annot['line'][0])},{int(annot['line'][1])}),({int(annot['line'][0] + annot['line'][2])},{int(annot['line'][1] + annot['line'][3])})</box>\n"
            else:
                annot_str += f"<ref>{annot['category_name']}</ref><box>({int(annot['bbox'][0])},{int(annot['bbox'][1])}),({int(annot['bbox'][0] + annot['bbox'][2])},{int(annot['bbox'][1] + annot['bbox'][3])})</box>\n"
        current_img_item["conversations"].append({"from": "assistant", "value": annot_str})
        
        finetune_datasets.append(copy.deepcopy(current_img_item))
        data_num+=1

        # finetune_datasets.append(current_img_item)
    print(f"finetune_datasets: {len(finetune_datasets)}")
    tools.write_json(finetune_datasets, "/root/LLM-based-graph-tool/data/datasets/v1_frdetr/v3_frdetr_train.json")



def main():
    annotation_path = f"{FRDETR_PATH}/annotations/frdetr_train2017.json"
    tools.write_json(load_frdetr_dataset(annotation_path, FRDETR_TRAIN_PATH), "/root/LLM-based-graph-tool/data/datasets/frdetr_dataset/tain_dataset.json")


if __name__== "__main__":
    construct_finetune_datasets_v2()