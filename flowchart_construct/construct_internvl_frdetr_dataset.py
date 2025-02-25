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
        id2imgpath[img["id"]] = {
            "p": f"{img_path}/{img['file_name']}",
            "size": [img["width"], img["height"]]
        }

    id2categorie = {}
    for cate in frdetr_train_annotation["categories"]:
        id2categorie[cate["id"]] = cate["name"]
    print(id2categorie)
    id2annots = {}
    for annot in frdetr_train_annotation["annotations"]:
        if annot["image_id"] not in id2annots.keys(): id2annots[annot["image_id"]] = list()
        annot["category_name"] = id2categorie[str(annot["category_id"])]
        annot["image_path"] = id2imgpath[annot["image_id"]]["p"]
        annot["size"] = id2imgpath[annot["image_id"]]["size"]
        id2annots[annot["image_id"]].append(annot)

    return id2annots

processed_data_path = "/root/LLM-based-graph-tool/data/datasets/frdetr_dataset/internvl_frdetr_tain_dataset.json"

def construct_finetune_datasets():
    processed_data = tools.read_json(processed_data_path)
    finetune_datasets = list()
    data_num=0
    for img_id, img_annots  in processed_data.items():
        if len(img_annots)==0:
            print(img_id)
            continue
        img_path = img_annots[0]['image_path']
        # print(img_annots[0])
        w,h = img_annots[0]["size"][0], img_annots[0]["size"][1]
        ######################
        img_name_set = set([obj["category_name"] for obj in img_annots])
        for n in img_name_set:
            conversations = list()
            conversations.append({
                "from": "human",
                "value": f"<image>\nPlease provide the bounding box coordinate of the region this sentence describes: <ref>{n}</ref>"
            })
            annot_str = f"<ref>{n}</ref><box>["
            
            for annot in img_annots:
                if annot['category_name']!=n: continue
                if annot["category_name"]=="line":
                    annot_str += f"[{int(annot['line'][0])},{int(annot['line'][1])},{int(annot['line'][0] + annot['line'][2])},{int(annot['line'][1] + annot['line'][3])}],"
                else:
                    annot_str += f"[{int(annot['bbox'][0])},{int(annot['bbox'][1])},{int(annot['bbox'][0] + annot['bbox'][2])},{int(annot['bbox'][1] + annot['bbox'][3])}],"
            annot_str = annot_str[:-1] + "]</box>"
            
            conversations.append({"from": "gpt", "value": annot_str})
            finetune_datasets.append({"id": data_num,
                "image": img_path,
                "width": w,
                "height": h,
                "conversations": copy.deepcopy(conversations)
            })
            data_num+=1

        #######################
        current_img_item = {
            "id": f"img_{data_num}",
            "conversations": list()
        }
        conversations = list()
        conversations.append({"from": "human", "value": f"<image>\nPlease detect and label all objects in the following image and mark their positions."})

        
        image = cv2.imread(img_path)

        shape_type_annots = {}
        for annot in img_annots:
            if annot["category_name"] not in shape_type_annots.keys(): shape_type_annots[annot["category_name"]] = list()
            if annot["category_name"]=="line":
                cv2.rectangle(image, (int(annot['line'][0]),int(annot['line'][1])), (int(annot['line'][0] + annot['line'][2]),int(annot['line'][1] + annot['line'][3])), (0, 0, 255))
                shape_type_annots[annot["category_name"]].append([int(annot['line'][0]),int(annot['line'][1]),int(annot['line'][0] + annot['line'][2]),int(annot['line'][1] + annot['line'][3])])
            else:
                cv2.rectangle(image, (int(annot['bbox'][0]),int(annot['bbox'][1])), (int(annot['bbox'][0] + annot['bbox'][2]),int(annot['bbox'][1] + annot['bbox'][3])), (0, 0, 255))
                shape_type_annots[annot["category_name"]].append([int(annot['bbox'][0]),int(annot['bbox'][1]),int(annot['bbox'][0] + annot['bbox'][2]),int(annot['bbox'][1] + annot['bbox'][3])])

        cv2.imwrite("/root/LLM-based-graph-tool/data/datasets/verify_dataset/frdetr/" + img_path.split("/")[-1], image)

        annot_str = "Sure, I will detect and label all objects in the image and mark their positions.\n\n```\n"
        for k in shape_type_annots.keys():
            annot_str += f"<ref>{k}</ref><box>{str(shape_type_annots[k])}</box>\n"
        annot_str += "```\n"
        conversations.append({"from": "gpt", "value": annot_str})
        
        finetune_datasets.append({"id": data_num,
            "image": img_path,
            "width": w,
            "height": h,
            "conversations": copy.deepcopy(conversations)
        })
        data_num+=1

        # finetune_datasets.append(current_img_item)
    print(f"finetune_datasets: {len(finetune_datasets)}")
    tools.write_json(finetune_datasets, "/root/LLM-based-graph-tool/data/datasets/internvl_frdetr/internvl_frdetr_tain_dataset_v2.json")



def main():
    annotation_path = f"{FRDETR_PATH}/annotations/frdetr_train2017.json"
    save_dataset = load_frdetr_dataset(annotation_path, FRDETR_TRAIN_PATH)
    tools.write_json(save_dataset, "/root/LLM-based-graph-tool/data/datasets/frdetr_dataset/internvl_frdetr_tain_dataset.json")
    print(f"save_dataset:{len(save_dataset)}")


if __name__== "__main__":
    # main()
    construct_finetune_datasets()