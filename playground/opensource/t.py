import json
with open("/root/LLM-based-graph-tool/data/datasets/internvl_baseGraphV1/dataset/base_graph_data_v2.json", "r", encoding="utf-8") as f:
    base_graph_data = json.load(f)

with open("./base_graph_data.jsonl", "a", encoding="utf-8") as f:
    for item in base_graph_data:
        f.write(json.dumps(item)+"\n")

# with open("/root/LLM-based-graph-tool/data/datasets/internvl_baseGraphV1/dataset/base_graph_data_v2.json", "r", encoding="utf-8") as f:
#     internvl_frdetr_tain_dataset = json.load(f)

# with open("./base_graph_data.jsonl", "a", encoding="utf-8") as f:
#     for item in internvl_frdetr_tain_dataset:
#         f.write(json.dumps(item)+"\n")