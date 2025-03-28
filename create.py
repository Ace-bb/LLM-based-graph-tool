from conf.Tools import Tools

tools = Tools()
jsonl_data = tools.read_jsonl("data/datasets/FlowchartTrainDatasets/V3/Datasets/flowchart2dot_test.jsonl")

test_data = {}
for data in jsonl_data:
    image_key = data['image']
    dot_content = tools.read_file(f"data/datasets/FlowchartTrainDatasets/V3/Dot/{image_key.replace('.png', '')}.dot")
    data['dot'] = dot_content
    test_data[image_key] = data
tools.write_2_json(test_data, "data/datasets/FlowchartTrainDatasets/V3/test.json")