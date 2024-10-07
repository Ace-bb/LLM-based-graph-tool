from Eval.flowchart_eval import generate_llm_output, eval_model_output, run_gpt
from lmdeploy import pipeline
from conf.Tools import Tools
def main():
    # pipe = pipeline('/root/LLM-based-graph-tool/models/InternVL2-8B-flow2json_v1') #'/root/models/InternVL2-8B'
    # pipe = pipeline('/root/models/InternVL2-8B') 
    # generate_llm_output("data/datasets/Flowchart2DotDatasets/dotV1/datasets/flowchart2dot_eval.json", pipe, "data/evaluation/Dot/GPT-4o-dot-evaluation.json")
    tools = Tools()
    inferenced_data = tools.read_json("data/evaluation/Dot/GPT-4o-dot-evaluation.json")
    print(len(inferenced_data))
    eval_model_output(inferenced_data)

def trans2json():
    from Eval.flowchart_eval import transform_dot_2_json
    tools = Tools()
    data = tools.read_json("/root/LLM-based-graph-tool/data/evaluation/Dot/GPT-4o-dot-evaluation-transed.json")
    # for item in data:
    #     item['llm_json_output'] = transform_dot_2_json(item['output'])
    #     item["golden_json_output"] = transform_dot_2_json(item["conversations"][-1]['value'])
    
    tools.write_2_json(data[0:10], "/root/LLM-based-graph-tool/data/evaluation/Dot/GPT-4o-dot-evaluation-transed0-10.json")
if __name__=="__main__":
    # from Eval.flowchart_eval import transform_dot_2_json
    main()