from conf.settings import *
import logging
from tqdm import tqdm
from LLMs.llm import LLM
from conf.logger import setup_logger
from prompts.prompts import load_textualizer_prompt
from prompts.prompt_utils import load_messages
from utils.utils import extract_representation, encode_image


def inference(args):
    print(args)
    dataset = args.dataset
    model_name = args.model_name
    output_type = args.output_type # 'qwen-plus', 
    engine = args.engine
    run_llm_inference(dataset, model_name, output_type)
    # if engine == 'api':
    #     for model_name in ['qwen-vl-plus-latest', 'qwen2.5-vl-72b-instruct', 'qwen2.5-vl-7b-instruct', 'qwen2.5-vl-3b-instruct', 'llama3.2-90b-vision-instruct', 'llama3.2-11b-vision']:
    #         run_llm_inference(dataset, model_name, output_type)
    # elif engine == "lmdeploy":
    #     run_llm_deploy_inference(dataset, model_name, output_type)

def run_llm_deploy_inference(dataset, model_name, output_type):
    config = load_config(model_name)
    
    # Setup logger
        
def run_llm_inference(dataset, model_name, output_type):
    print(dataset, model_name, output_type)
    config = load_config(model_name)
    
    # Setup logger
    log_file = os.path.join(
        config["logging"]["log_dir"],
        dataset,
        f"{output_type}_textulizer_{model_name}_{timestamp}.log",
    )
    logger = setup_logger(log_file, config["logging"]["log_level"].upper())
    logger = logging.getLogger(__name__)
    logger.info("Starting the Vision model_name program...")
    # for arg, value in vars(args).items():
    #     logger.info(f"{arg}: {value}")
    logger.info(f"Logs saved to {os.path.abspath(log_file)}")

    # model = ModelWrapper(model_name)
    print(config['api_key'], config['base_url'], model_name)  
    model = LLM(config['api_key'], config['base_url'], model_name)
    # model = LLM("1234567890", "http://0.0.0.0:23333/v1", model_name)
    
    data_path = os.path.join(config["file_paths"][dataset], "test.json")
    with open(data_path, "r") as file:
        data = json.load(file)
    keys = list(data.keys())

    results = {}
    image_extension = "jpeg" if dataset == "flowlearn" else "png"
    run_paras = []
    for key in tqdm(keys):
        image_path = os.path.join(
            config["file_paths"][dataset], "images", f"{key}"
        )
        if "DifferentialDiagnosisEnglish" not in image_path: continue
        prompt = "请分析以下流程图，识别其中所有的节点和边，并提取每个节点的类型和内容，同时标明节点间的连接关系。请用Graphviz dot语言将该流程图转化为描述性图形。" # load_textualizer_prompt(output_type)
        image = encode_image(image_path, model_name) if image_path else None
        print(image)
        messages = load_messages(model_name, prompt, image)
        run_paras.append((key, messages))

    def run(k, msgs):
        return k, model.run(msgs)
    run_res = tools.multi_thread_run(10, run, run_paras, "Vision model_name")
    
        # response = model.run(messages)
    for key, response in run_res:
        # Extract text representation (mermaid, graphviz, or plantuml) from response
        representation = extract_representation(response)
        item = data[key]
        item["llm"] = representation
        results[key] = item

    output_dir = os.path.join(config["file_paths"]["output"], dataset, model_name)
    output_file = os.path.join(output_dir, f"{output_type}.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "w", encoding='utf-8') as file:
        json.dump(results, file, indent=4, ensure_ascii=False)
    logger.info(f"Results saved to {os.path.abspath(output_file)}")