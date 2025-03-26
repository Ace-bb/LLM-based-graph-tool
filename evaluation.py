import argparse
import json
import logging
import os
from datetime import datetime

from tqdm import tqdm

from conf.settings import *
from prompts.prompts import load_evaluation_prompt
from conf.logger import setup_logger
# from utils.utils import majority_vote
from LLMs.llm import LLM
from Evaluation.flowchart_eval import eval_llm_result

def main():
    parser = argparse.ArgumentParser(
        description="Run the Question Answering Evaluation program."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="flowvqa",
        help="Dataset to use (flowvqa, flowvqa_bottom_top or flowlearn).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-4o",
        help="The LLM used as the evaluator.",
    )
    parser.add_argument(
        "--output_type",
        type=str,
        default="mermaid",
        help="Text representation output format (mermaid, graphviz or plantuml)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="output/flowvqa/textflow/mermaid_reasoner_Llama-3.1-8B_textualizer_Qwen2-VL-7B.json",
        help="Data path of the experiment result to evaluate.",
    )
    args = parser.parse_args()
    model_name = args.model_name
    data_path = args.data_path
    eval_llm_result(data_path)
    return
    dataset = args.dataset
    output_type = args.output_type
    exp_dir = os.path.dirname(data_path)
    exp_name = os.path.splitext(os.path.basename(data_path))[0]
    config = load_config(model_name)

    # Setup logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(
        config["logging"]["log_dir"],
        dataset,
        f"evaluation_{exp_name}_{timestamp}.log",
    )
    logger = setup_logger(log_file, config["logging"]["log_level"].upper())
    logger = logging.getLogger(__name__)
    
    
    logger.info("Starting the Question Answering Evaluation program...")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    logger.info(f"Logs saved to {os.path.abspath(log_file)}")

    # model = ModelWrapper(model_name)
    model = LLM(os.environ['OPENAI_BASE_KEY'], os.environ['OPENAI_BASE_URL'], model_name)
    ground_data_path = os.path.join(config["file_paths"][dataset], "test.json")
    llm_data_path = data_path
    ground_data = tools.read_json(ground_data_path)
    llm_data = tools.read_json(llm_data_path)
    for k in ground_data.keys():
        ground_data[k]['llm'] = llm_data[k]
        
    for key, sample in tqdm(ground_data.items()):
        ground_json = sample["json"]
        ...

    with open(data_path, "w") as file:
        json.dump(data, file, indent=4)
    logger.info(f"Results saved to {os.path.abspath(data_path)}")

    # Calculate accuracy
    correct_count = 0
    total_count = len(data)

    for sample in data.values():
        if sample["final_decision"] == "Correct":
            correct_count += 1

    accuracy = correct_count / total_count if total_count > 0 else 0
    logger.info(f"Accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()

# python evaluation.py --model_name gpt-4o --data_path output/flowchart/graphviz/gpt-4o.json
# python evaluation.py --model_name gpt-4o --data_path output/flowchart/qwen-plus/graphviz.json
# python evaluation.py --model_name gpt-4o --data_path output/flowchart/internvl2.5-8B/graphviz.json