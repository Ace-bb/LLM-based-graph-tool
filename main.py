import argparse
import json
import logging
import os

from tqdm import tqdm

from conf.settings import *
from conf.logger import setup_logger
# from models import ModelWrapper
from LLMs.llm import LLM
from prompts.prompts import load_textualizer_prompt
from prompts.prompt_utils import load_messages
from utils.utils import extract_representation, encode_image


def main():
    parser = argparse.ArgumentParser(
        description="Run the Vision model_name program. (Convert flowchart to text Representation)"
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
        default="Qwen2-VL-7B",
        help="The VLM to generate the text represenation.",
    )
    parser.add_argument(
        "--output_type",
        type=str,
        default="mermaid",
        help="Text representation output format (mermaid, graphviz or plantuml)",
    )
    args = parser.parse_args()
    dataset = args.dataset
    model_name = args.model_name
    output_type = args.output_type
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
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    logger.info(f"Logs saved to {os.path.abspath(log_file)}")

    # model = ModelWrapper(model_name)
    model = LLM(config['api_key'], config['base_url'], model_name)
    
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
        prompt = load_textualizer_prompt(output_type)
        image = encode_image(image_path, model_name) if image_path else None
        messages = load_messages(model_name, prompt, image)
        run_paras.append((key, messages))

    def run(k, msgs):
        return k, model.run(msgs)
    run_res = tools.multi_thread_run(10, run, run_paras, "Vision model_name")
    
        # response = model.run(messages)
    for key, response in run_res:
        # Extract text representation (mermaid, graphviz, or plantuml) from response
        representation = extract_representation(response)
        results[key] = representation

    output_dir = os.path.join(config["file_paths"]["output"], dataset, output_type)
    output_file = os.path.join(output_dir, f"{model_name}.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "w") as file:
        json.dump(results, file, indent=4)
    logger.info(f"Results saved to {os.path.abspath(output_file)}")


if __name__ == "__main__":
    main()
    
# python main.py --dataset flowchart --model_name gpt-4o --output_type graphviz
