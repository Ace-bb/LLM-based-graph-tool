import argparse
import json
import logging
import os

from tqdm import tqdm

from conf.settings import *
from conf.logger import setup_logger
# from models import ModelWrapper
from LLMs.llm import LLM
from Inference.Inference import inference
from flowchart_construct.construct_train_datasets import flowchart_operate

def train(args):
    ...

def evaluate(args):
    ...
    
def main():
    parser = argparse.ArgumentParser(
        add_help=False,
        description="Run the Vision model_name program. (Convert flowchart to text Representation)"
    )
    subparsers = parser.add_subparsers(help='sub parser', dest='module')
    
    """推理测试生成流程图识别结果
    python main.py infer --dataset flowchart --model_name gpt-4o-mini --output_type graphviz
    
    python main.py infer --dataset flowchart --model_name internvl2.5-local --output_type graphviz
    python main.py infer --dataset flowchart --model_name internvl2.5-8B --output_type graphviz
    """
    inference_parser = subparsers.add_parser(name='infer', help='在各个模型上进行推理测试生成流程图识别结果', parents=[parser])
    inference_parser.add_argument("--dataset",type=str,default="flowvqa",help="Dataset to use (flowvqa, flowvqa_bottom_top or flowlearn).")
    inference_parser.add_argument("--model_name",type=str, default="Qwen2-VL-7B", help="The VLM to generate the text represenation.")
    inference_parser.add_argument("--output_type", type=str, default="mermaid", help="Text representation output format (mermaid, graphviz or plantuml)")
    inference_parser.add_argument("--engine", type=str, default="api", help="调用接口推理，还是本地部署推理)")
    inference_parser.set_defaults(func = inference)
    
    """构造训练数据集
    python main.py flowchart --input_file data/datasets/Flowchart2DotDatasets/dotV2/Dot --output_file Image2DotV1 --op_type construct
    python main.py flowchart --input_file data/datasets/GenerateDiseaseDots/Dot --output_file data/datasets/FlowchartTrainDatasets/V1 --op_type construct
    python main.py flowchart --input_file data/FlowchartDatasets/TrainDatasets/Image2DotV1 --output_file data/FlowchartDatasets/TrainDatasets/Image2DotV2 --op_type filter
    python main.py flowchart --input_file data/datasets/DXDiseases --output_file data/datasets/GenerateDiseaseDots --op_type generate
    python main.py flowchart --input_file data/datasets/DifferentialDiagnosisEnglish --output_file data/datasets/DifferentialDiagnosisEnglish --op_type json2dot
    """
    flowchart_parser = subparsers.add_parser(name='flowchart', help='将JSON格式的流程图数据转换成Dot格式和mermaid格式，并构造成数据集', parents=[parser])
    flowchart_parser.add_argument("--input_file", type=str, help="The input file path of the flowchart data in JSON format.")
    flowchart_parser.add_argument("--output_file", type=str, help="The output file path of the flowchart data in Dot and mermaid format.")
    flowchart_parser.add_argument("--op_type", type=str, default='construct', choices=["construct", "filter", "generate", "json2dot"], help="操作类型，构造数据集还是过滤数据集")
    flowchart_parser.set_defaults(func = flowchart_operate)
    
    """模型训练
    """
    train_parser = subparsers.add_parser(name='train', help='训练模型', parents=[parser])
    train_parser.add_argument("--dataset", type=str, help="Dataset to use (flowvqa, flowvqa_bottom_top or flowlearn).")
    train_parser.add_argument("--model_name", type=str, help="The VLM to generate the text represenation.")
    train_parser.add_argument("--output_type", type=str, help="Text representation output format (mermaid, graphviz or plantuml)")
    train_parser.set_defaults(func = train)
    
    """模型评估
    """
    evaluate_parser = subparsers.add_parser(name='evaluate', help='评估模型', parents=[parser])
    evaluate_parser.add_argument("--dataset", type=str, help="Dataset to use (flowvqa, flowvqa_bottom_top or flowlearn).")
    evaluate_parser.add_argument("--model_name", type=str, help="The VLM to generate the text represenation.")
    evaluate_parser.add_argument("--output_type", type=str, help="Text representation output format (mermaid, graphviz or plantuml)")
    evaluate_parser.set_defaults(func = evaluate)
    
    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args) 
    


if __name__ == "__main__":
    main()
