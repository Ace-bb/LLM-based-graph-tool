from peft import AutoPeftModelForCausalLM  # 确保导入所需的模块
from transformers import (
     AutoTokenizer
)
 
 
path_to_adapter = "/root/LLM-based-graph-tool/output_qwen/fcr_v1"
# 从预训练模型中加载自定义适配器模型
model = AutoPeftModelForCausalLM.from_pretrained(
    path_to_adapter,  # 适配器的路径
    device_map="auto",  # 自动映射设备
    trust_remote_code=True  # 信任远程代码
).eval()  # 设置为评估模式
new_model_directory = "/root/LLM-based-graph-tool/output_qwen/merged/v1"
tokenizer = AutoTokenizer.from_pretrained(
    path_to_adapter, trust_remote_code=True,
)
tokenizer.save_pretrained(new_model_directory)
 
# 合并并卸载模型
merged_model = model.merge_and_unload()
# 保存合并后的模型
merged_model.save_pretrained(new_model_directory, max_shard_size="2048MB", safe_serialization=True)