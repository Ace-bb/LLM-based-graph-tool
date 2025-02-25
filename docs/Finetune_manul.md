## 微调

我们提供了`finetune.py`这个脚本供用户实现在自己的数据上进行微调的功能，以接入下游任务。此外，我们还提供了shell脚本减少用户的工作量。这个脚本支持 [DeepSpeed](https://github.com/microsoft/DeepSpeed) 和 [FSDP](https://engineering.fb.com/2021/07/15/open-source/fsdp/) 。我们提供的shell脚本使用了DeepSpeed，因此建议您确保已经安装DeepSpeed。

首先，你需要准备你的训练数据。你需要将所有样本放到一个列表中并存入json文件中。每个样本对应一个字典，包含id和conversation，其中后者为一个列表。示例如下所示：
```json
[
  {
    "id": "identity_0",
    "conversations": [
      {
        "from": "user",
        "value": "你好"
      },
      {
        "from": "assistant",
        "value": "我是Qwen-VL,一个支持视觉输入的大模型。"
      }
    ]
  },
  {
    "id": "identity_1",
    "conversations": [
      {
        "from": "user",
        "value": "Picture 1: <img>https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg</img>\n图中的狗是什么品种？"
      },
      {
        "from": "assistant",
        "value": "图中是一只拉布拉多犬。"
      },
      {
        "from": "user",
        "value": "框出图中的格子衬衫"
      },
      {
        "from": "assistant",
        "value": "<ref>格子衬衫</ref><box>(588,499),(725,789)</box>"
      }
    ] 
  },
  { 
    "id": "identity_2",
    "conversations": [
      {
        "from": "user",
        "value": "Picture 1: <img>assets/mm_tutorial/Chongqing.jpeg</img>\nPicture 2: <img>assets/mm_tutorial/Beijing.jpeg</img>\n图中都是哪"
      },
      {
        "from": "assistant",
        "value": "第一张图片是重庆的城市天际线，第二张图片是北京的天际线。"
      }
    ]
  }
]
```
为针对多样的VL任务，我们增加了一下的特殊tokens： `<img> </img> <ref> </ref> <box> </box>`.

对于带图像输入的内容可表示为 `Picture id: <img>img_path</img>\n{your prompt}`，其中`id`表示对话中的第几张图片。"img_path"可以是本地的图片或网络地址。 

对话中的检测框可以表示为`<box>(x1,y1),(x2,y2)</box>`，其中 `(x1, y1)` 和`(x2, y2)`分别对应左上角和右下角的坐标，并且被归一化到`[0, 1000)`的范围内. 检测框对应的文本描述也可以通过`<ref>text_caption</ref>`表示。


准备好数据后，你可以使用我们提供的shell脚本实现微调。注意，你需要在脚本中指定你的数据的路径。

微调脚本能够帮你实现：
- 全参数微调
- LoRA
- Q-LoRA

### 全参数微调
默认下全参数微调在训练过程中更新LLM所有参数。我们的实验中，在微调阶段不更新ViT的参数会取得更好的表现。你可以运行这个脚本开始训练：

```bash
# 分布式训练。由于显存限制将导致单卡训练失败，我们不提供单卡训练脚本。
sh finetune/finetune_ds.sh
```

尤其注意，你需要在脚本中指定正确的模型名称或路径、数据路径、以及模型输出的文件夹路径。如果你想修改deepspeed配置，可以删除掉`--deepspeed`这个输入或者自行根据需求修改DeepSpeed配置json文件。此外，我们支持混合精度训练，因此你可以设置`--bf16 True`或者`--fp16 True`。经验上，如果你的机器支持bf16，我们建议使用bf16，这样可以和我们的预训练和对齐训练保持一致，这也是为什么我们把默认配置设为它的原因。

### LoRA
运行LoRA的方法类似全参数微调。但在开始前，请确保已经安装`peft`代码库。另外，记住要设置正确的模型、数据和输出路径。我们建议你为模型路径使用绝对路径。这是因为LoRA仅存储adapter部分参数，而adapter配置json文件记录了预训练模型的路径，用于读取预训练模型权重。同样，你可以设置bf16或者fp16。

```bash
# 单卡训练
sh finetune/finetune_lora_single_gpu.sh
# 分布式训练
sh finetune/finetune_lora_ds.sh
```

与全参数微调不同，LoRA ([论文](https://arxiv.org/abs/2106.09685)) 只更新adapter层的参数而无需更新原有语言模型的参数。这种方法允许用户用更低的显存开销来训练模型，也意味着更小的计算开销。

注意，如果你使用预训练模型进行LoRA微调，而非chat模型，模型的embedding和输出层的参数将被设为可训练的参数。这是因为预训练模型没有学习过ChatML格式中的特殊token，因此需要将这部分参数设为可训练才能让模型学会理解和预测这些token。这也意味着，假如你的训练引入新的特殊token，你需要通过代码中的`modules_to_save`将这些参数设为可训练的参数。如果你想节省显存占用，可以考虑使用chat模型进行LoRA微调，显存占用将大幅度降低。下文的显存占用和训练速度的记录将详细介绍这部分细节。

### Q-LoRA
如果你依然遇到显存不足的问题，可以考虑使用Q-LoRA ([论文](https://arxiv.org/abs/2305.14314))。该方法使用4比特量化模型以及paged attention等技术实现更小的显存开销。运行Q-LoRA你只需运行如下脚本：

```bash
# 单卡训练
sh finetune/finetune_qlora_single_gpu.sh
# 分布式训练
sh finetune/finetune_qlora_ds.sh
```

我们建议你使用我们提供的Int4量化模型进行训练，即Qwen-VL-Chat-Int4。请**不要使用**非量化模型！与全参数微调以及LoRA不同，Q-LoRA仅支持fp16。此外，上述LoRA关于特殊token的问题在Q-LoRA依然存在。并且，Int4模型的参数无法被设为可训练的参数。所幸的是，我们只提供了Chat模型的Int4模型，因此你不用担心这个问题。但是，如果你执意要在Q-LoRA中引入新的特殊token，很抱歉，我们无法保证你能成功训练。

与全参数微调不同，LoRA和Q-LoRA的训练只需存储adapter部分的参数。假如你需要使用LoRA训练后的模型，你需要使用如下方法。你可以用如下代码读取模型：

```python
from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained(
    path_to_adapter, # path to the output directory
    device_map="auto",
    trust_remote_code=True
).eval()
```

如果你觉得这样一步到位的方式让你很不安心或者影响你接入下游应用，你可以选择先合并并存储模型（LoRA支持合并，Q-LoRA不支持），再用常规方式读取你的新模型，示例如下：

```python
from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained(
    path_to_adapter, # path to the output directory
    device_map="auto",
    trust_remote_code=True
).eval()

merged_model = model.merge_and_unload()
# max_shard_size and safe serialization are not necessary. 
# They respectively work for sharding checkpoint and save the model to safetensors
merged_model.save_pretrained(new_model_directory, max_shard_size="2048MB", safe_serialization=True)
```

注意：分布式训练需要根据你的需求和机器指定正确的分布式训练超参数。此外，你需要根据你的数据、显存情况和训练速度预期，使用`--model_max_length`设定你的数据长度。


## 模型合并
```python
from peft import AutoPeftModelForCausalLM  # 确保导入所需的模块
from transformers import (
     AutoTokenizer
)
 
 
path_to_adapter = "/root/LLM-based-graph-tool/output_qwen"
# 从预训练模型中加载自定义适配器模型
model = AutoPeftModelForCausalLM.from_pretrained(
    path_to_adapter,  # 适配器的路径
    device_map="auto",  # 自动映射设备
    trust_remote_code=True  # 信任远程代码
).eval()  # 设置为评估模式
new_model_directory = "/root/LLM-based-graph-tool/output_qwen/merged"
tokenizer = AutoTokenizer.from_pretrained(
    path_to_adapter, trust_remote_code=True,
)
tokenizer.save_pretrained(new_model_directory)
 
# 合并并卸载模型
merged_model = model.merge_and_unload()
# 保存合并后的模型
merged_model.save_pretrained(new_model_directory, max_shard_size="2048MB", safe_serialization=True)
```

合并完之后可以使用推理代码进行推理
