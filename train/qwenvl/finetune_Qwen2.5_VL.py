from PIL import Image
import torch
import torchvision.models as models
from torch import nn
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from swanlab.integration.transformers import SwanLabCallback
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
)
import swanlab
import json
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.models as models
 
import torchvision.transforms as T
 
# 图像预处理变换
patch_h = 75
patch_w = 50
feat_dim = 384
 
transform = T.Compose([
    T.GaussianBlur(9, sigma=(0.1, 2.0)),
    T.Resize((patch_h * 14, patch_w * 14)),
    T.CenterCrop((patch_h * 14, patch_w * 14)),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
 
# 加载DINOv2模型
dinov2_vits14 = torch.hub.load('', 'dinov2_vits14', source='local').cuda()
 
class CustomResNet(nn.Module):
    def __init__(self, output_size=(256, 1176)):
        super(CustomResNet, self).__init__()
        
        # 预训练的 ResNet 模型
        resnet = models.resnet50(pretrained=True)
        
        # 去掉 ResNet 的最后全连接层和池化层
        self.features = nn.Sequential(*list(resnet.children())[:-2])  # 去掉最后的FC层和AvgPool层
        
        # 自定义的卷积层，调整步幅和padding来控制尺寸
        self.conv1 = nn.Conv2d(2048, 2048, kernel_size=3, stride=1, padding=1)  # 保持大小
        self.conv2 = nn.Conv2d(2048, 2048, kernel_size=3, stride=1, padding=1)  # 保持大小
        self.conv3 = nn.Conv2d(2048, 2048, kernel_size=3, stride=1, padding=1)  # 保持大小
        
        # 上采样层，用于增加特征图的尺寸
        self.upconv1 = nn.ConvTranspose2d(2048, 2048, kernel_size=4, stride=4, padding=0)  # 上采样
        self.upconv2 = nn.ConvTranspose2d(2048, 2048, kernel_size=4, stride=4, padding=0)  # 上采样
        
        # 最终卷积层将特征图变为单通道输出（灰度图）
        self.final_conv = nn.Conv2d(2048, 1, kernel_size=1)  # 输出单通道
 
    def forward(self, x):
        # 获取ResNet的特征图
        x = self.features(x)
        
        # 经过卷积层
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # 上采样阶段：增加特征图的尺寸
        x = self.upconv1(x)  # 上采样1
        x = self.upconv2(x)  # 上采样2
        
        # 使用插值进行微调输出尺寸
        x = F.interpolate(x, size=(256, 1176), mode='bilinear', align_corners=False)
        
        # 通过最后的卷积层输出（单通道）
        x = self.final_conv(x)  # 通过最后的卷积层输出
        
        return x
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
 
# 创建模型并移动到设备上
model_ResNet = CustomResNet(output_size=(256, 1176)).to(device)
 
# 定义图像预处理过程
image_transform = transforms.Compose([
    transforms.Resize((800, 800)),  # 确保图像大小一致（通常为224x224）
    transforms.ToTensor(),  # 转换为Tensor并标准化
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])
 
def extract_resnet_features(image_path):
    """
    使用ResNet提取图像特征
    """
    image = Image.open(image_path).convert("RGB")  # 加载图像并转换为RGB
    image_tensor = image_transform(image).unsqueeze(0).to('cuda')  # 添加batch维度并转换为cuda Tensor
    # features = resnet_extractor(image_tensor)  # 从ResNet提取特征    
    features = model_ResNet(image_tensor)
 
    return features
 
def process_func(example):
    """
    将数据集进行预处理,加入ResNet特征提取
    """
    MAX_LENGTH = 8192
    input_ids, attention_mask, labels = [], [], []
    conversation = example["conversations"]
    input_content = conversation[0]["value"]
    output_content = conversation[1]["value"]
    file_path = input_content.split("<|vision_start|>")[1].split("<|vision_end|>")[0]  # 获取图像路径
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"{file_path}",
                    "resized_height": 224,  # 确保图像尺寸为224x224
                    "resized_width": 224,
                },
                {"type": "text", "text": "COCO Yes:"},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )  # 获取文本
    image_inputs, video_inputs = process_vision_info(messages)  # 获取数据数据（预处理过）
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
 
    # print("inputs['pixel_values'] shape: ", inputs['pixel_values'].shape)
 
    # 获取图像特征
    img = Image.open(file_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).cuda()  # 转换为tensor并移至GPU
    
    # 使用DINOv2提取特征
    with torch.no_grad():
        features_dict = dinov2_vits14.forward_features(img_tensor)
        features = features_dict['x_norm_patchtokens']  # 提取图像特征
 
    # Print the original features shape
    # print("Original features shape: ", features.shape)  # Expected to be (batch_size, num_patches, feature_dim)
 
    # Reshape the feature map to 2D (flatten patches)
    features_reshaped = features.reshape(-1, feat_dim).cuda()  # Flatten patches, ensure it's on the same device as the model
 
    # Print reshaped features shape and total number of elements
    # print(f"Reshaped features shape: {features_reshaped.shape}")
    total_elements = features_reshaped.numel()
    # print(f"Total number of elements in reshaped features: {total_elements}")
 
    # Define the target shape we want to downsample to: (256, 1176)
    desired_rows = 256
    desired_columns = 1176
 
    # Check the total number of elements in the reshaped feature
    desired_total_elements = 4 * desired_rows * desired_columns
    # print(f"Desired total elements for [4, 256, 1176]: {desired_total_elements}")
 
    # Check if we have enough elements
    # if total_elements != desired_total_elements:
    #     print(f"Warning: The total number of elements in reshaped features ({total_elements}) does not match the target shape ({desired_total_elements}).")
    #     print(f"Proceeding with element reshaping.")
 
    # Reshape the features before downsampling, based on required elements.
    # We will downsample both rows and columns to fit the desired total number of elements.
    # First, calculate the downsampling factor for the rows.
 
    current_num_patches = features.shape[1]  # This is the number of patches (3750 originally)
 
    # Check if the number of elements is compatible with the desired total number of elements
    if total_elements != desired_total_elements:
        # If the element number doesn't match, we need to adjust the downsampling strategy.
        # Downsample the number of patches using an adaptive pooling method.
        downsampled_features_patches = nn.AdaptiveAvgPool1d(desired_rows)(features_reshaped.T).T
 
        # Now, adjust feature dimension (384 -> 1176)
        linear_transform = nn.Linear(feat_dim, desired_columns).cuda()
        downsampled_features = linear_transform(downsampled_features_patches)
 
        # Print the shape after downsampling
        # print("Downsampled features shape: ", downsampled_features.shape)
    else:
        downsampled_features = features_reshaped
 
    # Now we reshape the final output to match the desired shape
    # Ensure the size is compatible with the target shape
    # print("&&&&&&&&&&&&&&Downsampled features shape: ", downsampled_features.shape)
    
    inputs['pixel_values'] = downsampled_features  
 
    inputs = {key: value.tolist() for key, value in inputs.items()}  # tensor -> list,为了方便拼接
    instruction = inputs
 
    response = tokenizer(f"{output_content}", add_special_tokens=False)
 
 
    input_ids = (
            instruction["input_ids"][0] + response["input_ids"] + [tokenizer.pad_token_id]
    )
 
    attention_mask = instruction["attention_mask"][0] + response["attention_mask"] + [1]
    labels = (
            [-100] * len(instruction["input_ids"][0])
            + response["input_ids"]
            + [tokenizer.pad_token_id]
    )
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
 
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    labels = torch.tensor(labels)
    inputs['pixel_values'] = torch.tensor(inputs['pixel_values'])
    inputs['image_grid_thw'] = torch.tensor(inputs['image_grid_thw']).squeeze(0)  # 由（1,h,w)变换为（h,w）
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels,
            "pixel_values": inputs['pixel_values'], "image_grid_thw": inputs['image_grid_thw']}
 
 
def predict(messages, model):
    # 准备推理
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
 
    # 生成输出
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0]
 
 
# 在modelscope上下载Qwen2-VL模型到本地目录下
model_dir = snapshot_download("Qwen/Qwen2-VL-2B-Instruct", cache_dir="./", revision="master")
 
# 使用Transformers加载模型权重
tokenizer = AutoTokenizer.from_pretrained("./Qwen/Qwen2-VL-2B-Instruct/", use_fast=False, trust_remote_code=True)
processor = AutoProcessor.from_pretrained("./Qwen/Qwen2-VL-2B-Instruct")
 
# 加载模型
model = Qwen2VLForConditionalGeneration.from_pretrained("./Qwen/Qwen2-VL-2B-Instruct/", device_map="cuda", torch_dtype=torch.bfloat16, trust_remote_code=True,)
model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法
model.config.use_cache = False
 
# 处理数据集：读取json文件
# 拆分成训练集和测试集，保存为data_vl_train.json和data_vl_test.json
train_json_path = "data_vl.json"
with open(train_json_path, 'r') as f:
    data = json.load(f)
    train_data = data[:-4]
    test_data = data[-4:]
 
with open("data_vl_train.json", "w") as f:
    json.dump(train_data, f)
 
with open("data_vl_test.json", "w") as f:
    json.dump(test_data, f)
 
train_ds = Dataset.from_json("data_vl_train.json")
train_dataset = train_ds.map(process_func)
 
# 配置LoRA
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=64, #64,  # Lora 秩
    lora_alpha= 16, #16,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.05,  # Dropout 比例
    bias="none",
)
 
# 获取LoRA模型
peft_model = get_peft_model(model, config)
 
# 配置训练参数
args = TrainingArguments(
    output_dir="./output/Qwen2-VL-2B",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=10,
    logging_first_step=5,
    num_train_epochs=2,
    save_steps=100,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
)
 
# 配置Trainer
trainer = Trainer(
    model=peft_model,
    args=args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)
 
# 开启模型训练
trainer.train()
 
 
# ====================测试模式===================
# 配置测试参数
val_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=True,  # 训练模式
    r=64,#64,  # Lora 秩
    lora_alpha=16,#16,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.05,  # Dropout 比例
    bias="none",
)
 
# 获取测试模型
val_peft_model = PeftModel.from_pretrained(model, model_id="./output/Qwen2-VL-2B/checkpoint-62", config=val_config)
 
# 读取测试数据
with open("data_vl_test.json", "r") as f:
    test_dataset = json.load(f)
 
test_image_list = []
for item in test_dataset:
    input_image_prompt = item["conversations"][0]["value"]
    # 去掉前后的<|vision_start|>和<|vision_end|>
    origin_image_path = input_image_prompt.split("<|vision_start|>")[1].split("<|vision_end|>")[0]
    
    messages = [{
        "role": "user", 
        "content": [
            {
            "type": "image", 
            "image": origin_image_path
            },
            {
            "type": "text",
            "text": "COCO Yes:"
            }
        ]}]
    
    response = predict(messages, val_peft_model)
    messages.append({"role": "assistant", "content": f"{response}"})
    print(messages[-1])
 
    test_image_list.append(swanlab.Image(origin_image_path, caption=response))