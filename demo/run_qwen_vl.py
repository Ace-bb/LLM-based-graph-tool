from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
torch.manual_seed(1234)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL", trust_remote_code=True)

# 打开bf16精度，A100、H100、RTX3060、RTX3070等显卡建议启用以节省显存
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL", device_map="auto", trust_remote_code=True, bf16=True).eval()
# 打开fp16精度，V100、P100、T4等显卡建议启用以节省显存
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL", device_map="auto", trust_remote_code=True, fp16=True).eval()
# 使用CPU进行推理，需要约32GB内存
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL", device_map="cpu", trust_remote_code=True).eval()
# 默认gpu进行推理，需要约24GB显存
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL", device_map="cuda", trust_remote_code=True).eval()

# 可指定不同的生成长度、top_p等相关超参（transformers 4.32.0及以上无需执行此操作）
# model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL", trust_remote_code=True)

query = tokenizer.from_list_format([
    {'image': '/root/LLM-based-graph-tool/demo/data/input/book/50.png'}, # Either a local path or an url
    {'text': 'Make a layout analysis of this picture in English with grounding:'},
])
inputs = tokenizer(query, return_tensors='pt')
inputs = inputs.to(model.device)
pred = model.generate(**inputs)
response = tokenizer.decode(pred.cpu()[0], skip_special_tokens=False)
print(response)
# <img>https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg</img>Generate the caption in English with grounding:<ref> Woman</ref><box>(451,379),(731,806)</box> and<ref> her dog</ref><box>(219,424),(576,896)</box> playing on the beach<|endoftext|>
image = tokenizer.draw_bbox_on_latest_picture(response)
if image:
    image.save('./data/output/2.jpg')
else:
    print("no box")

while True:
    try:
        history=None
        img_path = input("输入图片path或url：")
        text = input("问题：")
        if len(img_path) !=0 and img_path!="" and img_path!=None:
            query = tokenizer.from_list_format([
                {'image': img_path}, # Either a local path or an url
                {'text': text},
            ])
        else:
            query = text
        print(query)
        inputs = tokenizer(query, return_tensors='pt')
        inputs = inputs.to(model.device)
        pred = model.generate(**inputs)
        # response, history = model.chat(tokenizer, '框出图中Fatigue的位置', history=history)
        response = tokenizer.decode(pred.cpu()[0], skip_special_tokens=False)
        print(response)
        # <ref>击掌</ref><box>(536,509),(588,602)</box>
        image = tokenizer.draw_bbox_on_latest_picture(response)
        print(image)
        if image:
            image.save('./data/output/1.jpg')
        else:
            print("no box")
    except Exception as e:
        print(e)
        break
        