from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
torch.manual_seed(1234)

# 请注意：分词器默认行为已更改为默认关闭特殊token攻击防护。
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

# 打开bf16精度，A100、H100、RTX3060、RTX3070等显卡建议启用以节省显存
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
# 打开fp16精度，V100、P100、T4等显卡建议启用以节省显存
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
# 使用CPU进行推理，需要约32GB内存
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cpu", trust_remote_code=True).eval()
# 默认gpu进行推理，需要约24GB显存
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True).eval()

# 可指定不同的生成长度、top_p等相关超参（transformers 4.32.0及以上无需执行此操作）
# model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

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
    response, history = model.chat(tokenizer, '框出图中Fatigue的位置', history=history)
    print(response)
    # <ref>击掌</ref><box>(536,509),(588,602)</box>
    image = tokenizer.draw_bbox_on_latest_picture(response, history)
    print(image)
    if image:
      image.save('./data/output/1.jpg')
    else:
      print("no box")
  except Exception as e:
    print(e)
    break
    
# 第一轮对话
# query = tokenizer.from_list_format([
#     {'image': '/root/LLM-based-graph-tool/demo/data/input/Fatigue.png'}, # Either a local path or an url
#     {'text': '这是什么?'},
# ])
# response, history = model.chat(tokenizer, query=query, history=None)
# print(response)
# # 图中是一名女子在沙滩上和狗玩耍，旁边是一只拉布拉多犬，它们处于沙滩上。

# # 第二轮对话
# response, history = model.chat(tokenizer, '框出图中Fatigue的位置', history=history)
# print(response)
# # <ref>击掌</ref><box>(536,509),(588,602)</box>
# image = tokenizer.draw_bbox_on_latest_picture(response, history)
# if image:
#   image.save('./data/output/1.jpg')
# else:
#   print("no box")