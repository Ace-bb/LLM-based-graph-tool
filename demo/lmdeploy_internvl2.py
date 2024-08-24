from lmdeploy import pipeline
from lmdeploy.vl import load_image
import cv2, json

def draw_img(img_path, res):
    res = json.loads(res)
    image = cv2.imread(img_path)
    for item in res:
        cv2.rectangle(image, (item[0],item[1]), (item[2], item[3]), (0, 0, 255), 1)
    cv2.imwrite("/root/LLM-based-graph-tool/demo/data/output" + img_path.split("/")[-1], image)

img_path = '/root/LLM-based-graph-tool/data/datasets/internvl_baseGraphV1/Images/0.png'
pipe = pipeline('/root/LLM-based-graph-tool/output_internvl/InternVL2-2B/v1')

{
    # "arrow": 1, # √
    "rec": 2,   # √
    "diamond": 3, # √
    # "roundrec": 4,
    # "hex": 5,
    "ellipse": 6, # √
    "circle": 7, # √
    "parallel": 8, # √
    "line": 10 # √
}

image = load_image(img_path)
response = pipe(('Please provide the bounding box coordinate of the region this sentence describes: <ref>rec</ref>', image))
print(response.text[response.text.find("["):])
draw_img(img_path,response.text[response.text.find("["):])
while True:
    try:
        img_path = input("Image path:")
        tag = input("Tag:")
        image = load_image(img_path)
        response = pipe((f'Please provide the bounding box coordinate of the region this sentence describes: <ref>{tag}</ref>', image))
        print(response.text[response.text.find("["):])
        draw_img(img_path,response.text[response.text.find("["):])
    except:
        break
