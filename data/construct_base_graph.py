from xml.etree import ElementTree as et
import xml.dom.minidom as minidom
import json
import cv2
import os
import numpy as np
import random
from tqdm import tqdm
import math

train_images_savepath = './datasets/train/images'
train_annots_savepath = './datasets/train/annots'
valid_images_savepath = './datasets/validation/images'
valid_annots_savepath = './datasets/validation/annots'
# VOC_base_path = "/root/nas/projects/deep-learning-for-image-processing/pytorch_object_detection/faster_rcnn/Datasets/v15_arrow_mix/VOCdevkit/VOC2012"
VOC_base_path = "../FasterRCNN/Datasets/v3OnlyLine/VOCdevkit/VOC2012"
VOC_img_savepath = f"{VOC_base_path}/JPEGImages"
VOC_annots_savepath = f"{VOC_base_path}/Annotations"
VOC_main_savepath = f"{VOC_base_path}/ImageSets/Main"
# SHAPE_CLASSES = {
#     "process": 1,
#     "decision": 2,
#     "start_end": 3,
#     "scan": 4,
#     "arrow": 5,
#     "line": 6
# }
SHAPE_CLASSES = {
    # "arrow": 1,
    "line": 2
}
# SHAPE_CLASSES_RANDOM_RATE=[0.6, 0.03, 0.03, 0.04, 0.3]
# SHAPE_CLASSES_RANDOM_RATE=[0.4, 0.03, 0.03, 0.03, 0.05, 0.46]
SHAPE_CLASSES_RANDOM_RATE=[1]

min_w, min_y = 40, 40
def load_frdetr_datassets(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        frdetr_datassets = json.load(f)
    return frdetr_datassets['images'], frdetr_datassets['annotations'], frdetr_datassets['categories']

def split_data():
    random.seed(0)  # 设置随机种子，保证随机结果可复现

    files_path = f"{VOC_base_path}/Annotations"
    assert os.path.exists(files_path), "path: '{}' does not exist.".format(files_path)
    val_rate = 0.5

    files_name = sorted([file.split(".")[0] for file in os.listdir(files_path)])
    files_num = len(files_name)
    val_index = random.sample(range(0, files_num), k=int(files_num*val_rate))
    train_files = []
    val_files = []
    for index, file_name in enumerate(files_name):
        if index in val_index:
            val_files.append(file_name)
        else:
            train_files.append(file_name)
    try:
        train_f = open(f"{VOC_base_path}/ImageSets/Main/train.txt", "x")
        eval_f = open(f"{VOC_base_path}/ImageSets/Main/val.txt", "x")
        train_f.write("\n".join(train_files))
        eval_f.write("\n".join(val_files))
    except FileExistsError as e:
        print(e)
        exit(1)
        
def construct_xml_obj(folder, filename, img_size, objects, save_path):
    root = et.Element('annotation')
    et.SubElement(root, 'folder').text = folder
    et.SubElement(root, 'filename').text = f"{filename}.jpg"
    et.SubElement(root, 'path').text = f"../images/{filename}.jpg"
    source = et.SubElement(root, 'source')
    et.SubElement(source, 'database').text = 'flowchart_datasets'
    size = et.SubElement(root, 'size')
    et.SubElement(size, 'width').text = str(img_size['width'])
    et.SubElement(size, 'height').text = str(img_size['height'])
    et.SubElement(size, 'depth').text = '3'
    et.SubElement(root, 'segmented').text = '0'
    for img_obj in objects:
        obj = et.SubElement(root, 'object')
        et.SubElement(obj, 'name').text = img_obj['name'] 
        et.SubElement(obj, 'pose').text = 'Unspecified'
        et.SubElement(obj, 'truncated').text = '0'
        et.SubElement(obj, 'difficult').text = '0'
        bndbox = et.SubElement(obj, 'bndbox')
        et.SubElement(bndbox, 'xmin').text = str(img_obj['xmin'])
        et.SubElement(bndbox, 'ymin').text = str(img_obj['ymin'])
        et.SubElement(bndbox, 'xmax').text = str(img_obj['xmax'])
        et.SubElement(bndbox, 'ymax').text = str(img_obj['ymax'])
    tree = et.ElementTree(root)
    # tree.write(f"save_path/{filename}.xml", encoding = 'utf-8')
    rough_str = et.tostring(root, 'utf-8')
    # 格式化
    reparsed = minidom.parseString(rough_str)
    new_str = reparsed.toprettyxml(indent='\t')
    with open(f"{save_path}/{filename}.xml", 'w', encoding='utf-8') as f:
        f.write(new_str)
    # rawtext = et.tostring(root)
    # dom = minidom.parseString(rawtext)
    # with open("output.xml", "w") as f:
    #     dom.writexml(f, indent="\t", newl="", encoding="utf-8")

def generate_arrow(img, x_start, y_start, lineType, sw, sh):
    arrow_list = ['arrow_line_up', 'arrow_line_down', 'arrow_line_left', 'arrow_line_right']
    shape_type = np.random.choice(arrow_list, 1, p=[0.25, 0.25, 0.25, 0.25])[0]
    line_data = list()
    arrow_data = list()
    a_w = random.randint(4,10)# 箭头的宽度
    a_h = random.randint(5,12)# 箭头的高度
    rand_arrow_type = random.random()
    xmin, ymin, xmax, ymax = -1, -1, -1, -1
    
    xs, ys = random.randint(sw//8, sw//4-1), random.randint(sw//8, sh//4-1)
    line_len = random.randint(sh//4, 3*sh//4)
    if shape_type == 'arrow_line_up' or shape_type == 'arrow_line_down':
        xs, ys = random.randint(sw//4, 3*sw//4-1), random.randint(sw//8, sh//4-1)
        xmin, ymin, xmax, ymax = x_start+xs, y_start+ys, x_start+xs, y_start+ys+line_len
    # elif shape_type == 'arrow_line_down':
    #     xmin, ymin, xmax, ymax = x_start+xs, y_start+ys, x_start+xs, y_start+ys+line_len
    # elif shape_type == 'arrow_line_left':
    #     xmin, ymin, xmax, ymax = x_start+xs-line_len, y_start+ys, x_start+xs, y_start+ys
    else:
        xs, ys = random.randint(sw//8, sh//4-1), random.randint(sw//4, 3*sw//4-1)
        xmin, ymin, xmax, ymax = x_start+xs, y_start+ys, x_start+xs+line_len, y_start+ys
    
    if rand_arrow_type > 0.4: # 画直线
        cv2.line(img, (xmin, ymin), (xmax, ymax), (0,0,0), lineType)
        line_data.append({
            "name": "line",
            "xmin": xmin-lineType,
            "ymin": ymin-lineType,
            "xmax": xmax+lineType,
            "ymax": ymax+lineType
        })
        # cv2.rectangle(img, (xmin-lineType*2, ymin-lineType*2), (xmax+lineType*2, ymax+lineType*2), (255, 0, 0), 1)
    elif rand_arrow_type > 0: # 画折线
        bios = random.randint(30, 50)*np.random.choice([-1, 1], 1, p=[0.5, 0.5])[0] # 随机往两个方向偏移2
        bios = int(bios)
        turn_l = random.randint(line_len//4, 3*line_len//4) 
        if shape_type == 'arrow_line_up': # 偏移终点
            xmin, ymin = xmin + bios, ymin
            line_turn_start_point = (xmin, ymin+turn_l)
            line_turn_end_point = (xmax, ymin+turn_l)
        elif shape_type == 'arrow_line_down':
            xmax, ymax = xmax + bios, ymax
            line_turn_start_point = (xmin, ymin+turn_l)
            line_turn_end_point = (xmax, ymin+turn_l)
        elif shape_type == 'arrow_line_left':
            xmin, ymin = xmin, ymin+bios
            line_turn_start_point = (xmin+turn_l, ymin)
            line_turn_end_point = (xmin+turn_l, ymax)
        else:
            xmax, ymax = xmax, ymax + bios
            line_turn_start_point = (xmin+turn_l, ymin)
            line_turn_end_point = (xmin+turn_l, ymax)
        cv2.line(img, (xmin, ymin), line_turn_start_point, (0,0,0), lineType)
        cv2.line(img, line_turn_start_point, line_turn_end_point, (0,0,0), lineType)
        cv2.line(img, line_turn_end_point, (xmax, ymax), (0,0,0), lineType)
        line_data.append({ "name": "line", 
                        "xmin": min(xmin, line_turn_start_point[0])-lineType, "ymin": min(ymin, line_turn_start_point[1])-lineType, 
                        "xmax": max(xmin, line_turn_start_point[0])+lineType, "ymax": max(ymin, line_turn_start_point[1])+lineType })
        line_data.append({ "name": "line", 
                        "xmin": min(line_turn_start_point[0], line_turn_end_point[0])-lineType, "ymin": min(line_turn_start_point[1], line_turn_end_point[1])-lineType, 
                        "xmax": max(line_turn_start_point[0], line_turn_end_point[0])+lineType, "ymax": max(line_turn_start_point[1], line_turn_end_point[1])+lineType })
        line_data.append({ "name": "line", 
                        "xmin": min(xmax, line_turn_end_point[0])-lineType, "ymin": min(ymax, line_turn_end_point[1])-lineType, 
                        "xmax": max(xmax, line_turn_end_point[0])+lineType, "ymax": max(ymax, line_turn_end_point[1])+lineType })
        # cv2.rectangle(img, 
        #             (min(xmin, line_turn_start_point[0])-lineType*3, min(ymin, line_turn_start_point[1])-lineType*3), 
        #             (max(xmin, line_turn_start_point[0])+lineType*3, max(ymin, line_turn_start_point[1])+lineType*3), (0, 0, 255), 1)
        # cv2.rectangle(img, 
        #             (min(line_turn_start_point[0], line_turn_end_point[0])-lineType*3, min(line_turn_start_point[1], line_turn_end_point[1])-lineType*3), 
        #             (max(line_turn_start_point[0], line_turn_end_point[0])+lineType*3, max(line_turn_start_point[1], line_turn_end_point[1])+lineType*3), (0, 0, 255), 1)
        # cv2.rectangle(img, 
        #             (min(xmax, line_turn_end_point[0])-lineType*3, min(ymax, line_turn_end_point[1])-lineType*3), 
        #             (max(xmax, line_turn_end_point[0])+lineType*3, max(ymax, line_turn_end_point[1])+lineType*3), (0, 0, 255), 1)
    else: # 画斜线
        ...
    
    # 画箭头
    if shape_type == 'arrow_line_up':
        acp = (xmin,ymin) # 箭头的中心点
        p1, p2, p3 = [acp[0]-a_w//2, acp[1]], [acp[0], acp[1]-a_h], [acp[0]+a_w//2, acp[1]]
    elif shape_type == 'arrow_line_down':
        acp = (xmax,ymax) # 箭头的中心点
        p1, p2, p3 = [acp[0]-a_w//2, acp[1]], [acp[0], acp[1]+a_h], [acp[0]+a_w//2, acp[1]]
    elif shape_type == 'arrow_line_left':
        acp = (xmin,ymin) # 箭头的中心点
        p1, p2, p3 = [acp[0], acp[1]-a_h//2], [acp[0]-a_w, acp[1]], [acp[0], acp[1]+a_h//2]
    else:
        acp = (xmax,ymax) # 箭头的中心点
        p1, p2, p3 = [acp[0], acp[1]-a_h//2], [acp[0]+a_w, acp[1]], [acp[0], acp[1]+a_h//2]
    
    cv2.fillPoly(img, [np.array([p1, p2, p3])], (0, 0, 0)) # 画实心箭头
    arrow_data.append({
        "name": "arrow",
        "xmin": min(p1[0], p2[0], p3[0]),
        "ymin": min(p1[1], p2[1], p3[1]),
        "xmax": max(p1[0], p2[0], p3[0]),
        "ymax": max(p1[1], p2[1], p3[1])
    })
    
    return line_data, arrow_data

def generate_img(img_width=600, img_height=600, shape_num=9, save_path='./', file_name='1.png'):
    # shape_list = ['process', 'start_end', 'decision', 'scan', 'arrow']
    shape_list = list(SHAPE_CLASSES.keys())
    img = np.ones((img_width, img_height, 3), dtype=np.uint8)
    img *= 255 # white background
    rn = int(math.sqrt(shape_num))
    sw, sh = img_width // rn, img_height // rn
    img_objects = list()
    for i in range(shape_num):
        x_start, y_start = sw * (i%rn), sh * ((i//rn)%rn)
        lineType = random.randint(1,3)
        shape_type = np.random.choice(shape_list, 1, p=SHAPE_CLASSES_RANDOM_RATE)[0]
        if shape_type == 'process':
            xmin, ymin = random.randint(sw//10, sw//6-1), random.randint(sw//8, sh//5-1)
            xmax, ymax = random.randint(sw//4, sw-4), random.randint(sh//4, 2*sw//3)
            # xmax, ymax = xmin+140, ymin+60
            cv2.rectangle(img, (x_start+xmin, y_start+ymin), (x_start+xmax, y_start+ymax), (0, 0, 0), lineType)
            img_objects.append({"name":shape_type, "xmin": x_start+xmin, "ymin": y_start+ymin, "xmax": x_start+xmax, "ymax": y_start+ymax})
        elif shape_type == 'start_end': # 椭圆
            centre_x, centre_y = random.randint(2*sw//5, 3*sw//5), random.randint(2*sh//5, 3*sh//5)
            axes_x, axes_y = random.randint(1*sw//6, 2*sw//5), random.randint(sw//8, sw//3)
            cv2.ellipse(img, (x_start+centre_x, y_start+centre_y), (max(axes_x, axes_y), min(axes_x, axes_y)), 0, 0, 360, (0, 0, 0), lineType) #画椭圆
            img_objects.append({"name":shape_type, 
                                "xmin": x_start+centre_x-max(axes_x, axes_y), 
                                "ymin": y_start+centre_y-min(axes_x, axes_y), 
                                "xmax": x_start+centre_x+max(axes_x, axes_y), 
                                "ymax": y_start+centre_y+min(axes_x, axes_y)})
        elif shape_type == 'circle': #圆
            centre_x, centre_y = random.randint(2*sw//5, 3*sw//5), random.randint(2*sh//5, 3*sh//5)
            # radius = random.randint(60, min(centre_x, sw-centre_x, centre_y, sh-centre_y)-4)
            radius = random.randint(1*sw//6, 2*sw//5)
            cv2.circle(img, (x_start+centre_x,y_start+centre_y), radius, (0,0,0), lineType)
            img_objects.append({"name":shape_type, 
                                "xmin": x_start+centre_x-radius, 
                                "ymin": y_start+centre_y-radius, 
                                "xmax": x_start+centre_x+radius, 
                                "ymax": y_start+centre_y+radius})
            
        elif shape_type == 'decision': #菱形
            centre_x, centre_y = random.randint(2*sw//5, 3*sw//5), random.randint(2*sh//5, 3*sh//5)
            # axes_x, axes_y = random.randint(min_w, min(centre_x, sw-centre_x)-4), random.randint(min_y, min(centre_y, sh-centre_y)-4)
            axes_x, axes_y = random.randint(sw//6, 2*sw//5), random.randint(sw//6, 2*sw//5)
            pts = np.array([[x_start+centre_x, y_start+centre_y-axes_y], 
                            [x_start+centre_x+axes_x, y_start+centre_y], 
                            [x_start+centre_x, y_start+centre_y+axes_y], 
                            [x_start+centre_x-axes_x, y_start+centre_y]])
            cv2.polylines(img, [pts], True, (0, 0, 0), lineType)
            img_objects.append({"name":shape_type, 
                                "xmin": x_start+centre_x-axes_x, 
                                "ymin": y_start+centre_y-axes_y, 
                                "xmax": x_start+centre_x+axes_x, 
                                "ymax": y_start+centre_y+axes_y})
            
        elif shape_type == 'scan': # 平行四边形
            xmin, ymin = random.randint(0, sw//2-min_w), random.randint(0, sh//2-min_y)
            # xmax, ymax = random.randint(sw//2+min_w, sw-4), random.randint(sh//2+min_y, sh//2+min_y+30)
            xmax, ymax = xmin + 100, ymin+40
            bios = random.randint(6, (xmax-xmin)//3)
            pts = np.array([[x_start+xmin+bios, y_start+ymin], [x_start+xmax, y_start+ymin], [x_start+xmax-bios, y_start+ymax], [x_start+xmin, y_start+ymax]])
            cv2.polylines(img, [pts], True, (0, 0, 0), lineType)
            img_objects.append({"name":shape_type, "xmin": x_start+xmin, "ymin": y_start+ymin, "xmax": x_start+xmax, "ymax": y_start+ymax})
        elif shape_type == 'arrow':
            # continue
            amax_wh = 24
            xmin, ymin = random.randint(amax_wh, sw-amax_wh), random.randint(amax_wh, sh-amax_wh)
            a_w, a_h = random.randint(8,amax_wh), random.randint(8,amax_wh)# 箭头的宽度
            xmax, ymax = xmin + a_w, ymin+a_h
            arrow_list = ['up', 'down', 'left', 'right']
            rand_arrow_type = np.random.choice(arrow_list, 1, p=[0.25, 0.25, 0.25, 0.25])[0]
            if rand_arrow_type == 'up':
                pts = np.array([[x_start+xmin, y_start+ymax], [x_start+(xmin+xmax)//2, y_start+ymin], [x_start+xmax, y_start+ymax]])
            elif rand_arrow_type == 'down':
                pts = np.array([[x_start+xmin, y_start+ymin], [x_start+(xmin+xmax)//2, y_start+ymax], [x_start+xmax, y_start+ymin]])
            elif rand_arrow_type == 'left':
                pts = np.array([[x_start+xmax, y_start+ymin], [x_start+xmin, y_start+(ymin+ymax)//2], [x_start+xmax, y_start+ymax]])
            else:
                pts = np.array([[x_start+xmin, y_start+ymin], [x_start+xmax, y_start+(ymin+ymax)//2], [x_start+xmin, y_start+ymax]])
            
            cv2.fillPoly(img, [pts], (0, 0, 0)) # 画实心箭头
            img_objects.append({"name":shape_type, "xmin": x_start+xmin, "ymin": y_start+ymin, "xmax": x_start+xmax, "ymax": y_start+ymax})
        elif shape_type == "line":
            line_data, arrow_data = generate_arrow(img, x_start, y_start, lineType, sw, sh)
            img_objects.extend(line_data)
            img_objects.extend(arrow_data)
    
    cv2.imwrite(f"{save_path}/{file_name}", img)
    return img_objects

def generate_datasets(args):
    for i in tqdm(range(args['num']), desc="Datasets:"):
        i_w = random.randint(60, 100)*10
        i_h = random.randint(i_w//10, 101)*10
        max_item_num= np.random.choice([4, 9, 16], 1, p=[0.1, 0.3,0.6])[0]
        img_objects = generate_img(i_w, i_w, max_item_num, args['images_savepath'], f"{i}.jpg")
        construct_xml_obj(args['folder'], i, {"width":i_w, "height":i_h}, img_objects, args['annots_savepath'])

def construct_frdetr(args):
    with open(args['datasets_path'], 'r', encoding='utf-8') as f:
        frdetr_datasets = json.load(f)
    images_mapping = frdetr_datasets['images']
    images_annotations = frdetr_datasets['annotations']
    img_start_num = len(os.listdir(args['images_savepath']))
    shape_type_mapping = {
        '1': 'arrow',
        # '2': 'process',
        # '3': 'decision',
        # '6': 'start_end',
        # '8': 'scan',
        "10": "line"
    }
    for imgs in tqdm(images_mapping, total=len(images_mapping), desc="Frdetr:"):
        # print(f"construct {img_start_num}th data!")
        if args['folder']== 'train':
            img = cv2.imread(f"./frdetr_dataset/train/train/{imgs['file_name']}")
        else:
            img = cv2.imread(f"./frdetr_dataset/val/val/{imgs['file_name']}")
        img_name = img_start_num
        cv2.imwrite(f"{args['images_savepath']}/{img_name}.jpg", img)
        img_id = imgs['id']
        img_annots = list(filter(lambda annot: annot['image_id']==img_id, images_annotations))
        img_objects = list()
        for annot in img_annots:
            if str(annot['category_id']) not in shape_type_mapping.keys(): continue
            if str(annot['category_id'])=="10":
                if int(annot['line'][3]) ==0: annot['line'][3] = 1
                if int(annot['line'][2]) ==0: annot['line'][2] = 1
                if int(annot['line'][3])<0:
                    img_objects.append({"name":shape_type_mapping[str(annot['category_id'])], 
                                "xmin": int(annot['line'][0]), "ymin": annot['line'][1]+int(annot['line'][3]), 
                                "xmax": int(annot['line'][0])+int(annot['line'][2]), 
                                "ymax": annot['line'][1]-int(annot['line'][3])})
                else:
                    img_objects.append({"name":shape_type_mapping[str(annot['category_id'])], 
                                "xmin": int(annot['line'][0]), "ymin": int(annot['line'][1]), 
                                "xmax": int(annot['line'][0])+int(annot['line'][2]), 
                                "ymax": annot['line'][1]+int(annot['line'][3])})
            else:
                img_objects.append({"name":shape_type_mapping[str(annot['category_id'])], 
                                "xmin": int(annot['bbox'][0]), "ymin": int(annot['bbox'][1]), 
                                "xmax": int(annot['bbox'][0])+int(annot['bbox'][2]), 
                                "ymax": annot['bbox'][1]+int(annot['bbox'][3])})
        
        construct_xml_obj(args['folder'], img_name, {"width":imgs['width'], "height":imgs['height']}, img_objects, args['annots_savepath'])
        img_start_num+=1
    
    
train_images_num = 1000
validation_images_num = 120
# cv2.imshow('result', img)
train_args = {
    'folder': 'train',
    'num': 8000,
    'images_savepath': VOC_img_savepath,
    'annots_savepath': VOC_annots_savepath
}
valid_args = {
    'folder': 'validation',
    'num': 300,
    'images_savepath': VOC_img_savepath,
    'annots_savepath': VOC_annots_savepath
}
frdetr_train_args = {
    'folder': 'train',
    'datasets_path': './frdetr_dataset/annotations/annotations/frdetr_train2017.json',
    'images_savepath': VOC_img_savepath,
    'annots_savepath': VOC_annots_savepath
    # 'images_savepath': './frdetr_dataset/datasets/train/images',
    # 'annots_savepath': './frdetr_dataset/datasets/train/annots'
}
frdetr_valid_args = {
    'folder': 'validation',
    'datasets_path': './frdetr_dataset/annotations/annotations/frdetr_val2017.json',
    'images_savepath': VOC_img_savepath,
    'annots_savepath': VOC_annots_savepath
}
if __name__ == '__main__':
    # if not os.path.exists(train_images_savepath): os.makedirs(train_images_savepath)
    # if not os.path.exists(train_annots_savepath): os.makedirs(train_annots_savepath)
    # if not os.path.exists(valid_images_savepath): os.makedirs(valid_images_savepath)
    # if not os.path.exists(valid_annots_savepath): os.makedirs(valid_annots_savepath)
    if not os.path.exists(VOC_img_savepath): os.makedirs(VOC_img_savepath)
    if not os.path.exists(VOC_annots_savepath): os.makedirs(VOC_annots_savepath)
    if not os.path.exists(VOC_main_savepath): os.makedirs(VOC_main_savepath)
    generate_datasets(train_args)
    # generate_datasets(valid_args)
    construct_frdetr(frdetr_train_args)
    construct_frdetr(frdetr_valid_args)
    split_data()
    with open(f"{VOC_main_savepath}/classes.json", 'w', encoding="utf-8") as f:
        json.dump(SHAPE_CLASSES, f, ensure_ascii=False)
