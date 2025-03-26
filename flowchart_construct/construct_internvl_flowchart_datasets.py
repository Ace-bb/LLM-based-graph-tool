from xml.etree import ElementTree as et
import xml.dom.minidom as minidom
import json
import cv2
import os
import numpy as np
import random
from tqdm import tqdm
import math
import copy
import cv2
import numpy as np

        # point_size = 1
        # point_color = (0, 0, 255) # BGR
        # thickness = 4 # 可以为 0 、4、8

        # # 要画的点的坐标
        # # points_list = [(160, 160), (136, 160), (150, 200), (200, 180), (120, 150), (145, 180)]

        # # for point in points_list:
        #     cv2.circle(img, (200, 10), point_size, point_color, thickness)

VOC_base_path = "/root/LLM-based-graph-tool/data/datasets/InternVL2_flowchart_Dataset/vbaseall5000"
VOC_img_savepath = f"{VOC_base_path}/Images"
VOC_annots_savepath = f"{VOC_base_path}/dataset"
SHAPE_CLASSES = {
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
SHAPE_CLASSES_RANDOM_RATE=[1/len(SHAPE_CLASSES.keys())]*len(SHAPE_CLASSES.keys())

min_w, min_y = 40, 40        
data_num = 0

def construct_json_obj(folder, _id, img_size, objects, save_path):
    global data_num
    conversations = list()
    cur_dataset = list()
    
    
    # GrayImage=cv2.cvtColor(image ,cv2.COLOR_BGR2GRAY)
    # h, w = image.shape[:2]
    # h, w = map(int, [h/4, w/4])
    # print(h,w)
    # # no flip
    image = cv2.imread(save_path)
    for img_obj in objects:
        cv2.rectangle(image, (img_obj['xmin'], img_obj['ymin']), (img_obj['xmax'], img_obj['ymax']), (0, 0, 255))
        # cv2.line(img, (x_start, y_start), (img_width, y_start), (0,0,0), 1)
        # cv2.line(img, (x_start, y_start), (x_start, img_height), (0,0,0), 1)

    cv2.imwrite("/root/LLM-based-graph-tool/data/datasets/verify_dataset/basegraph/" + save_path.split("/")[-1], image)

    img_name_set = set([obj["name"] for obj in objects])
    for n in img_name_set:
        break
        tmp_conversations = list()
        tmp_conversations.append({
            "from": "human",
            "value": f"<image>\nPlease provide the bounding box coordinate of the region this sentence describes: <ref>{n}</ref>"
        })
        n_res = f"<ref>{n}</ref><box>["
        for img_obj in objects:
            if img_obj['name']!=n: continue
            n_res += f"[{img_obj['xmin']},{img_obj['ymin']},{img_obj['xmax']},{img_obj['ymax']}],"

        n_res = n_res[:-1] + "]</box>"
        tmp_conversations.append({"from": "gpt", "value": n_res})
        cur_dataset.append({"id": data_num,
            "image": save_path,
            "width": img_size["width"],
            "height": img_size["height"],
            "conversations": copy.deepcopy(tmp_conversations)
        })
        data_num+=1

    fi_conversations = []
    fi_conversations.append({"from": "human", "value": f"<image>\nPlease detect and label all objects in the following image and mark their positions."})
    n_res = "Sure, I will detect and label all objects in the image and mark their positions.\n\n```\n"
    for img_obj in objects:
        n_res += f"<ref>{img_obj['name']}</ref><box>[[{img_obj['xmin']},{img_obj['ymin']},{img_obj['xmax']},{img_obj['ymax']}], "

    n_res = n_res[:-1] + "]</box>\n```\n"
    fi_conversations.append({"from": "gpt", "value": n_res})

    cur_dataset.append({"id": data_num,
            "image": save_path,
            "width": img_size["width"],
            "height": img_size["height"],
            "conversations": fi_conversations})
    data_num+=1
    return cur_dataset

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
    img = np.ones((img_height, img_width, 3), dtype=np.uint8)
    img *= 255 # white background
    rn = int(math.sqrt(shape_num))
    sw, sh = (img_width) // rn, (img_height) // rn
    img_objects = list()
    for i in range(shape_num):
        lineType = random.randint(1,3)
        x_start, y_start =  sw * (i%rn), sh * ((i//rn)%rn)


        shape_type = np.random.choice(shape_list, 1, p=SHAPE_CLASSES_RANDOM_RATE)[0]
        if shape_type == 'rec':
            xmin, ymin = random.randint(sw//10, sw//6-1), random.randint(sh//8, sh//5-1)
            xmax, ymax = random.randint(sw//4, sw-4), random.randint(sh//4, sh//2)
            # xmax, ymax = xmin+140, ymin+60
            cv2.rectangle(img, (x_start+xmin, y_start+ymin), (x_start+xmax, y_start+ymax), (0, 0, 0), lineType)
            img_objects.append({"name":shape_type, "xmin": x_start+xmin, "ymin": y_start+ymin, "xmax": x_start+xmax, "ymax": y_start+ymax})
        elif shape_type == 'ellipse': # 椭圆
            centre_x, centre_y = random.randint(3*sw//7, sw//2), random.randint(3*sh//7, sh//2)
            axes_x, axes_y = random.randint(1*sw//6, 2*sw//5), random.randint(sh//8, sh//5)
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
            
        elif shape_type == 'diamond': #菱形
            centre_x, centre_y = random.randint(2*sw//5, 3*sw//5), random.randint(2*sh//5, 3*sh//5)
            # axes_x, axes_y = random.randint(min_w, min(centre_x, sw-centre_x)-4), random.randint(min_y, min(centre_y, sh-centre_y)-4)
            axes_x, axes_y = random.randint(sw//6, 2*sw//5), random.randint(sh//6, 2*sh//5)
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
            
        elif shape_type == 'parallel': # 平行四边形
            xmin, ymin = random.randint(0, sw//4), random.randint(0, sh//4)
            xmax, ymax = random.randint(sw//2+10, sw-10), random.randint(sh//2+10, sh//2+30)
            # xmax, ymax = xmin + 100, ymin+40
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
    all_base_graph_dataset = list()
    for i in tqdm(range(args['num']), desc="Datasets:"):
        i_w = random.randint(60, 100)*10
        i_h = random.randint(i_w//10-1, 100)*10
        max_item_num= np.random.choice([4, 9, 16, 25], 1, p=[0.1, 0.1, 0.5, 0.3])[0]
        img_objects = generate_img(i_w, i_h, max_item_num, args['images_savepath'], f"{i}.png")
        tmp_dataset = construct_json_obj(args['folder'], i, {"width":i_w, "height":i_h}, img_objects, f"{args['images_savepath']}/{i}.png")
        # all_base_graph_dataset.append({"id": f"graph_{i}", "conversations": conversations})
        all_base_graph_dataset.extend(tmp_dataset)
    
    with open(args['save_path'] + "/base_graph_data.json", 'w', encoding="utf-8") as f:
        json.dump(all_base_graph_dataset, f, ensure_ascii=False)
    

train_args = {
    'folder': 'train',
    'num': 5000,
    'images_savepath': VOC_img_savepath,
    'save_path': VOC_annots_savepath
}

if __name__ == '__main__':
    if not os.path.exists(VOC_img_savepath): os.makedirs(VOC_img_savepath)
    if not os.path.exists(VOC_annots_savepath): os.makedirs(VOC_annots_savepath)
    generate_datasets(train_args)
    
