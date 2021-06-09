# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 21:21:55 2021

@author: JerryDai
"""
import os
from PIL import Image, ImageDraw
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import pandas as pd
from tqdm import tqdm

# In[] def 

def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a

def bbox2point(bbox, img_w, img_h):
    '''
    reverse normalization box format from txt
    normalization: division with image width & height
    box format: x, y, w, h
    point: xmin, xmax, ymin, ymax
    '''
    x, y, w, h = bbox
    xmin = (2 * x - w) * img_w /2
    xmax = (2 * x + w) * img_w /2
    ymin = (2 * y - h) * img_h /2
    ymax = (2 * y + h) * img_h /2
    return xmin, ymin, xmax, ymax 

def point2bbox(point, img_w, img_h):
    '''
    normalize point into yolo normalization box format
    normalization: division with image width & height
    box format: x, y, w, h
    point: xmin, xmax, ymin, ymax
    '''
    xmin, ymin, xmax, ymax = point
    x = (xmin + (xmax-xmin)/2) * 1.0 / img_w
    y = (ymin + (ymax-ymin)/2) * 1.0 / img_h
    w = (xmax-xmin) * 1.0 / img_w
    h = (ymax-ymin) * 1.0 / img_h
    return x, y, w, h

def get_place_position(position_index, new_img_w, new_img_h, model_w, model_h):
    
    position_x = 0
    position_y = 0
    
    if position_index == 0:
        # 左上
        position_x = 0
        position_y = 0
        
    elif position_index == 1:
        # 左下    
        position_x = 0
        position_y = model_h - new_img_h
        
    elif position_index == 2:
        # 右下
        position_x = model_w - new_img_w
        position_y = model_h - new_img_h
        
    elif position_index == 3:
        # 右上
        position_x = model_w - new_img_w
        position_y = 0
    
    return position_x, position_y

def cut_image(position_index, new_image, image_data, cutx, cuty):
    
    if position_index == 0:
        # 左上
        new_image[:cuty, :cutx, :] = image_data[:cuty, :cutx, :]
    elif position_index == 1:
        # 左下    
        new_image[cuty:, :cutx, :] = image_data[cuty:, :cutx, :]
    elif position_index == 2:
        # 右下
        new_image[cuty:, cutx:, :] = image_data[cuty:, cutx:, :]
    elif position_index == 3:
        # 右上
        new_image[:cuty, cutx:, :] = image_data[:cuty, cutx:, :]
    
    return new_image
    
    
def change_HSV(image, hue=.1, sat=1.5, val=1.5):
    
    # 進行色域變換
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand() < 0.5 else 1 / rand(1, sat)
    val = rand(1, val) if rand() < 0.5 else 1 / rand(1, val)
    x = rgb_to_hsv(np.array(image)/255.0)
    
    ## Hue
    x[..., 0] += hue
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    
    ## Saturation
    x[..., 1] *= sat
    
    ## Value
    x[..., 2] *= val
    
    x[x > 1] = 1
    x[x < 0] = 0
    
    image = hsv_to_rgb(x)
    image = Image.fromarray((image * 255).astype(np.uint8))
    
    return image

def resize_box(boxes, new_img_w, new_img_h, model_w, model_h, dx, dy):
    
    if len(boxes) > 0:
        ## 對xmin, xmax進行縮放
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * new_img_w / model_w + dx
        ## 對ymin, ymax進行縮放
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * new_img_h / model_h + dy
        ## 對xmin, ymin確認邊界
        boxes[:, 0:2][boxes[:, 0:2] < 0] = 0
        ## 對xmax確認邊界
        boxes[:, 2][boxes[:, 2] > model_w] = model_w
        ## 對ymax確認邊界
        boxes[:, 3][boxes[:, 3] > model_h] = model_h
        ## 計算框寬
        boxes_w = boxes[:, 2] - boxes[:, 0]
        ## 計算框高
        boxes_h = boxes[:, 3] - boxes[:, 1]
        ##
        boxes = boxes[np.logical_and(boxes_w > 1, boxes_h > 1)]
    
    return boxes

def check_box_with_cutpoint(position_index, boxes, cutx, cuty, labels): # boxes = i_boxes.copy()  # labels = i_labels.copy()
    
    new_boxes = np.zeros([1, 4])
    new_labels = []
    
    if position_index == 0:
        # 左上:  xlimit = [0, cutx] ylimit = [0, cuty]
        for index, i_box in enumerate(boxes):
            xmin, ymin, xmax, ymax = i_box
            i_box = np.array([i_box])
            
            if xmin > cutx or ymin > cuty: # box 高寬任一出界
                continue
            
            if xmax < cutx and ymax < cuty: # box 完全界內
                new_boxes = np.append(new_boxes, i_box, axis = 0)
                new_labels.append(labels[index])
                continue
            
            if xmin < cutx <= xmax: # box之寬跨界
                i_box[:, 2] = cutx
                            
            if ymin < cuty <= ymax: # box之高跨界
                i_box[:, 3] = cuty
                
            new_boxes = np.append(new_boxes, i_box, axis = 0)
            new_labels.append(labels[index])
            
    elif position_index == 1:
        # 左下 xlimit = [0, cutx] ylimit = [cuty, model_h]
        for index, i_box in enumerate(boxes):
            xmin, ymin, xmax, ymax = i_box
            i_box = np.array([i_box])
            
            if xmin > cutx or ymax < cuty: # box 高寬任一出界
                continue
            
            if xmax < cutx and ymin > cuty: # box 完全界內
                new_boxes = np.append(new_boxes, i_box, axis = 0)
                new_labels.append(labels[index])
                continue
            
            if xmin < cutx <= xmax: # box之寬跨界
                i_box[:, 2] = cutx
                            
            if ymin < cuty <= ymax: # box之高跨界
                i_box[:, 1] = cuty
            
            new_boxes = np.append(new_boxes, i_box, axis = 0)
            new_labels.append(labels[index])    
            
        
    elif position_index == 2:
        # 右下 xlimit = [cutx, model_w] ylimit = [cuty, model_h]
        for index, i_box in enumerate(boxes):
            xmin, ymin, xmax, ymax = i_box
            i_box = np.array([i_box])
            
            if xmax < cutx or ymax < cuty: # box 高寬任一出界
                continue
            
            if xmin > cutx and ymin > cuty: # box 完全界內
                new_boxes = np.append(new_boxes, i_box, axis = 0)
                new_labels.append(labels[index])
                continue
            
            if xmin < cutx <= xmax: # box之寬跨界
                i_box[:, 0] = cutx
                            
            if ymin < cuty <= ymax: # box之高跨界
                i_box[:, 1] = cuty
                
            new_boxes = np.append(new_boxes, i_box, axis = 0)
            new_labels.append(labels[index])    
            
        
    elif position_index == 3:
        # 右上 xlimit = [cutx, model_w] ylimit = [0, cuty]
        for index, i_box in enumerate(boxes):
            xmin, ymin, xmax, ymax = i_box
            i_box = np.array([i_box])
            
            if xmax < cutx or ymin > cuty: # box 高寬任一出界
                continue
            
            if xmin > cutx and ymax < cuty: # box 完全界內
                new_boxes = np.append(new_boxes, i_box, axis = 0)
                new_labels.append(labels[index])
                continue
            
            if xmin < cutx <= xmax: # box之寬跨界
                i_box[:, 0] = cutx
                            
            if ymin < cuty <= ymax: # box之高跨界
                i_box[:, 3] = cuty
                
            new_boxes = np.append(new_boxes, i_box, axis = 0)
            new_labels.append(labels[index])    
    
    new_boxes = np.delete(new_boxes, 0, axis = 0)
    
    return new_boxes, new_labels

def get_mosaic_data(mosaic_path):
    
    imgaes_path = []
    for file in os.listdir(mosaic_path):
        if file.endswith(".jpg"):
            imgaes_path.append(os.path.join(mosaic_path, file))
    
    annotations_path = []
    for file in os.listdir(mosaic_path):
        if file.endswith(".txt"):
            annotations_path.append(os.path.join(mosaic_path, file))
    
    # yolo model input size
    model_w, model_h = [416, 416]
    
    # 縮放限制
    min_offset_x = 0.4
    min_offset_y = 0.4
    scale_low = 1 - min(min_offset_x, min_offset_y)
    scale_high = scale_low + 0.2

    # 圖片合併起點位置：1左上 2左下 3右下 4右上
    # place_x = [0, 0, int(model_w * min_offset_x), int(model_w * min_offset_x)]
    # place_y = [0, int(model_h * min_offset_y), int(model_h * min_offset_y), 0]
    
    # 圖片合併起點位置指標
    position_index = 0
    
    # 合併圖片切割點    
    cutx = np.random.randint(int(model_w * min_offset_x), int(model_w * (1 - min_offset_x)))
    cuty = np.random.randint(int(model_h * min_offset_y), int(model_h * (1 - min_offset_y)))
    
    # 取4個隨機亂數
    random_index = np.random.randint(len(imgaes_path), size = 4)
    
    # 建立灰底空白圖片
    new_image = np.zeros([model_w, model_h, 3])
    
    # 新圖片的框資訊
    new_boxes = np.zeros([1,4])
    new_labels = []
    
    for i_index in random_index:
        
        # 讀取圖片
        image = Image.open(imgaes_path[i_index])
        image = image.convert("RGB") 
        
        # 圖片大小
        img_w, img_h = image.size
        
        # 讀取框的資訊
        
        i_labels = []
        i_boxes = np.zeros([1,4])
        with open(annotations_path[i_index], 'r') as f:
            if f:
                for item in f.readlines():
                    
                    label, x, y, w, h = item.split(' ')
                    i_labels.append(label)
                    
                    box_i = float(x), float(y), float(w), float(h)
                    point_i = np.array([bbox2point(box_i, img_w, img_h)])
                    # i_boxes.append(list(point_i))
                    i_boxes = np.append(i_boxes, point_i, axis = 0)
        
        i_boxes = np.delete(i_boxes, 0, axis = 0)
        
        # print(f'box_number:{len(i_boxes)}')
        # print(f'label_number:{len(i_labels)}\n')
        
        ####### check
        # image2 = image.copy()
        # for j in range(len(i_boxes)):
        #     thickness = 3
        #     left, top, right, bottom  = i_boxes[j]
        #     draw = ImageDraw.Draw(image2)
        #     for i in range(thickness):
        #         draw.rectangle([left + i, top + i, right - i, bottom - i], outline = (255, 255, 255))
        # image2.show()
        
        # 以隨機亂數決定是否水平翻轉圖片
        flip = rand() < 0.5
        if flip and len(i_boxes) > 0:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            i_boxes[:, [0,2]] = img_w - i_boxes[:, [2,0]]
        
        ####### check
        # image3 = image.copy()
        # for j in range(len(i_boxes)):
        #     thickness = 3
        #     left, top, right, bottom  = i_boxes[j]
        #     draw = ImageDraw.Draw(image3)
        #     for i in range(thickness):
        #         draw.rectangle([left + i, top + i, right - i, bottom - i], outline = (255, 255, 255))
        # image3.show()
        
        
        # 根據模型圖片框架大小縮放目標圖片
        new_ar = model_w / model_h
        scale = rand(scale_low, scale_high)
        if new_ar < 1:
            new_img_h = int(scale * model_h)
            new_img_w = int(new_img_h * new_ar)
        else:
            new_img_w = int(scale * model_w)
            new_img_h = int(new_img_w / new_ar)
        image = image.resize((new_img_w, new_img_h), Image.BICUBIC)

        # 進行色域變換
        image = change_HSV(image)
        
        # 以起始位置放置圖片
        position_x, position_y = get_place_position(position_index, new_img_w, new_img_h, model_w, model_h)
        tmp_image = Image.new('RGB', (model_w, model_h), (128, 128, 128))
        tmp_image.paste(image, (position_x, position_y))
        
        # 對box進行縮放
        i_boxes = resize_box(i_boxes, new_img_w, new_img_h, model_w, model_h, position_x, position_y)
        
        ####### check
        # image4 = tmp_image.copy()
        # for j in range(len(i_boxes)):
        #     thickness = 2
        #     left, top, right, bottom  = i_boxes[j]
        #     draw = ImageDraw.Draw(image4)
        #     for i in range(thickness):
        #         draw.rectangle([left + i, top + i, right + i, bottom + i], outline = (255, 255, 255))
        # image4.show()
        
        # 根據切割點合併圖片
        image_data = np.array(tmp_image)
        new_image = cut_image(position_index, new_image, image_data, cutx, cuty)
        
        ####### check
        # image5 = Image.fromarray(new_image.astype(np.uint8), mode = 'RGB')
        # image5.show()
        
        # 根據切割點確認框的資訊
        new_i_boxes, new_i_labels = check_box_with_cutpoint(position_index, i_boxes, cutx, cuty, i_labels)
        # print(f'box_number:{len(new_i_boxes)}')
        # print(f'label_number:{len(new_i_labels)}\n')
        
        ####### check
        # image6 = Image.fromarray(new_image.astype(np.uint8), mode = 'RGB')
        # for j in range(len(new_i_boxes)):
        #     thickness = 2
        #     left, top, right, bottom  = new_i_boxes[j]
        #     draw = ImageDraw.Draw(image6)
        #     for i in range(thickness):
        #         draw.rectangle([left + i, top + i, right + i, bottom + i], outline = (255, 255, 255))
        # image6.show()
        
        new_boxes = np.append(new_boxes, new_i_boxes, axis = 0)
        new_labels += new_i_labels
        
        position_index += 1
        
    new_boxes = np.delete(new_boxes, 0, axis = 0)
    
    # print(f'box_number:{len(new_boxes)}')
    # print(f'label_number:{len(new_labels)}\n')
    
    ####### check
    # image7 = Image.fromarray(new_image.astype(np.uint8), mode = 'RGB')
    # for j in range(len(new_boxes)):
    #     thickness = 2
    #     left, top, right, bottom  = new_boxes[j]
    #     draw = ImageDraw.Draw(image7)
    #     for i in range(thickness):
    #         draw.rectangle([left + i, top + i, right + i, bottom + i], outline = (255, 255, 255))
    # image7.show()
    
    
    label_df = pd.DataFrame(new_labels)
    
    for index, i_box in enumerate(new_boxes):
        new_boxes[index] = point2bbox(new_boxes[index], model_w, model_h)
        
    box_df = pd.DataFrame(new_boxes)
    
    new_annotation = pd.concat([label_df, box_df], axis = 1)
    new_image = Image.fromarray(new_image.astype(np.uint8), mode = 'RGB')
    
    return new_image, new_annotation



# In[]
mosaic_path = r'D:\NCKU\Class In NCKU\DeepLearning\HW6\Object_detection\data\yolo_total'
mosaic_save_path = r'D:\NCKU\Class In NCKU\DeepLearning\HW6\Object_detection\data\yolo_mosaic'

mosaic_number = 5000
for i in tqdm(range(mosaic_number)):
    new_image, new_annotation = get_mosaic_data(mosaic_path)
    new_image.save(os.path.join(mosaic_save_path, f'yolo_mosaic{i}.jpg'))
    new_annotation.to_csv(os.path.join(mosaic_save_path, f'yolo_mosaic{i}.txt'), sep = ' ', index = False, header = False)

        
