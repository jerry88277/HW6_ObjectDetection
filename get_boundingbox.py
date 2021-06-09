# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 10:44:33 2021

@author: JerryDai
"""
import cv2
import darknet
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

# In[]

def bbox2point(bbox):
    '''
    reverse box format into point 
    box format: x, y, w, h
    point: xmin, xmax, ymin, ymax
    '''
    x, y, w, h = bbox
    xmin = x - (w / 2)
    xmax = x + (w / 2)
    ymin = y - (h / 2)
    ymax = y + (h / 2)
    return (xmin, ymin, xmax, ymax)

def point2bbox(point):
    '''
    change point into box format
    box format: x, y, w, h
    point: xmin, xmax, ymin, ymax
    '''
    x1,y1,x2,y2 = point
    x = (x1+x2)/2
    y = (y1+y2)/2
    w = (x2-x1)
    h = (y2-y1)
    return (x,y,w,h)

# In[]

win_title = 'YOLOv4 CUSTOM DETECTOR'
test_data_path = 'cfg/total.txt'

cfg_file = 'cfg/yolov4-tiny-obj_test.cfg'
data_file = 'cfg/train.data'
weight_file = 'cfg/weights/yolov4-tiny-obj_test_last_loss1.6.weights'
thre = 0.20
show_coordinates = True

network, class_names, class_colors = darknet.load_network(cfg_file,
                                                          data_file,
                                                          weight_file,
                                                          batch_size = 1 
                                                          )

width = darknet.network_width(network)
height = darknet.network_height(network)

# In[]
voc_labels = ('aquarium', 'bottle', 'bowl', 'box', 'bucket', 'plastic_bag', 'plate', 'styrofoam', 'tire', 'toilet', 'tub', 'washing_machine', 'water_tower')
all_classes = {k: v + 1 for v, k in enumerate(voc_labels)}

submission = pd.DataFrame([], columns = ['image_filename', 'label_id', 'x', 'y', 'w', 'h', 'confidence'])

test_image = pd.read_csv(test_data_path, header=None)[0]

for i_image_path in tqdm(test_image):
    
    #i_image_path = r'D:\NCKU\Class In NCKU\DeepLearning\HW6\Object_detection\data\yolo_total\yolo_total0.jpg'
    
    # get image name
    image_name = i_image_path.split('\\')[-1]
    i_image = cv2.imread(i_image_path)
    
    # judge image channel
    if len(i_image.shape) == 2:
        i_image = np.stack([i_image]*3, axis=-1)
    # get original image size
    orig_h, orig_w = i_image.shape[:2]
    
    # get net width & height
    width = darknet.network_width(network)
    height = darknet.network_height(network)

    # change image channel order
    image_rgb = cv2.cvtColor(i_image, cv2.COLOR_BGR2RGB)
    
    # resize to darknet image size
    image_resized = cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_LINEAR)


    darknet_image = darknet.make_image(width, height, 3)
    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())

    
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thre)
    darknet.print_detections(detections, show_coordinates)
    darknet.free_image(darknet_image)
    
    new_detections = []
    for detection in detections:
        pred_label, pred_conf, (x, y, w, h) = detection
        new_x = x / width * orig_w
        new_y = y / height * orig_h
        new_w = w / width * orig_w
        new_h = h / height * orig_h
 
        # reverse 
        (x1,y1,x2,y2) = bbox2point((new_x, new_y, new_w, new_h))
        x1 = x1 if x1 > 0 else 0
        x2 = x2 if x2 < orig_w else orig_w
        y1 = y1 if y1 > 0 else 0
        y2 = y2 if y2 < orig_h else orig_h
 
        (new_x, new_y, new_w, new_h) = point2bbox((x1,y1,x2,y2))
 
        new_detections.append((pred_label, pred_conf, (new_x, new_y, new_w, new_h)))
        
        pred_label_index = all_classes.get(pred_label)
        submission = submission.append(pd.DataFrame([[image_name, pred_label_index, new_x, new_y, new_w, new_h, float(pred_conf) / 100]], columns = ['image_filename', 'label_id', 'x', 'y', 'w', 'h', 'confidence']))
    image = darknet.draw_boxes(new_detections, i_image, class_colors)
    
    # cv2.imshow(image_name, image)
    
    cv2.imwrite(os.path.join('train_image_box', image_name), image)
    
    
# In[]

# submission.to_csv('submission_yolov4_loss1.6.csv', index = False)





