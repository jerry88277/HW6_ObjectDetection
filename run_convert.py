# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 01:39:14 2021

@author: JerryDai
"""
import os
import shutil
from bs4 import BeautifulSoup
# In[]

def point2bbox(point, img_w, img_h):
    xmin, ymin, xmax, ymax = point
    x = (xmin + (xmax-xmin)/2) * 1.0 / img_w
    y = (ymin + (ymax-ymin)/2) * 1.0 / img_h
    w = (xmax-xmin) * 1.0 / img_w
    h = (ymax-ymin) * 1.0 / img_h
    return x, y, w, h

def run_convert(all_classes, train_img, train_annotation, yolo_path, write_txt):
    now_path = os.getcwd()
    data_counter = 0

    for data_file in os.listdir(train_annotation):
        try:
            with open(os.path.join(train_annotation, data_file), 'r') as f:
                print("read file...")
                soup = BeautifulSoup(f.read(), 'html.parser')
                img_name = soup.select_one('filename').text

                for size in soup.select('size'):
                    img_w = int(size.select_one('width').text)
                    img_h = int(size.select_one('height').text)
                    
                img_info = []
                for obj in soup.select('object'):
                    xmin = int(obj.select_one('xmin').text)
                    xmax = int(obj.select_one('xmax').text)
                    ymin = int(obj.select_one('ymin').text)
                    ymax = int(obj.select_one('ymax').text)
                    objclass = all_classes.get(obj.select_one('name').text)

                    # x = (xmin + (xmax-xmin)/2) * 1.0 / img_w
                    # y = (ymin + (ymax-ymin)/2) * 1.0 / img_h
                    # w = (xmax-xmin) * 1.0 / img_w
                    # h = (ymax-ymin) * 1.0 / img_h
                    x, y, w, h = point2bbox((xmin, ymin, xmax, ymax), img_w, img_h)
                    
                    img_info.append(' '.join([str(objclass), str(x),str(y),str(w),str(h)]))

                # copy image to yolo path and rename
                img_path = os.path.join(train_img, img_name)
                img_format = img_name.split('.')[1]  # jpg or png
                shutil.copyfile(img_path, yolo_path + str(data_counter) + '.' + img_format)
                
                # create yolo bndbox txt
                with open(yolo_path + str(data_counter) + '.txt', 'a+') as f:
                    f.write('\n'.join(img_info))

                # create train or val txt
                with open(write_txt, 'a') as f:
                    path = os.path.join(now_path, yolo_path)
                    line_txt = [path + str(data_counter) + '.' + img_format, '\n']
                    f.writelines(line_txt)

                data_counter += 1
                    
        except Exception as e:
            print(e)
           
    print('the file is processed')

def run_convert_forTest(test_img, write_txt):
    now_path = os.getcwd()

    for data_file in os.listdir(test_img):
        try:
            # copy image to yolo path and rename
            img_path = os.path.join(test_img, data_file)
            img_format = data_file.split('.')[1]  # jpg or png

            # create train or val txt
            with open(write_txt, 'a') as f:
                line_txt = [img_path, '\n']
                f.writelines(line_txt)

                    
        except Exception as e:
            print(e)
           
    print('the file is processed')

# In[]

# all_classes = {'class_2': 2, 'class_1': 1, 'class_0': 0}
voc_labels = ('aquarium', 'bottle', 'bowl', 'box', 'bucket', 'plastic_bag', 'plate', 'styrofoam', 'tire', 'toilet', 'tub', 'washing_machine', 'water_tower')
all_classes = {k: v for v, k in enumerate(voc_labels)}
# all_classes['background'] = 0

train_img = r'D:\NCKU\Class In NCKU\DeepLearning\HW6\Object_detection\data\train_cdc\total_images'
train_annotation = r'D:\NCKU\Class In NCKU\DeepLearning\HW6\Object_detection\data\train_cdc\total_annotations'
yolo_path = r'D:\NCKU\Class In NCKU\DeepLearning\HW6\Object_detection\data\yolo_total\yolo_total'
write_txt = 'cfg/total.txt'

if not os.path.exists(yolo_path):
    os.mkdir(yolo_path)
else:
    lsdir = os.listdir(yolo_path)
    for name in lsdir:
        if name.endswith('.txt') or name.endswith('.jpg') or name.endswith('.png'):
            os.remove(os.path.join(yolo_path, name))

cfg_file = write_txt.split('/')[0]
if not os.path.exists(cfg_file):
    os.mkdir(cfg_file)
    
if os.path.exists(write_txt):
    file=open(write_txt, 'w')

run_convert(all_classes, train_img, train_annotation, yolo_path, write_txt)

# In[]

# test_img = r'D:\NCKU\Class In NCKU\DeepLearning\HW6\Object_detection\data\test_cdc\test_images'
# write_txt = 'cfg/test.txt'
# run_convert_forTest(test_img, write_txt)

# In[]    
# images_path = r'D:\NCKU\Class In NCKU\DeepLearning\HW6\Object_detection\data\yolo_total'
# total = []
# for file in os.listdir(images_path):
#     if file.endswith(".jpg"):
#         total.append(os.path.join(images_path, file))

# with open('total.txt', 'w') as f:
#     for item in total:
#         f.write("%s\n" % item)