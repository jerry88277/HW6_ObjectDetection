# HW6_ObjectDetection
yolo套件安裝以及模型訓練方式請參考以下網址

yolo套件安裝、模型訓練：https://github.com/AlexeyAB/darknet
訓練範例：https://medium.com/ching-i/yolo-c49f70241aa7

相關套件安裝完畢後，在darknet.exe所在資料夾開啟cmd，輸入以下指令開始訓練模型

darknet.exe detector train ./cfg/train.data ./cfg/yolov4-tiny-obj_test.cfg ./cfg/weights/yolov4-tiny-obj_10000_loss4.weights -clear -map

## data_path.7z
圖片路徑
## train.data
模型資料輸入設定

## run_convert.py
將原始資料整理成yoloV4模型所需格式
## data_Mosaic.py
在原始資料中，以隨機亂數挑選4張圖片進行合併，增加圖片多樣性
## get_boundingbox.py
模型對圖片進行物件偵測，繪製圖片並輸出預測表
## yolov4-tiny-obj_test.cfg
yoloV4模型訓練設定
## yolov4-tiny-obj_10000_loss4.weights
yoloV4模型訓練權重
