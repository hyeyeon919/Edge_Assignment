import time
from ultralytics import YOLO
import os

results_file = "results_fp32.txt"

# COCO128 데이터셋 및 yaml 경로
dataset_path = "./coco128/images/train2017"
yaml_path = "./coco128.yaml"  # coco128.yaml 경로를 지정

# FP32 모델 처리
print("Exporting YOLOv8 TensorRT FP32 model...")
model_fp32 = YOLO("yolov8n.pt")
model_fp32.export(format="engine", dynamic=False, batch=8, workspace=4, data=yaml_path)

