import time
from ultralytics import YOLO
import os

results_file = "results_fp16.txt"

# COCO128 데이터셋 및 yaml 경로
dataset_path = "./coco128/images/train2017"
yaml_path = "./coco128.yaml"  # coco128.yaml 경로를 지정

# FP32 모델 처리
print("Exporting YOLOv8 TensorRT FP32 model...")
model_fp32 = YOLO("yolov8n.pt")
model_fp32.export(format="engine", dynamic=False, batch=8, workspace=4, half=True, data=yaml_path)

print("Loading TensorRT FP16 model...")
tensorrt_fp16 = YOLO("yolov8n.engine")

print("Calculating FP16 model metrics on COCO128...")
dataset_accuracy_fp16, dataset_speed_fp16 = calculate_model_metrics(tensorrt_fp16, dataset_path)
save_results_to_file(results_file, "TensorRT FP16 (COCO128)", dataset_speed_fp16, dataset_accuracy_fp16)

print("Comparison complete. Results saved to", results_file)
