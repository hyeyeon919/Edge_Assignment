import time
from ultralytics import YOLO
import os

results_file = "results.txt"

# COCO128 데이터셋 및 yaml 경로
dataset_path = "./coco128/images/train2017"
yaml_path = "./coco128.yaml"  # coco128.yaml 경로를 지정

def save_results_to_file(file, model_type, speed, accuracy):
    with open(file, "a") as f:
        f.write("Model Type: {}\n".format(model_type))
        f.write("Mean Inference Time: {:.4f} seconds per image\n".format(speed))
        f.write("Mean Accuracy: {:.4f}\n".format(accuracy))
        f.write("-" * 50 + "\n")

def calculate_model_metrics(model, dataset_path):
    total_accuracy = 0
    image_count = 0
    total_time = 0
    for image_file in os.listdir(dataset_path):
        if image_file.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(dataset_path, image_file)
            start_time = time.time()
            results = model.predict(image_path)
            elapsed_time = time.time() - start_time
            total_time += elapsed_time
            if len(results) > 0:
                total_accuracy += sum(r.probs.max().item() for r in results) / len(results)
            image_count += 1
    if image_count == 0:
        return 0, 0
    mean_accuracy = total_accuracy / image_count
    mean_speed = total_time / image_count
    return mean_accuracy, mean_speed

# FP32 모델 처리
print("Exporting YOLOv8 TensorRT FP32 model...")
model_fp32 = YOLO("yolov8n.pt")
fp32_engine_file = "yolov8n_fp32.engine"  # FP32 파일 이름 지정
model_fp32.export(format="engine", dynamic=False, batch=8, workspace=4, data=yaml_path, name=fp32_engine_file)

print("Loading TensorRT FP32 model...")
tensorrt_fp32 = YOLO(fp32_engine_file)

print("Calculating FP32 model metrics on COCO128...")
dataset_accuracy_fp32, dataset_speed_fp32 = calculate_model_metrics(tensorrt_fp32, dataset_path)
save_results_to_file(results_file, "TensorRT FP32 (COCO128)", dataset_speed_fp32, dataset_accuracy_fp32)

# FP16 모델 처리
print("Exporting YOLOv8 TensorRT FP16 model...")
fp16_engine_file = "yolov8n_fp16.engine"  # FP16 파일 이름 지정
model_fp32.export(format="engine", dynamic=False, batch=8, workspace=4, half=True, data=yaml_path, name=fp16_engine_file)

print("Loading TensorRT FP16 model...")
tensorrt_fp16 = YOLO(fp16_engine_file)

print("Calculating FP16 model metrics on COCO128...")
dataset_accuracy_fp16, dataset_speed_fp16 = calculate_model_metrics(tensorrt_fp16, dataset_path)
save_results_to_file(results_file, "TensorRT FP16 (COCO128)", dataset_speed_fp16, dataset_accuracy_fp16)

# INT8 모델 처리
print("Exporting YOLOv8 TensorRT INT8 model...")
int8_engine_file = "yolov8n_int8.engine"  # INT8 파일 이름 지정
model_fp32.export(format="engine", dynamic=False, batch=8, workspace=4, int8=True, data=yaml_path, name=int8_engine_file)

print("Loading TensorRT INT8 model...")
tensorrt_int8 = YOLO(int8_engine_file)

print("Calculating INT8 model metrics on COCO128...")
dataset_accuracy_int8, dataset_speed_int8 = calculate_model_metrics(tensorrt_int8, dataset_path)
save_results_to_file(results_file, "TensorRT INT8 (COCO128)", dataset_speed_int8, dataset_accuracy_int8)

print("Comparison complete. Results saved to", results_file)
