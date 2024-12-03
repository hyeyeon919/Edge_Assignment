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
    batch_size = 8
    input_images = []  # 배치 생성용 리스트

    for image_file in os.listdir(dataset_path):
        if image_file.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(dataset_path, image_file)
            image = preprocess_image(image_path)  # 이미지 전처리 함수
            input_images.append(image)

            # 배치 처리
            if len(input_images) == batch_size:
                input_tensor = torch.stack(input_images)  # 배치 생성
                start_time = time.time()
                results = model(input_tensor)
                elapsed_time = time.time() - start_time
                total_time += elapsed_time

                # 결과 처리
                total_accuracy += calculate_batch_accuracy(results)
                image_count += len(input_images)
                input_images = []  # 배치 초기화

    # 마지막 남은 배치 처리
    if input_images:
        input_tensor = torch.stack(input_images)
        start_time = time.time()
        results = model(input_tensor)
        elapsed_time = time.time() - start_time
        total_time += elapsed_time
        total_accuracy += calculate_batch_accuracy(results)
        image_count += len(input_images)

    mean_accuracy = total_accuracy / image_count
    mean_speed = total_time / image_count
    return mean_accuracy, mean_speed

print("Exporting YOLOv8 TensorRT FP32 model...")
model_fp32 = YOLO("yolov8n.pt")
# model_fp32.export(format="engine", dynamic=False, batch=8, workspace=4, data=yaml_path)

print("Loading TensorRT FP32 model...")
tensorrt_fp32 = YOLO("yolov8n.engine")

print("Calculating FP32 model metrics on COCO128...")
dataset_accuracy_fp32, dataset_speed_fp32 = calculate_model_metrics(tensorrt_fp32, dataset_path)
save_results_to_file(results_file, "TensorRT FP32 (COCO128)", dataset_speed_fp32, dataset_accuracy_fp32)

print("Exporting YOLOv8 TensorRT FP16 model...")
model_fp32.export(format="engine", dynamic=False, batch=8, workspace=4, half=True, data=yaml_path)

print("Loading TensorRT FP16 model...")
tensorrt_fp16 = YOLO("yolov8n_fp16.engine")

print("Calculating FP16 model metrics on COCO128...")
dataset_accuracy_fp16, dataset_speed_fp16 = calculate_model_metrics(tensorrt_fp16, dataset_path)
save_results_to_file(results_file, "TensorRT FP16 (COCO128)", dataset_speed_fp16, dataset_accuracy_fp16)

print("Exporting YOLOv8 TensorRT INT8 model...")
model_fp32.export(format="engine", dynamic=False, batch=8, workspace=4, int8=True, data=yaml_path)

print("Loading TensorRT INT8 model...")
tensorrt_int8 = YOLO("yolov8nint8.engine")

print("Calculating INT8 model metrics on COCO128...")
dataset_accuracy_int8, dataset_speed_int8 = calculate_model_metrics(tensorrt_int8, dataset_path)
save_results_to_file(results_file, "TensorRT INT8 (COCO128)", dataset_speed_int8, dataset_accuracy_int8)

print("Comparison complete. Results saved to", results_file)
