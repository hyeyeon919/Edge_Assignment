import time
from ultralytics import YOLO
import os

results_file = "results_fp32.txt"

# COCO128 데이터셋 및 yaml 경로
dataset_path = "./coco128/images/train2017"
yaml_path = "./coco128.yaml"  # coco128.yaml 경로를 지정

def save_results_to_file(file, model_type, speed, accuracy):
    with open(file, "a") as f:
        f.write("Model Type: {}\n".format(model_type))
        f.write("Mean Inference Time: {:.4f} seconds per image\n".format(speed))
        f.write("Mean Accuracy: {:.4f}\n".format(accuracy))
        f.write("-" * 50 + "\n")

def calculate_model_metrics(model, dataset_path, batch_size=8):
    total_accuracy = 0
    total_time = 0
    image_count = 0
    input_images = []  # 배치 데이터를 저장할 리스트

    for image_file in os.listdir(dataset_path):
        if image_file.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(dataset_path, image_file)

            # 이미지 로드 및 모델에 맞게 처리 (여기선 모델이 자동 처리)
            input_images.append(image_path)

            # 배치 크기에 도달하면 처리
            if len(input_images) == batch_size:
                start_time = time.time()
                results = model.predict(source=input_images)  # 배치 처리
                elapsed_time = time.time() - start_time
                total_time += elapsed_time

                # 결과에서 정확도 계산
                total_accuracy += sum(1.0 for _ in results)  # 예시로 더미 정확도 사용
                image_count += len(input_images)
                input_images = []  # 배치 초기화

    # 마지막 남은 이미지를 처리
    if input_images:
        start_time = time.time()
        results = model.predict(source=input_images)
        elapsed_time = time.time() - start_time
        total_time += elapsed_time

        total_accuracy += sum(1.0 for _ in results)  # 예시로 더미 정확도 사용
        image_count += len(input_images)

    # 평균 속도 및 정확도 계산
    mean_speed = total_time / image_count
    mean_accuracy = total_accuracy / image_count
    return mean_accuracy, mean_speed

# FP32 모델 처리
print("Exporting YOLOv8 TensorRT FP32 model...")
model_fp32 = YOLO("yolov8n.pt")
model_fp32.export(format="engine", dynamic=False, batch=8, workspace=4, data=yaml_path)

print("Loading TensorRT FP32 model...")
tensorrt_fp32 = YOLO("yolov8n.engine")

print("Calculating FP32 model metrics on COCO128...")
dataset_accuracy_fp32, dataset_speed_fp32 = calculate_model_metrics(tensorrt_fp32, dataset_path)
save_results_to_file(results_file, "TensorRT FP32 (COCO128)", dataset_speed_fp32, dataset_accuracy_fp32)

print("Comparison complete. Results saved to", results_file)
