import time
from ultralytics import YOLO
import os

# 결과를 저장할 txt 파일
results_file = "results.txt"

# 결과 저장 함수
def save_results_to_file(file, model_type, speed, accuracy):
    with open(file, "a") as f:
        f.write(f"Model Type: {model_type}\n")
        f.write(f"Mean Inference Time: {speed:.4f} seconds per image\n")
        f.write(f"Mean Accuracy: {accuracy:.4f}\n")
        f.write("-" * 50 + "\n")

# 정확도 및 추론 시간 계산 함수
def calculate_model_metrics(model, dataset_path):
    total_accuracy = 0
    image_count = 0
    total_time = 0
    # 테스트 데이터셋의 모든 이미지 파일 가져오기
    for image_file in os.listdir(dataset_path):
        if image_file.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(dataset_path, image_file)
            start_time = time.time()  # 각 이미지 추론 시작 시간
            results = model.predict(image_path)  # 추론 수행
            elapsed_time = time.time() - start_time
            total_time += elapsed_time
            # 검출된 객체의 가장 높은 확률 값을 누적
            if len(results) > 0:
                total_accuracy += sum(r.probs.max().item() for r in results) / len(results)
            image_count += 1

    # 평균 정확도와 평균 속도 계산
    if image_count == 0:
        return 0, 0  # 이미지가 없을 경우 0 반환
    mean_accuracy = total_accuracy / image_count
    mean_speed = total_time / image_count
    return mean_accuracy, mean_speed

# COCO128 데이터셋 경로 (ultralytics 라이브러리가 자동으로 다운로드)
dataset_path = YOLO("yolov8n.pt").data.get("val")  # COCO128의 val 데이터 경로

# 1. FP32 모델
print("Exporting YOLOv8 TensorRT FP32 model...")
model_fp32 = YOLO("yolov8n.pt")
model_fp32.export(
    format="engine",
    dynamic=True,
    batch=8,
    workspace=4,
    data="coco128.yaml"
)

print("Loading TensorRT FP32 model...")
tensorrt_fp32 = YOLO("yolov8n.engine")

# COCO128 데이터셋 추론 (FP32)
print("Calculating FP32 model metrics on COCO128...")
dataset_accuracy_fp32, dataset_speed_fp32 = calculate_model_metrics(tensorrt_fp32, dataset_path)

# 결과 저장 (FP32)
save_results_to_file(results_file, "TensorRT FP32 (COCO128)", dataset_speed_fp32, dataset_accuracy_fp32)

# 2. FP16 모델
print("Exporting YOLOv8 TensorRT FP16 model...")
model_fp32.export(
    format="engine",
    dynamic=True,
    batch=8,
    workspace=4,
    half=True,  # FP16 활성화
    data="coco128.yaml"
)

print("Loading TensorRT FP16 model...")
tensorrt_fp16 = YOLO("yolov8n_fp16.engine")

# COCO128 데이터셋 추론 (FP16)
print("Calculating FP16 model metrics on COCO128...")
dataset_accuracy_fp16, dataset_speed_fp16 = calculate_model_metrics(tensorrt_fp16, dataset_path)

# 결과 저장 (FP16)
save_results_to_file(results_file, "TensorRT FP16 (COCO128)", dataset_speed_fp16, dataset_accuracy_fp16)

# 3. INT8 모델
print("Exporting YOLOv8 TensorRT INT8 model...")
model_fp32.export(
    format="engine",
    dynamic=True,
    batch=8,
    workspace=4,
    int8=True,  # INT8 활성화
    data="coco128.yaml"
)

print("Loading TensorRT INT8 model...")
tensorrt_int8 = YOLO("yolov8nint8.engine")

# COCO128 데이터셋 추론 (INT8)
print("Calculating INT8 model metrics on COCO128...")
dataset_accuracy_int8, dataset_speed_int8 = calculate_model_metrics(tensorrt_int8, dataset_path)

# 결과 저장 (INT8)
save_results_to_file(results_file, "TensorRT INT8 (COCO128)", dataset_speed_int8, dataset_accuracy_int8)

print("Comparison complete. Results saved to", results_file)
