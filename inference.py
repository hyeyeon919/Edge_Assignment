from ultralytics import YOLO

def main():
    # 모델 경로
    models = {
        "FP32": "yolov8n_fp32.engine",
        "FP16": "yolov8n_fp16.engine",
        "INT8": "yolov8n_int8.engine",
    }
    # COCO 데이터셋 구성 파일 경로 (coco.yaml)
    coco_yaml_path = "coco8.yaml"

    # 결과 저장
    result_file = "inference_results.txt"
    with open(result_file, "w") as f:
        for model_name, model_path in models.items():
            print(f"Evaluating {model_name} model...")

            # YOLO 모델 로드
            model = YOLO(model_path)

            # 검증 수행
            results = model.val(data=coco_yaml_path, imgsz=640, batch=16)

            # 결과 저장
            f.write(f"Model: {model_name}\n")
            f.write(f"mAP@0.5: {results.box.map50:.4f}\n")
            f.write(f"mAP@0.5:0.95: {results.box.map:.4f}\n")
            f.write(f"Average Inference Time per Image: {results.speed[1]:.4f} ms\n")
            f.write("\n")

if __name__ == "__main__":
    main()
