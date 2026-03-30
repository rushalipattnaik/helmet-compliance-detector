from ultralytics import YOLO

def train():
    model = YOLO("yolov8s.pt")

    results = model.train(
        data="data/dataset/data.yaml",
        epochs=30,
        imgsz=640,
        batch=8,
        name="helmet_detector",
        project="runs",
        patience=10,
        save=True,
        plots=True,
        verbose=True,
    )
    print("✅ Training complete!")
    print("Best model saved at: runs/helmet_detector/weights/best.pt")

if __name__ == "__main__":
    train()