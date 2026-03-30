from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

MODEL_PATH = "runs/detect/runs/helmet_detector/weights/best.pt"
DATA_YAML  = "data/dataset/data.yaml"

def evaluate():
    model = YOLO(MODEL_PATH)
    metrics = model.val(data=DATA_YAML)

    print("\n📊 Evaluation Results:")
    print(f"  mAP50      : {metrics.box.map50:.4f}")
    print(f"  mAP50-95   : {metrics.box.map:.4f}")
    print(f"  Precision  : {metrics.box.mp:.4f}")
    print(f"  Recall     : {metrics.box.mr:.4f}")

    plots_dir = "runs/helmet_detector"
    for plot_file in ["results.png", "confusion_matrix.png", "PR_curve.png"]:
        path = os.path.join(plots_dir, plot_file)
        if os.path.exists(path):
            img = mpimg.imread(path)
            plt.figure(figsize=(12, 6))
            plt.imshow(img)
            plt.axis("off")
            plt.title(plot_file.replace(".png", "").replace("_", " "))
            plt.tight_layout()
            plt.savefig(f"runs/{plot_file}")
            plt.show()

if __name__ == "__main__":
    evaluate()