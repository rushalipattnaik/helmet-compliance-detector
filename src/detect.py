import cv2
from ultralytics import YOLO
import argparse

MODEL_PATH = "runs/helmet_detector/weights/best.pt"
CLASSES    = ["With Helmet", "Without Helmet"]
COLORS     = {
    "With Helmet"    : (0, 255, 0),
    "Without Helmet" : (0, 0, 255),
}

def draw_boxes(frame, results):
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            label  = CLASSES[cls_id]
            color  = COLORS.get(label, (255, 255, 255))
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1-th-8), (x1+tw, y1), color, -1)
            cv2.putText(frame, text, (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    labels_detected = [CLASSES[int(b.cls[0])] for r in results for b in r.boxes]
    if "Without Helmet" in labels_detected:
        cv2.putText(frame, "WARNING: NO HELMET!", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    elif "With Helmet" in labels_detected:
        cv2.putText(frame, "COMPLIANT", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
    return frame

def run_on_image(img_path):
    model   = YOLO(MODEL_PATH)
    frame   = cv2.imread(img_path)
    results = model(frame)
    frame   = draw_boxes(frame, results)
    cv2.imwrite("runs/output_image.jpg", frame)
    print("✅ Saved to runs/output_image.jpg")
    cv2.imshow("Helmet Detection", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run_on_video(video_path):
    model  = YOLO(MODEL_PATH)
    cap    = cv2.VideoCapture(0 if video_path == "webcam" else video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, verbose=False)
        frame   = draw_boxes(frame, results)
        if out is None:
            h, w = frame.shape[:2]
            out = cv2.VideoWriter("runs/output_video.mp4", fourcc, 20, (w, h))
        out.write(frame)
        cv2.imshow("Helmet Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="webcam")
    args = parser.parse_args()
    if args.source.endswith((".jpg", ".jpeg", ".png")):
        run_on_image(args.source)
    else:
        run_on_video(args.source)