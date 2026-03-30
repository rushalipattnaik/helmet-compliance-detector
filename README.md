# 🪖 Helmet Compliance Detector (Worker Safety System)

An AI-powered Computer Vision system designed to monitor **worker safety compliance** by detecting whether individuals are wearing helmets or not.

This project is particularly useful for:
- 🏗️ Construction sites  
- 🏭 Industrial environments  
- 🚧 Road safety monitoring  

---

## 🎯 Problem Statement

Ensuring that workers wear helmets is critical for preventing injuries. Manual monitoring is inefficient and error-prone.

👉 This system automates helmet compliance detection using deep learning, enabling:
- Real-time monitoring  
- Instant violation alerts  
- Improved workplace safety  

---

## 🚀 Features

- 🔍 Detects **Helmet / No Helmet**  
- ⚠️ Highlights **non-compliant workers**  
- 🎥 Supports **image, video & webcam detection**  
- 🌐 Interactive **Streamlit Web App**  
- 📊 Model evaluation with metrics & plots  
- ⚡ Fast inference using YOLOv8  

---

## 🧠 Tech Stack

- Python  
- YOLOv8 (Ultralytics)  
- OpenCV  
- Streamlit  
- NumPy  
- Pandas  
- Matplotlib  

---

## 📊 Results

| Metric | Value |
|--------|-------|
| mAP50 | 92.5% |
| mAP50-95 | 59.9% |
| Precision | 84.7% |
| Recall | 89.2% |
| With Helmet Accuracy | 95.5% |
| Without Helmet Accuracy | 89.5% |
| Training Time | ~22 mins (Colab T4 GPU) |
| Dataset Size | ~150MB (3648 training images) |

---

## 🏗️ System Workflow

~~~
Input (Image / Video / Webcam)
        ↓
YOLOv8 Model (Helmet Detection)
        ↓
Bounding Boxes + Labels
        ↓
Compliance Check
        ↓
⚠️ Alert (No Helmet) / ✅ Safe
~~~

---

## 📂 Project Structure

~~~
helmet-compliance-detector/
│
├── app/
│   └── streamlit_app.py        # Web UI
│
├── src/
│   ├── train.py                # Model training
│   ├── detect.py               # Detection
│   ├── evaluate.py             # Evaluation
│   └── prepare_data.py         # Data preprocessing
│
├── data/
│   └── dataset/
│       └── data.yaml
│
├── runs/
├── requirements.txt
├── README.md
└── .gitignore
~~~

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository
~~~bash
git clone https://github.com/rushapipattnaik/helmet-compliance-detector.git
cd helmet-compliance-detector
~~~

### 2️⃣ Install Dependencies
~~~bash
pip install -r requirements.txt
~~~

---

## ▶️ Usage

### 🔹 Run Detection (Webcam / Video / Image)
~~~bash
python src/detect.py --source webcam
~~~

OR

~~~bash
python src/detect.py --source input.jpg
~~~

---

### 🌐 Run Streamlit Web App
~~~bash
streamlit run app/streamlit_app.py
~~~

👉 Upload an image and get:
- Detection results  
- Compliance status (Safe / Violation)  

---

## 🧪 Model Training

~~~bash
python src/train.py
~~~

**Training Details:**
- Model: YOLOv8s  
- Epochs: 30  
- Image Size: 640  
- Batch Size: 8  

---

## 📈 Evaluation

~~~bash
python src/evaluate.py
~~~

Generates:
- 📊 Confusion Matrix  
- 📉 Precision-Recall Curve  
- 📈 Training Results Graph  

---

## 📁 Dataset

- Source: Roboflow (Helmet Detection Dataset)  
- Classes:
  - With Helmet  
  - Without Helmet  

⚠️ Dataset not included due to size constraints.

---

## ⚠️ Output Behavior

- 🟢 Green Box → Helmet detected (Safe)  
- 🔴 Red Box → No Helmet (Violation)  
- ⚠️ Warning displayed for non-compliance  

---

## 🔮 Future Improvements

- Real-time CCTV integration  
- Cloud deployment (Streamlit / HuggingFace)  
- Alert system (SMS / Email)  
- Multi-worker tracking  
- PPE detection (vests, gloves, etc.)  

---

## 👩‍💻 Author

**Rushali Pattnaik**

---
