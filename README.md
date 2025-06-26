# 6D Pose Estimation via Keypoint Heatmap Regression with RGB-D Residual Neural Networks

This repository provides the official implementation of our Machine Learning & Deep Learning project focused on 6D object pose estimation.

---

## 📂 Structure

```
.
├── data/                                                                        # images, depth maps, labels, and keypoints
├── docs/                                                                        # project report and related documents
│   ├── s337769_s337006_s344174_s342583_ALJOSEVIC_ALMASI_PAROVIC_SHAFIEI.pdf     # project report
│   └── instructions.pdf                                                         # instructions of project structure and execution
├── models/                                                                      # trained model checkpoints
│   ├── resnet/                                                                  # ResNet-based models
│   ├── yolov10m/                                                                # YOLO model artifacts
│   └── yolov10m.pt                                                              # pretrained YOLO weights
├── notebooks/                                                                   # Jupyter notebooks for pipeline phases
│   └── end_to_end/                                                              # complete pipeline notebook and modules
├── requirements.txt                                                             # Python dependencies
└── README.md                                                                    # project overview
```

---

## 🚀 Getting Started

### Step 1: Clone the Repository

```bash
git clone https://github.com/emirmasood/HeatNet.git
cd HeatNet
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Download Data and Models

Due to GitHub's file size restrictions, download large files separately:

* **Dataset:** [Google Drive Data Folder](https://drive.google.com/drive/folders/1bMuIT9NpPXCQPV6SGFvr6aIEn42B3BZ-?usp=sharing)
* * **ResNet Checkpoints:** [ResNet Checkpoints](https://drive.google.com/drive/folders/14pTckwpHFnaL27vCwQ3DRbv9XOCgZZOM?usp=drive_link)
* **YOLOv10m pretrained weights:** [YOLOv10m Checkpoint](https://drive.google.com/file/d/1mRdriU3u85oxcL0CPeIhJBxX795iENse/view?usp=drive_link)

Place downloaded files into their respective folders as indicated in the folder structure above.

### Step 4: Run the Full Pipeline

```bash
jupyter lab notebooks/end_to_end/ph5_01_end_to_end.ipynb
```

---

## 👥 Authors
This project was created by:

Ismail Aljosevic (ismail.aljosevic@studenti.polito.it)

Amir Masoud Almasi (amirmasoud.almasi@studenti.polito.it)

Ana Parovic (ana.parovic@studenti.polito.it)

Ashkan Shafiei (ashkan.shafiei@studenti.polito.it)
