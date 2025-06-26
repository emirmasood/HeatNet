# 6D Pose Estimation via Keypoint Heatmap Regression with RGB-D Residual Neural Networks

This repository provides the official implementation of our Machine Learning & Deep Learning project focused on 6D object pose estimation. The pipeline utilizes YOLOv10-medium for initial object detection, a ResNet18-based convolutional neural network for keypoint heatmap regression, and the Perspective-n-Point (PnP) algorithm with RANSAC for pose estimation. To enhance accuracy, we introduced depth information into our baseline RGB model through a cross-fusion strategy, combining RGB and depth modalities effectively. Our best model achieved an impressive mean ADD accuracy of 91.74% on the LINEMOD dataset.

---

## 📂 Structure

```
.
├── data/             # images, depth maps, labels, and keypoints
├── docs/             # project report and related documents
├── models/           # trained model checkpoints
│   ├── resnet/       # ResNet-based models
│   ├── yolov10m/     # YOLO model artifacts
│   └── yolov10m.pt   # pretrained YOLO weights
├── notebooks/        # Jupyter notebooks for pipeline phases
│   └── end_to_end/   # complete pipeline notebook and modules
├── requirements.txt  # Python dependencies
└── README.md         # project overview
```

---

## 🚀 Getting Started

### Step 1: Clone the Repository

```bash
git clone https://github.com/emirmasood/HeatNet.git
cd 6d-pose-estimation
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Download Data and Models

Due to GitHub's file size restrictions, download large files separately:

* **Dataset:** [Google Drive Data Folder](https://drive.google.com/drive/folders/1bMuIT9NpPXCQPV6SGFvr6aIEn42B3BZ-?usp=sharing)
* **YOLOv10m pretrained weights:** [YOLOv10m Checkpoint](https://drive.google.com/file/d/1mRdriU3u85oxcL0CPeIhJBxX795iENse/view?usp=drive_link)
* **ResNet Checkpoints:** [ResNet Checkpoints](https://drive.google.com/drive/folders/14pTckwpHFnaL27vCwQ3DRbv9XOCgZZOM?usp=drive_link)

Place downloaded files into their respective folders as indicated in the folder structure above.

### Step 4: Run the Full Pipeline

```bash
jupyter lab notebooks/end_to_end/ph5_01_end_to_end.ipynb
```

---

## 📔 Project Notebooks

* **Phase 1:** Data preparation and cleaning
* **Phase 2:** YOLO training and object detection
* **Phase 3:** RGB keypoint regression and PnP pose estimation
* **Phase 4:** RGB-D fusion and extended experiments
* **Phase 5:** End-to-end unified pipeline (`ph5_01_end_to_end.ipynb`)

---

## 📜 Documentation

All relevant documentation, including the project report, is available in the `docs/` folder:

* **[s337769_s337006_s344174_s342583 _ALJOSEVIC_ALMASI_ PAROVIC_SHAFIEI.pdf](docs/s337769_s337006_s344174_s342583 _ALJOSEVIC_ALMASI_ PAROVIC_SHAFIEI.pdf):** Comprehensive project report detailing methodology, experiments, and results.
* **[Instructions.pdf](docs/instructions.pdf):** Internal guide detailing project structure and execution.

---

## 📝 Citation

```bibtex
@misc{aljosevic2025pose,
  title   = {6D Pose Estimation via Keypoint Heatmap Regression with RGB-D ResNet},
  author  = {Aljosevic, I. and Almasi, A. M. and Parovic, A. and Shafiei, A.},
  year    = {2025},
  howpublished = {\url{https://github.com/<your_username>/6d-pose-estimation}}
}
```
