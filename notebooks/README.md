# Notebooks

This folder contains all Jupyter/Colab notebooks to run each step of the project.

## Contents

- **`ph1_01_data_prep.ipynb`**  
  Executes the essential data preparation steps required for the project

- **`ph2_01_data_prep.ipynb`**  
  Prepares the dataset for YOLO, which requires a specific structure and format.

- **`ph2_02_object_detection.ipynb`**  
  Performs object detection on RGB images using YOLO.

- **`ph3_01_data_prep.ipynb`**  
  Parses YOLO detections, crops RGB and depth patches to 256×256, and prepares them for heatmap‐based keypoint regression.

- **`ph3_02_point_sampling.ipynb`**  
  Applies curvature‐based (CPS) and farthest‐point (FPS) sampling to CAD point clouds and exports them for the next phase of the project.

- **`ph3_03_dimension_projection.ipynb`**  
  Samples keypoints from the 3D point cloud, projects them into 2D image coordinates and converts them into Gaussian heatmaps.

- **`ph3_04_baseline_model_definition.ipynb`**  
  Trains a convolutional neural network using a heatmap-based approach to accurately predict position of object keypoints from RGB images

- **`ph3_05_pnp_add.ipynb`**  
  Runs the PnP algorithm together with the RANSAC method to robustly estimate the 6D pose of the object based on known 2D-3D correspondences.

- **`ph4_01_depth_extension_with_general_training_experiments.ipynb`**  
  Extends our heatmap-based keypoint regression to leverage both RGB and depth inputs, carries out general training experiments to compare different activation functions and learning-rate schedulers.

- **`ph5_01_end_to_end.ipynb`**  
  Unifies all pipeline stages into a single cohesive workflow, and modules/ folder supplying all necessary scripts and utilities to execute the full system.
