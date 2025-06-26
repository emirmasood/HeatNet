# Data

This folder contains all the images, depth maps, labels and keypoint files used by the project.

> **Note:** Due to the size limit, large data files are **not** stored here. You can download them from:
> https://drive.google.com/drive/folders/1bMuIT9NpPXCQPV6SGFvr6aIEn42B3BZ-?usp=sharing

## Folder structure

- **`raw/`**  
  Original LineMOD dataset, organized by object ID with subfolders:
  - `rgb/`, `depth/`, `mask/`  
  - Metadata files (`gt.yml`, `info.yml`, `train.txt`, `test.txt`)

- **`full_data/`**  
  Complete RGB + depth set split into `train/` and `test/`, with `gt.json` for 6D pose ground truth.

- **`cropped_resized_data/`**  
  RGB images cropped by YOLO bounding boxes and resized to 256×256 for keypoint heatmap training.

- **`cropped_resized_depth_data/`**  
  Corresponding depth maps cropped and resized to 256×256 for the depth-extended model.

- **`point_sampling_data/`**  
  - `3D_50_keypoints_fps.json` / `cps.json`: 50 points sampled on the 3D mesh  
  - `2D_50_keypoints_labels_fps.json` / `cps.json`: those points projected to 2D  
  - `heatmaps_sigma_2/`: Gaussian heatmaps for training

- **`predicted_key_points/`**  
  2D keypoint JSON outputs from each model variant, used by the PnP solver to compute the final 6D pose.

- **`yolo_data/`**  
  YOLO-formatted data for object detection:
  - `train/` and `val/` image folders  
  - `.txt` label files  
  - `data.yml` configuration
