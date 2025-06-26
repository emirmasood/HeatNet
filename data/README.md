# Data

This folder contains all the images, depth maps, labels and keypoint files used by the project.
**Note:** Large files (models, full datasets) are **not** stored in this repo.
> Download them here:  
> https://drive.google.com/drive/folders/1bMuIT9NpPXCQPV6SGFvr6aIEn42B3BZ-?usp=sharing

## Folder structure

The data folder contains following subfolders:

● cropped_resized_data/ - Contains RGB images cropped by YOLO bounding boxes and resized to 256×256 for heatmap‐based keypoint training.
● cropped_resized_depth_data/ - Contains corresponding depth maps cropped and resized in the same way for the depth extension model.
● full_data/ - Holds the complete set of RGB and depth images split into train/test folders, along with gt.json supplying 6D ground‐truth poses.
● point_sampling_data/ - Includes 3D_50_keypoints_fps/cps.json (FPS/CPS‐sampled 3D points), 2D_50_keypoints_labels_fps/cps.json (projected 2D points), and the heatmaps_sigma_2 folder with Gaussian heatmaps used for training.
● predicted_key_points/ - Stores JSON files of 2D keypoint coordinates predicted by different model variants. These keypoints are used as input to a PnP solver to compute the final 6D object pose.
● raw/ - Contains the original LineMOD dataset organized by object ID, with subfolders rgb/, depth/, mask/ and metadata files (gt.yml, info.yml, train.txt, test.txt) for 6D pose annotation.
● yolo_data/ - Provides train/ and val/ splits of images, label files, and a data.yml configuration for training the YOLO object detection.
