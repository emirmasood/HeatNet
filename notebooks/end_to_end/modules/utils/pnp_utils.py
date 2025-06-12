

import numpy as np
import cv2

# ─── CONSTANTS ─────────────────────────────────────────────────────────────────
K = np.array([
    [572.4114,   0.0,      325.2611],
    [  0.0,    573.57043,  242.04899],
    [  0.0,      0.0,        1.0   ]
], dtype=np.float64)



def eliminate_duplicate_pairs(pts2d, pts3d):

  pairs = list(zip(pts3d, pts2d))
  unique_pairs = []
  seen_2d = set()

  for p3d, p2d in pairs:
      key = tuple(np.round(p2d))

      if key in seen_2d:
          continue

      seen_2d.add(key)
      unique_pairs.append((p3d, p2d))

  if len(unique_pairs) < 4:
      print(f"[WARNING] Not enough unique 2D keypoints after deduplication")
      return None


  pts3d_clean = np.array([p3d for p3d, _ in unique_pairs], dtype=np.float64)
  pts2d_clean = np.array([p2d for _, p2d in unique_pairs], dtype=np.float64)

  return pts2d_clean, pts3d_clean


def run_pnp(img_id, keypoints_2d, kp3d_dict, camera_matrix=K):
  
  # select 3D model
  obj_key = img_id.split("_")[0]

  pts3d = np.array(kp3d_dict[obj_key])
  pts2d = np.array(keypoints_2d, np.float64)

  #this call is for eliminating duplicate points, but it does not solve out issue
  pts2d, pts3d = eliminate_duplicate_pairs(pts2d, pts3d)

  success, rvec, tvec, inliers = cv2.solvePnPRansac(
      objectPoints = pts3d,
      imagePoints = pts2d,
      cameraMatrix = K,
      distCoeffs = None,
      reprojectionError = 4,      # Very high threeshold, it was 15, after 5, #I tried  1 and 3, worse results
      confidence = 0.99,  #
      flags = cv2.SOLVEPNP_ITERATIVE
  )


  if not success:
    #Solution could not be found under this reprojection error
      return None

  R_mat, _ = cv2.Rodrigues(rvec)
  return obj_key, {"R": R_mat.tolist(), "t": tvec.flatten().tolist()}, inliers
