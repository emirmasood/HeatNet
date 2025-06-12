
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple

def get_gt_pose(img_id, gt_data):
    entry = gt_data[img_id]
    R_list, t_list = entry[0], entry[1]
    R = np.array(R_list, float).reshape(3, 3)
    t = np.array(t_list, float).flatten().reshape(3, 1)
    return R, t


def evaluate_pose_estimation(pnp_results,
                             kp3d,
                             gt_data,
                             diameter_map,
                             symmetric_objects,
                             threshold_ratio=0.1):

    correct_by_class = defaultdict(int)
    total_by_class   = defaultdict(int)
    results_distribution_class = defaultdict(list)
    high_error_samples = defaultdict(list)

    for img_id, (est, _) in pnp_results.items():
        cls = img_id.split("_")[0]

        try:
            Rg, tg = get_gt_pose(img_id, gt_data)
        except KeyError:
            continue

        Rp = np.array(est["R"], dtype=float)
        tp = np.array(est["t"], dtype=float).reshape(3, 1)

        # project keypoints
        pts3d = np.array(kp3d[cls], dtype=np.float32)  # (N,3)
        gt_tr  = (Rg @ pts3d.T + tg).T                 # (N,3)
        pnp_tr = (Rp @ pts3d.T + tp).T                 # (N,3)

        if cls in symmetric_objects:
            # ADD-S: minimal distance between each GT point and closest predicted
            add_s_err = np.mean([
                np.min(np.linalg.norm(gt_point - pnp_tr, axis=1))
                for gt_point in gt_tr
            ])
            add_err = add_s_err
        else:
            # standard ADD
            add_err = np.mean(np.linalg.norm(gt_tr - pnp_tr, axis=1))

        total_by_class[cls] += 1
        results_distribution_class[cls].append(add_err / diameter_map[cls])

        if add_err < threshold_ratio * diameter_map[cls]:
            correct_by_class[cls] += 1
        else:
            high_error_samples[cls].append((img_id, add_err))

    # compute accuracy percentages
    accuracy_results = {
        cls: 100.0 * correct_by_class[cls] / total_by_class[cls]
        for cls in total_by_class
    }

    # sort incorrect results by error
    for cls in high_error_samples:
        high_error_samples[cls].sort(key=lambda x: x[1], reverse=True)

    return accuracy_results, results_distribution_class, high_error_samples
