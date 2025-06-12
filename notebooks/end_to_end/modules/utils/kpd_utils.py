

import torch

def extract_keypoints_to_original_image_space(heatmaps, bbox, hmap_size=64):
    _, K, H, W = heatmaps.shape
    x1, y1, x2, y2 = bbox
    crop_width = x2 - x1
    crop_height = y2 - y1
    keypoints = []
    heatmaps = heatmaps[0].cpu()

    for k in range(K):
        hm = heatmaps[k]
        idx = torch.argmax(hm.view(-1)).item()
        y_hm, x_hm = divmod(idx, W)
        x_crop = (x_hm + 0.5) * (256 / hmap_size)
        y_crop = (y_hm + 0.5) * (256 / hmap_size)
        x_orig = x1 + (x_crop / 256) * crop_width
        y_orig = y1 + (y_crop / 256) * crop_height
        keypoints.append([x_orig, y_orig])

    return torch.tensor(keypoints)
