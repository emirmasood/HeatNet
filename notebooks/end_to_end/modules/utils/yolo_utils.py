

import cv2
import numpy as np
import matplotlib.pyplot as plt


def is_valid_bbox(x1, y1, x2, y2, img_width=640, img_height=480):
    return (
        x2 > x1 and y2 > y1 and
        x1 >= 0 and y1 >= 0 and
        x2 <= img_width and y2 <= img_height
    )

def crop_and_resize(image_path, result, bbox_index=0, output_path=None, resize_shape=(256, 256)):

    image = cv2.imread(image_path)
    img_height, img_width = image.shape[:2]

    try:
        box = result[0].boxes[bbox_index]
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        if not is_valid_bbox(x1, y1, x2, y2, img_width, img_height):
            return None, "Invalid bbox"

        cropped = image[y1:y2, x1:x2]
        resized = cv2.resize(cropped, resize_shape)        
        return resized, None

    except Exception as e:
        return None, str(e)


def plot_detection_and_crop(results, cropped_image):

    original_with_boxes = results[0].plot()

    plt.figure(figsize=(10, 5))


    #YOLO Detection
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original_with_boxes, cv2.COLOR_BGR2RGB))
    plt.title("YOLO Detection")
    plt.axis('off')

    #Cropped and Resized output
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    plt.title("Cropped and Resized YOLO output")
    plt.axis('off')

    plt.tight_layout()
    plt.show()



