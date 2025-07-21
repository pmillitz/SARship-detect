import numpy as np
import cv2

def letterbox_resize(image, labels, new_shape=640, color=(0.1, 0.1, 0.1), scaleup=True):
    """
    Resize image and bounding boxes using letterbox method.

    Args:
        image (ndarray):   Input image with shape (C, H, W) and normalised values [0, 1].
        labels (ndarray):  Bounding boxes in normalised YOLO format (cls xc yc w h).
        color: (float:):   When padding applied to make image square, OpenCV fills the
                           border with a constant color (normalised-color value).
        scaleup (Boolean): Only applicable when images size is less than 640.
    Returns:
        image_resized (ndarray): Resized and padded image (H', W', C).
        labels (ndarray):        Adjusted normalised bounding boxes (cls xc yc w h).
    """    
    shape = image.shape[:2]  # current shape [height, width]
    new_shape = (new_shape, new_shape)

    # Compute scale ratio (new / old) and padding
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down
        r = min(r, 1.0)

    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw = new_shape[1] - new_unpad[0]  # width padding
    dh = new_shape[0] - new_unpad[1]  # height padding
    dw /= 2
    dh /= 2

    # Resize
    image_resized = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    # Pad
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image_resized = cv2.copyMakeBorder(image_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    # Adjust labels
    if labels.size > 0:
        labels[:, 1] = r * labels[:, 1] + left    # x_center
        labels[:, 2] = r * labels[:, 2] + top     # y_center
        labels[:, 3] = r * labels[:, 3]           # width
        labels[:, 4] = r * labels[:, 4]           # height
        labels[:, [1, 3]] /= new_shape[1]         # normalize x, w
        labels[:, [2, 4]] /= new_shape[0]         # normalize y, h

    return image_resized, labels
