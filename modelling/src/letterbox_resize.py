import numpy as np
import cv2

def letterbox_resize(image, labels, new_shape=640, color=(114/255, 114/255, 114/255), scaleup=True):
    """
    Resize image and bounding boxes using letterbox method.
    
    [Note: 114/255 is approx equal to 0.447 which is the padded value expected by Ultralytics.]

    Args:
        image (ndarray):   Input image with shape (C, H, W) and normalised values [0, 1].
        labels (ndarray):  Bounding boxes in normalised YOLO format (cls xc yc w h).
        color: (float:):   When padding applied to make image square, OpenCV fills the
                           border with a constant color (normalised-color value).
        scaleup (Boolean): Only applicable when images size is less than 640.
    Returns:
        image_resized (ndarray): Resized and padded image (C, H', W').
        labels (ndarray):        Adjusted normalised bounding boxes (cls xc yc w h).
    """
    h0, w0 = image.shape[:2]  # original height, width
    new_shape = (new_shape, new_shape)

    # Calculate scale ratio
    r = min(new_shape[0] / h0, new_shape[1] / w0)
    if not scaleup:
      r = min(r, 1.0)

    # Calculate new unpadded size
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))  # (width, height)

    # Calculate padding
    dw = new_shape[1] - new_unpad[0]  # width padding
    dh = new_shape[0] - new_unpad[1]  # height padding
    dw /= 2  # split equally
    dh /= 2

    # Resize image
    image_resized = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    # Add padding
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image_resized = cv2.copyMakeBorder(
      image_resized, top, bottom, left, right,
      cv2.BORDER_CONSTANT, value=color
    )
        
    # Transform labels (working directly with normalised coordinates)
    if labels.size > 0:
          # Convert normalized coords to original pixel space
        labels[:, 1] *= w0  # x_center
        labels[:, 2] *= h0  # y_center  
        labels[:, 3] *= w0  # width
        labels[:, 4] *= h0  # height

        # Apply letterbox transformation
        labels[:, 1] = r * labels[:, 1] + left    # x_center
        labels[:, 2] = r * labels[:, 2] + top     # y_center
        labels[:, 3] = r * labels[:, 3]           # width
        labels[:, 4] = r * labels[:, 4]           # height

        # Normalize to new image size
        labels[:, [1, 3]] /= new_shape[1]  # normalise x, w
        labels[:, [2, 4]] /= new_shape[0]  # normalise y, h
          
    return image_resized, labels
