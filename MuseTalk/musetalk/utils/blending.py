from PIL import Image
import numpy as np
import cv2
import copy
from face_parsing import FaceParsing
from utils.utils import apply_super_resolution

fp = FaceParsing()

def get_crop_box(box, expand):
    x, y, x1, y1 = box
    x_c, y_c = (x + x1) // 2, (y + y1) // 2
    w, h = x1 - x, y1 - y
    s = int(max(w, h) // 2 * expand)
    crop_box = [x_c - s, y_c - s, x_c + s, y_c + s]
    return crop_box, s

def face_seg(image):
    seg_image = fp(image)
    if seg_image is None:
        print("Error: No person segment detected")
        return None

    seg_image = seg_image.resize(image.size)
    return seg_image

def blend_with_superresolution(image, face, face_box, mask_array, crop_box, method="GFPGAN"):
    """
    Blend the generated face into the image, applying super-resolution if necessary.
    Args:
        image (numpy.ndarray): The original frame.
        face (numpy.ndarray): The generated face subframe.
        face_box (tuple): The bounding box of the face (x, y, x1, y1).
        mask_array (numpy.ndarray): The mask for blending.
        crop_box (tuple): The bounding box for the crop.
        method (str): Super-resolution method ("GFPGAN" or "CodeFormer").
    Returns:
        numpy.ndarray: The blended frame.
    """
    x, y, x1, y1 = face_box
    x_s, y_s, x_e, y_e = crop_box
    original_res = (y1 - y, x1 - x)
    generated_res = face.shape[:2]

    # Apply super-resolution if necessary
    if generated_res[0] < original_res[0] or generated_res[1] < original_res[1]:
        face = apply_super_resolution(face, method=method)

    face_large = copy.deepcopy(image[y_s:y_e, x_s:x_e])
    face_large[y - y_s:y1 - y_s, x - x_s:x1 - x_s] = face

    mask_image = cv2.cvtColor(mask_array, cv2.COLOR_BGR2GRAY)
    mask_image = (mask_image / 255).astype(np.float32)

    blended_frame = cv2.blendLinear(face_large, image[y_s:y_e, x_s:x_e], mask_image, 1 - mask_image)
    image[y_s:y_e, x_s:x_e] = blended_frame

    return image
