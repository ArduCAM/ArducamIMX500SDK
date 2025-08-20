import cv2
import numpy as np


COLOURS = np.array(
[
    [128,   0,   0],
    [  0, 128,   0],
    [128, 128,   0],
    [  0,   0, 128],
    [128,   0, 128],
    [  0, 128, 128],
    [128, 128, 128],
    [ 64,   0,   0],
    [192,   0,   0],
    [ 64, 128,   0],
    [192, 128,   0],
    [ 64,   0, 128],
    [192,   0, 128],
    [ 64, 128, 128],
    [192, 128, 128],
    [  0,  64,   0],
    [128,  64,   0],
    [  0, 192,   0],
    [128, 192,   0],
    [  0,  64, 128],
    [  0,   0,   0],
])

def parse_deeplabv3plus(network, img):
    score_thr=0.5
    nn_input_map=(0.0, 0.0, 1.0, 1.0)
    dnn_input_img = network[0].input_tensors[0].data.copy()
    if dnn_input_img is None:
        raise Exception("Input tensor is None")
    if dnn_input_img.shape[0] == 3:
        w, h, c = dnn_input_img.shape[1], dnn_input_img.shape[2], dnn_input_img.shape[0]
        dnn_input_img = dnn_input_img.transpose(2, 0, 1).reshape(c, h, w).transpose(1, 2, 0)
    else:
        w, h, c = dnn_input_img.shape[1], dnn_input_img.shape[0], dnn_input_img.shape[2]
    dnn_input_img = cv2.cvtColor(dnn_input_img, cv2.COLOR_RGB2BGR)
    dnn_output_tensor = network[0].output_tensors[0].data
    if dnn_output_tensor is None:
        print("Warning: Output tensor is None")
        return None, None

    h, w = dnn_input_img.shape[:2]
    mask = dnn_output_tensor.astype(np.uint16)

    overlay_input = np.zeros((h, w, 4), dtype=np.uint8)
    for class_id in np.unique(mask):
        if class_id == 0 or class_id >= len(COLOURS):
            continue
        color = COLOURS[class_id].astype(np.uint8)
        class_mask = (mask == class_id)
        if class_mask.ndim == 3 and class_mask.shape[2] == 1:
            class_mask = class_mask[:, :, 0]
        overlay_input[class_mask] = [*color, 128]

    orig_h, orig_w = img.shape[:2]
    xmin, ymin, xmax, ymax = nn_input_map
    xmin_abs = int(xmin * orig_w)
    ymin_abs = int(ymin * orig_h)
    xmax_abs = int(xmax * orig_w)
    ymax_abs = int(ymax * orig_h)
    crop_w = xmax_abs - xmin_abs
    crop_h = ymax_abs - ymin_abs

    overlay_resized = cv2.resize(overlay_input, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)

    input_bgra = cv2.cvtColor(dnn_input_img, cv2.COLOR_BGR2BGRA)
    img_bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    alpha_input = overlay_input[:, :, 3:] / 255.0
    blended_input_rgb = (overlay_input[:, :, :3] * alpha_input + input_bgra[:, :, :3] * (1 - alpha_input)).astype(np.uint8)
    output_input_tensor = np.dstack((blended_input_rgb, input_bgra[:, :, 3]))

    alpha_crop = overlay_resized[:, :, 3:] / 255.0
    orig_crop = img_bgra[ymin_abs:ymax_abs, xmin_abs:xmax_abs]
    blended_crop_rgb = (overlay_resized[:, :, :3] * alpha_crop + orig_crop[:, :, :3] * (1 - alpha_crop)).astype(np.uint8)
    blended_crop = np.dstack((blended_crop_rgb, orig_crop[:, :, 3]))

    output_img = img_bgra.copy()
    output_img[ymin_abs:ymax_abs, xmin_abs:xmax_abs] = blended_crop

    return output_img, output_input_tensor

    
