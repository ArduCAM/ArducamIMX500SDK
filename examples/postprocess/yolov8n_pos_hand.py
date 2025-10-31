import cv2
import numpy as np


RNG = np.random.default_rng(3)
COLORS = RNG.uniform(0, 255, size=(1, 3))

hand_demo = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18),(18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

class KeyPoint:
    def __init__(self, x, y, score):
        self.x = x
        self.y = y
        self.score = score
        
def vis(img, boxes, scores, cls_ids, keypoints, class_names=["hand"], obejct_detect_conf=0.2, kpt_detect_conf=0.2):
    h, w = img.shape[:2]
    base_scale = min(h, w) / 640.0
    font_scale = max(0.4, base_scale * 0.7)
    box_thickness = max(1, int(base_scale * 2))
    text_thickness = max(1, int(base_scale * 2))
    kpt_radius = max(2, int(base_scale * 3))
    line_thickness = max(1, int(base_scale * 2))

    font = cv2.FONT_HERSHEY_SIMPLEX

    for i in range(len(boxes)):
        if scores[i] < obejct_detect_conf:
            continue

        box = boxes[i]
        cls_id = int(cls_ids[i])
        keypoints_raw = keypoints[i]

        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int((box[0] + box[2]))
        y1 = int((box[1] + box[3]))

        color = (COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = f"hand:{scores[i] * 100:.1f}%"
        txt_color = (0, 0, 0) if np.mean(COLORS[cls_id]) > 0.5 else (255, 255, 255)
        txt_size = cv2.getTextSize(text, font, font_scale, text_thickness)[0]
        txt_bk_color = (np.array(color) * 0.7).astype(np.uint8).tolist()

        cv2.rectangle(img, (x0, y0), (x1, y1), color, box_thickness)
        cv2.rectangle(img, (x0, y0 + 1), (x0 + txt_size[0] + 2, y0 + int(1.5 * txt_size[1])), txt_bk_color, -1)
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, font_scale, txt_color, text_thickness)

        kps = [KeyPoint(keypoints_raw[j*3], keypoints_raw[j*3+1], keypoints_raw[j*3+2]) for j in range(21)]

        for kp in kps:
            if kp.score >= 0.2:
                x = int(kp.x)
                y = int(kp.y)
                cv2.circle(img, (x, y), kpt_radius, (0, 0, 255), -1)

        def draw_lines(lines):
            for a, b in lines:
                ka, kb = kps[a], kps[b]
                if ka.score >= 0.2 and kb.score >= 0.2:
                    xa = int(ka.x)
                    ya = int(ka.y)
                    xb = int(kb.x)
                    yb = int(kb.y)
                    cv2.line(img, (xa, ya), (xb, yb), (255, 255, 255), line_thickness)

        draw_lines(hand_demo)
        return img

def parse_yolov8n_hand_pos(network, img):
    score_thr=0.5
    nn_input_map=(0.0, 0.0, 1.0, 1.0)
    dnn_input_img = network[0].input_tensors[0].data.copy()
    if dnn_input_img is None:
        raise Exception("Input tensor is None")

    w = None
    h = None
    c = None
    
    if (dnn_input_img.shape[0] == 3):
        w = dnn_input_img.shape[1]
        h = dnn_input_img.shape[2]
        c = dnn_input_img.shape[0]
        dnn_input_img = dnn_input_img.transpose(2, 0, 1).reshape(c, h, w).transpose(1, 2, 0)
    else:
        w = dnn_input_img.shape[1]
        h = dnn_input_img.shape[0]
        c = dnn_input_img.shape[2]
        
    dnn_input_img = cv2.cvtColor(dnn_input_img, cv2.COLOR_RGB2BGR)
    
    dnn_output_tensor = network[0].output_tensors[0].data
    if dnn_output_tensor is None:
        print("Warning: Output tensor is None")
        return None, None
        
    boxes = network[0].output_tensors[0].data
    scores = network[0].output_tensors[1].data
    cls_ids = network[0].output_tensors[2].data
    keypoints = network[0].output_tensors[3].data
    
    crop_xmin_abs = int(nn_input_map[0] * img.shape[1])
    crop_ymin_abs = int(nn_input_map[1] * img.shape[0])
    crop_xmax_abs = int(nn_input_map[2] * img.shape[1])
    crop_ymax_abs = int(nn_input_map[3] * img.shape[0])
    
    crop_w = crop_xmax_abs - crop_xmin_abs
    crop_h = crop_ymax_abs - crop_ymin_abs
    
    keypoints_yuv = keypoints.copy()
    x_scale = crop_w / w
    y_scale = crop_h / h
    boxes_yuv = boxes.astype(np.float32)
    boxes_yuv[:, 0] = boxes_yuv[:, 0] * x_scale + crop_xmin_abs
    boxes_yuv[:, 2] = boxes_yuv[:, 2] * x_scale + crop_xmin_abs
    boxes_yuv[:, 1] = boxes_yuv[:, 1] * y_scale + crop_ymin_abs
    boxes_yuv[:, 3] = boxes_yuv[:, 3] * y_scale + crop_ymin_abs
    for i in range(keypoints_yuv.shape[1]):
        if (i+1) % 3 == 0:
            pass
        elif (i+1) % 3 == 1:
            keypoints_yuv[:, i] = keypoints[:, i] * x_scale + crop_xmin_abs
        elif (i+1) % 3 == 2:
            keypoints_yuv[:, i] = keypoints[:, i] * y_scale + crop_ymin_abs
    boxes_yuv = boxes_yuv.astype(np.int32)
    
    vis(dnn_input_img, boxes, scores, cls_ids, keypoints)
    vis(img, boxes_yuv, scores, cls_ids, keypoints_yuv)

    return img, dnn_input_img