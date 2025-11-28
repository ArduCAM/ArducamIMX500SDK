import cv2
import numpy as np
from typing import Tuple, Optional, List


CLR_GREEN  = (0, 255, 0)
CLR_WHITE  = (255, 255, 255)

def _rect_corners(xyxy: Tuple[float,float,float,float]) -> np.ndarray:
    x1, y1, x2, y2 = xyxy
    return np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], dtype=np.float32)

def _point_in_rect(pt: np.ndarray, rect_xyxy: Tuple[float,float,float,float]) -> bool:
    x1, y1, x2, y2 = rect_xyxy
    return (x1 - 1e-6 <= pt[0] <= x2 + 1e-6) and (y1 - 1e-6 <= pt[1] <= y2 + 1e-6)

def _segments_intersect(a1, a2, b1, b2) -> bool:
    a1 = np.array(a1, dtype=np.float64); a2 = np.array(a2, dtype=np.float64)
    b1 = np.array(b1, dtype=np.float64); b2 = np.array(b2, dtype=np.float64)
    def orient(p, q, r): return np.cross(q - p, r - p)
    def on_seg(p, q, r):
        return (min(p[0], r[0]) - 1e-9 <= q[0] <= max(p[0], r[0]) + 1e-9 and
                min(p[1], r[1]) - 1e-9 <= q[1] <= max(p[1], r[1]) + 1e-9)
    o1 = orient(a1, a2, b1); o2 = orient(a1, a2, b2)
    o3 = orient(b1, b2, a1); o4 = orient(b1, b2, a2)
    if (o1 * o2 < 0) and (o3 * o4 < 0): return True
    if abs(o1) < 1e-9 and on_seg(a1, b1, a2): return True
    if abs(o2) < 1e-9 and on_seg(a1, b2, a2): return True
    if abs(o3) < 1e-9 and on_seg(b1, a1, b2): return True
    if abs(o4) < 1e-9 and on_seg(b1, a2, b2): return True
    return False

def poly_rect_intersect(poly: np.ndarray, rect_xyxy: Tuple[float,float,float,float]) -> bool:
    poly = poly.astype(np.float32)
    rect_pts = _rect_corners(rect_xyxy)
    for p in rect_pts:
        if cv2.pointPolygonTest(poly, (float(p[0]), float(p[1])), False) >= 0:
            return True
    for p in poly:
        if _point_in_rect(p, rect_xyxy): 
            return True
    poly_edges = [(poly[i], poly[(i+1) % len(poly)]) for i in range(len(poly))]
    r = rect_pts
    rect_edges = [(r[0], r[1]), (r[1], r[2]), (r[2], r[3]), (r[3], r[0])]
    for e1 in poly_edges:
        for e2 in rect_edges:
            if _segments_intersect(e1[0], e1[1], e2[0], e2[1]): 
                return True
    return False

def draw_warning_light(im: np.ndarray, intruded: bool, intruder_count: int = 0) -> None:

    h, w = im.shape[:2]
    radius = max(16, int(min(h, w) * 0.025))
    center = (w - radius - 20, radius + 20)
    color = CLR_GREEN if intruded else CLR_WHITE

    overlay = im.copy()
    cv2.circle(overlay, center, int(radius*1.8), (0,0,0), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.35, im, 0.65, 0, im)
    cv2.circle(im, center, int(radius*1.3), CLR_WHITE, 2, cv2.LINE_AA)
    cv2.circle(im, center, radius, color, -1, cv2.LINE_AA)

    txt = "" if intruded else ""
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(im, txt, (center[0]-tw//2, center[1]+radius+th+6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, CLR_WHITE, 2, cv2.LINE_AA)

    count_txt = f"COUNT: {intruder_count}"
    (ctw, cth), _ = cv2.getTextSize(count_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.putText(im, count_txt,
                (center[0]-ctw//2, center[1]+radius+th+cth+16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, CLR_WHITE, 2, cv2.LINE_AA)


class ParserYolov8nDetLabelPackage:

    def __init__(self, label_file_path="labels/person.txt", window_name="IMX500 Fence Setup"):
        self.rng = np.random.default_rng(3)
        self.class_names = self.load_labels(label_file_path)
        self.colors = self.rng.uniform(0, 255, size=(len(self.class_names), 3))

        self.window_name = window_name
        self._fence_poly: Optional[np.ndarray] = None
        self._setup_size: Optional[Tuple[int,int]] = None
        self._fence_active: bool = False
        self._points: List[Tuple[int,int]] = []
        self._did_prompt: bool = False

        self.current_intruder_count: int = 0

    def load_labels(self, filepath):
        labels = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(":", 1)
                label = parts[1].strip() if len(parts) == 2 else line
                labels.append(label)
        return labels

    def draw_masks(self, image, boxes, class_ids, mask_alpha=0.3, mask_maps=None):
        mask_img = image.copy()
        for i, (box, class_id) in enumerate(zip(boxes, class_ids)):
            color = self.colors[class_id]
            x1, y1, x2, y2 = box.astype(int)
            if mask_maps is None:
                cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)
            else:
                crop_mask = mask_maps[i][y1:y2, x1:x2, np.newaxis]
                crop_mask_img = mask_img[y1:y2, x1:x2]
                crop_mask_img = crop_mask_img * (1 - crop_mask) + crop_mask * color
                mask_img[y1:y2, x1:x2] = crop_mask_img
        return cv2.addWeighted(mask_img, mask_alpha, image, 1 - mask_alpha, 0)

    def draw_detections(self, image, boxes, confs, class_ids, mask_alpha=0.3, mask_maps=None):
        img_height, img_width = image.shape[:2]
        base_scale = min(img_height, img_width) / 640
        font_scale = max(0.4, base_scale * 0.7)
        text_thickness = max(1, int(base_scale * 2))
        box_thickness = max(1, int(base_scale * 2))

        mask_img = self.draw_masks(image, boxes, class_ids, mask_alpha, mask_maps)
        for box, score, class_id in zip(boxes, confs, class_ids):
            color = self.colors[class_id]
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, thickness=box_thickness)
            label = self.class_names[class_id]
            caption = f'{label} {int(score * 100)}%'
            (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
            th = int(th * 1.2)
            cv2.rectangle(mask_img, (x1, y1), (x1 + tw, y1 - th), color, -1)
            cv2.putText(mask_img, caption, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (255, 255, 255), thickness=text_thickness, lineType=cv2.LINE_AA)
        return mask_img

    def _mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._points.append((x, y))

    def _ensure_fence_interactive(self, frame: np.ndarray) -> None:

        if self._fence_poly is not None:
            return

        h, w = frame.shape[:2]
        canvas = frame.copy()
        if not self._did_prompt:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, max(960, w//2), max(540, h//2))
            cv2.setMouseCallback(self.window_name, self._mouse_cb)
            self._did_prompt = True
            self._points.clear()

        while True:
            canvas[:] = frame
            cv2.rectangle(canvas, (0, 0), (w, 38), (0, 0, 0), -1)
            msg = f"Click polygon vertices for fence. Points: {len(self._points)}   r=Reset  Enter=Confirm  q=Skip"
            cv2.putText(canvas, msg, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, CLR_WHITE, 2, cv2.LINE_AA)

            for i, p in enumerate(self._points):
                cv2.circle(canvas, p, 6, (0, 200, 255), -1, cv2.LINE_AA)
                cv2.putText(canvas, f"{i+1}", (p[0]+6, p[1]-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2, cv2.LINE_AA)
            if len(self._points) >= 2:
                for i in range(1, len(self._points)):
                    cv2.line(canvas, self._points[i-1], self._points[i],
                             (0, 200, 255), 2, cv2.LINE_AA)
                cv2.line(canvas, self._points[-1], self._points[0],
                         (0, 200, 255), 2, cv2.LINE_AA)

            cv2.imshow(self.window_name, canvas)
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                self._fence_poly = None
                self._fence_active = False
                cv2.destroyWindow(self.window_name)
                break
            elif key == ord('r'):
                self._points.clear()
            elif key in (13, 10):  # Enter
                if len(self._points) >= 3:
                    self._fence_poly = np.array(self._points, dtype=np.float32)
                    self._setup_size = (w, h)
                    self._fence_active = True
                    cv2.destroyWindow(self.window_name)
                    break

    def __call__(self, network, img):
        
        score_thr = 0.3
        nn_input_map = (0.0, 0.0, 1.0, 1.0)

        dnn_input_img = network[0].input_tensors[0].data.copy()
        if dnn_input_img is None:
            raise Exception("Input tensor is None")

        if dnn_input_img.shape[0] == 3:
            w, h, c = dnn_input_img.shape[1], dnn_input_img.shape[2], dnn_input_img.shape[0]
            dnn_input_img = dnn_input_img.transpose(2, 0, 1).reshape(c, h, w).transpose(1, 2, 0)
        else:
            w, h, c = dnn_input_img.shape[1], dnn_input_img.shape[0], dnn_input_img.shape[2]
        dnn_input_img = cv2.cvtColor(dnn_input_img, cv2.COLOR_RGB2BGR)

        if network[0].output_tensors[0].data is None:
            return img, dnn_input_img

        boxes = np.array(network[0].output_tensors[0].data)
        confs = np.array(network[0].output_tensors[1].data)
        cls_ids = np.array(network[0].output_tensors[2].data)
        valid_data_items_num = np.array(network[0].output_tensors[3].data).astype(np.int32)[0]

        boxes = boxes.astype(np.int32)[:valid_data_items_num, :]
        confs = confs[:valid_data_items_num]
        cls_ids = cls_ids.astype(np.int32)[:valid_data_items_num]

        confs_mask = confs > score_thr
        boxes = boxes[confs_mask, :]
        confs = confs[confs_mask]
        cls_ids = cls_ids[confs_mask]

        cls_ids_mask = cls_ids == 1
        boxes = boxes[cls_ids_mask, :]
        confs = confs[cls_ids_mask]
        cls_ids = cls_ids[cls_ids_mask]

        img_h, img_w = img.shape[:2]
        xmin, ymin, xmax, ymax = nn_input_map
        xmin_abs = int(xmin * img_w); ymin_abs = int(ymin * img_h)
        xmax_abs = int(xmax * img_w); ymax_abs = int(ymax * img_h)
        crop_w = xmax_abs - xmin_abs; crop_h = ymax_abs - ymin_abs

        dnn_input_h, dnn_input_w = dnn_input_img.shape[:2]
        x_scale = crop_w / dnn_input_w
        y_scale = crop_h / dnn_input_h

        boxes_yuv = boxes.astype(np.float32)
        boxes_yuv[:, 0] = boxes_yuv[:, 0] * x_scale + xmin_abs
        boxes_yuv[:, 2] = boxes_yuv[:, 2] * x_scale + xmin_abs
        boxes_yuv[:, 1] = boxes_yuv[:, 1] * y_scale + ymin_abs
        boxes_yuv[:, 3] = boxes_yuv[:, 3] * y_scale + ymin_abs
        boxes_yuv = boxes_yuv.astype(np.int32)

        if self._fence_poly is None:
            try:
                self._ensure_fence_interactive(img)
            except Exception:
                self._fence_poly = None
                self._fence_active = False

        if len(confs) > 0:
            dnn_input_img = self.draw_detections(
                image=dnn_input_img, boxes=boxes, confs=confs, class_ids=cls_ids
            )

        img_out = img.copy()
        if len(confs) > 0:
            img_out = self.draw_detections(
                image=img_out, boxes=boxes_yuv, confs=confs, class_ids=cls_ids
            )

        intruded = False
        intruder_count = 0

        if self._fence_active and self._fence_poly is not None and self._setup_size is not None:
            setup_w, setup_h = self._setup_size
            cur_h, cur_w = img.shape[:2]
            fence = self._fence_poly.copy()
            fence[:, 0] *= (cur_w / setup_w)
            fence[:, 1] *= (cur_h / setup_h)
            fence = fence.astype(np.float32)

            cv2.polylines(img_out, [fence.astype(np.int32)], True, CLR_GREEN, 2, cv2.LINE_AA)

            for i in range(boxes_yuv.shape[0]):
                x1, y1, x2, y2 = boxes_yuv[i]
                if poly_rect_intersect(fence, (float(x1), float(y1), float(x2), float(y2))):
                    intruder_count += 1

            if intruder_count > 0:
                intruded = True

        self.current_intruder_count = intruder_count

        draw_warning_light(img_out, intruded, intruder_count)

        return img_out, dnn_input_img
