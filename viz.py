import cv2
import numpy as np

def draw_detections(image_bgr, boxes, labels, scores=None, label_map=None, score_thresh=0.5):
    img = image_bgr.copy()
    for i, box in enumerate(boxes):
        sc = None if scores is None else float(scores[i])
        if sc is not None and sc < score_thresh:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        name = str(labels[i])
        if label_map and labels[i] in label_map:
            name = label_map[labels[i]]
        text = name if sc is None else f"{name}: {sc:.2f}"
        cv2.putText(img, text, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return img
