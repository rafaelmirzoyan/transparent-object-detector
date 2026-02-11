import cv2
from ultralytics import YOLO
import torch

#Transparent object model
transparent_model = YOLO(r"runs/segment/train/weights/best.pt")

#The original COCO model
coco_model = YOLO("yolov8n-seg.pt")
def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0

    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    return inter / (area1 + area2 - inter)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    #Segmentation for the transparent model & coco
    t_results = transparent_model(frame, conf=0.5, verbose=False)[0]
    c_results = coco_model(frame, conf=0.5, verbose=False)[0]
    t_boxes = t_results.boxes  # transparent detections
    c_boxes = c_results.boxes  # COCO detections

    annotated = t_results.plot()

    #Have COCO find the best guess for the transparent object
    for tb in t_boxes:
        tb_xyxy = tb.xyxy.cpu().numpy()[0]   #[x1,y1,x2,y2]
        best_iou = 0.0
        best_cls_name = None

        for cb in c_boxes:
            cb_xyxy = cb.xyxy.cpu().numpy()[0]
            i = iou(tb_xyxy, cb_xyxy)
            if i > best_iou:
                best_iou = i
                cls_idx = int(cb.cls.item())
                best_cls_name = c_results.names[cls_idx]

        if best_iou > 0.3 and best_cls_name is not None:
            #Draw text near the transparent box saying what COCO thinks it is
            x1, y1, x2, y2 = map(int, tb_xyxy)
            cv2.putText(
                annotated,
                f"transparent {best_cls_name}",
                (x1, max(0, y1-10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    cv2.imshow("Transparent + COCO", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#Turn everything off
cap.release()
cv2.destroyAllWindows()
