import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np

def main():
    frame_width, frame_height = [1280, 720]

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("yolov8m.pt")

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    while True:
        ret, frame = cap.read()

        result = model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_yolov8(result)

        labels = [f"{model.model.names[class_id]} {confidence:0.2f}" for _, _, confidence, class_id, _ in detections]

        frame = box_annotator.annotate(
            scene=frame, 
            detections=detections, 
            labels=labels
        )
            
        cv2.imshow("detected", frame)
        # cv2.imshow("detected2", frame2)

        if (cv2.waitKey(30) == ord('q')):  
            break


if __name__ == "__main__":
    main()