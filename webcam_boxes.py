import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture(0)
model = YOLO("yolov8m.pt") 


while True:
    ret, frame = cap.read()
    result = model(frame)[0]

    for box in result.boxes:
        class_id = result.names[box.cls[0].item()]
        cords = box.xyxy[0].tolist()
        cords = [round(x) for x in cords]
        conf = round(box.conf[0].item(), 2)

        message = f"{class_id}, {cords}, {conf}"
        image = cv2.putText(frame, message , (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

        start = cords[0:2]  # x1,y1
        end = cords[2:4]  # x2,y2
        image = cv2.rectangle(frame, start, end, (255, 0, 0), 2)

        # Display the frame
        cv2.imshow('result image', image)
        # Press `q` to quit
        if cv2.waitKey(1) == ord("q"):
            cap.release()
            cv2.destroyAllWindows()

