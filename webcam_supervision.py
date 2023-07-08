''' 
streamlit run webcam_supervision.py --server.headless true --server.port 8888
'''

import av
import cv2
import supervision as sv
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoHTMLAttributes, WebRtcMode
from ultralytics import YOLO

model = YOLO("yolov8m.pt") 
# model = YOLO("best.pt") 
st.title("Live Detection")

box_annotator = sv.BoxAnnotator(
    thickness=2,
    text_thickness=2,
    text_scale=1
)

def callback(frame):
    frame = frame.to_ndarray(format="bgr24")

    result = model(frame)[0]
    detections = sv.Detections.from_yolov8(result)
    labels = [
        f"{model.model.names[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, _ in detections

    ]
    image = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

    x1 = int(detections.xyxy[0][0])
    y1 = int(detections.xyxy[0][1])
    conf = round(detections.confidence.tolist()[0], 2) 
    number_of_detection = len(detections)
    print('-------', x1, y1, conf, number_of_detection, '-------')  
    

    return av.VideoFrame.from_ndarray(image, format="bgr24")


webrtc_ctx = webrtc_streamer(
    key="live",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
    video_frame_callback=callback
)




