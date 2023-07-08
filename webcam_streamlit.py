''' 
streamlit run webcam_streamlit.py --server.headless true --server.port 8888
'''

import av
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoHTMLAttributes, WebRtcMode
from ultralytics import YOLO

model = YOLO("yolov8m.pt") 
# model = YOLO("best.pt") 
st.title("Live Detection")

def callback(frame):
    frame = frame.to_ndarray(format="bgr24")

    result = model(frame)[0]
    for box in result.boxes:
        class_id = result.names[box.cls[0].item()]
        cords = box.xyxy[0].tolist()
        cords = [round(x) for x in cords]
        conf = round(box.conf[0].item(), 2)

        start = cords[0:2]  # x1,y1
        end = cords[2:4]  # x2,y2
        image = cv2.rectangle(frame, start, end, (255, 0, 0), 2)

        message = f"{class_id}, {cords}, {conf}"
        image = cv2.putText(frame, message , start , cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(image, format="bgr24")


webrtc_ctx = webrtc_streamer(
    key="live",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
    video_frame_callback=callback
)




