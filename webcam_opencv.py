import cv2

# Create a VideoCapture object
cap = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Could not open webcam")
    exit()

# Create a window to display the webcam feed
cv2.namedWindow("Webcam")

# Capture frames from the webcam
while True:
    # Capture the current frame
    ret, frame = cap.read()

    # Display the frame
    cv2.imshow("Webcam", frame)

    # Check if the user pressed the ESC key
    if cv2.waitKey(1) == 27:
        break

# Release the resources
cap.release()
cv2.destroyAllWindows()