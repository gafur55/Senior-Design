import cv2
from ultralytics import YOLO

# Load your trained YOLOv8 model
model = YOLO('/home/gafur/Documents/senior-design/code/local_env/runs_strawberry_chobani_300/runs/detect/train/weights/best.pt')
model = YOLO('/home/gafur/Documents/senior-design/code/local_env/runs_strawberry_chobani_300/runs/detect/train/weights/best.pt')

# Initialize camera (0 is typically the default camera, adjust if necessary)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame.")
        break

    # Use the model to perform object detection
    results = model(frame)

    # Render results directly on the frame (bounding boxes, labels, etc.)
    annotated_frame = results[0].plot()

    # Display the resulting frame
    cv2.imshow('YOLOv8 Detection', annotated_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
