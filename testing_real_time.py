import cv2
from ultralytics import YOLO
import logging
from tkinter import Tk, Label, StringVar

# Suppress ultralytics logging
logging.getLogger("ultralytics").setLevel(logging.WARNING)

# Load your trained YOLOv8 model
model = YOLO('/home/gafur/Documents/senior-design/code/local_env/runs_strawberry_chobani_300/runs/detect/train/weights/best.pt')

# Nutritional facts data
nutritional_facts = {
    "strawberry-chobani": {"Calories": "140 kcal", "Protein": "12g", "Fat": "2g"},
    "protein-chobani": {"Calories": "190 kcal", "Protein": "20g", "Fat": "5g"},
}

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.5

# Function to perform object detection
def perform_object_detection():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return {}

    detected_objects = {}

    print("Starting object detection. Press 'q' to finish detection.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Use YOLO model for object detection
        results = model(frame)
        detections = results[0].boxes.data if results[0].boxes is not None else []

        # Parse detections
        for det in detections:
            class_id = int(det[5])  # Class ID
            confidence = det[4].item()  # Confidence score
            object_name = model.names[class_id]  # Get object name

            if confidence >= CONFIDENCE_THRESHOLD:
                if object_name not in detected_objects:
                    detected_objects[object_name] = nutritional_facts.get(object_name, "N/A")

        # Annotate and display the frame
        annotated_frame = results[0].plot()
        cv2.imshow("YOLOv8 Detection", annotated_frame)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return detected_objects

# Function to launch the GUI
def launch_gui(detected_objects):
    root = Tk()
    root.title("Detected Objects and Nutritional Facts")
    root.geometry("400x300")

    # Labels for GUI
    Label(root, text="Detected Objects:", font=("Arial", 14)).pack(pady=10)
    for obj, facts in detected_objects.items():
        Label(root, text=obj, font=("Arial", 12), fg="blue").pack()
        if isinstance(facts, dict):
            for key, value in facts.items():
                Label(root, text=f"{key}: {value}", font=("Arial", 10)).pack()
        else:
            Label(root, text="Nutritional facts not available.", font=("Arial", 10)).pack()
        Label(root, text="").pack()  # Empty line for spacing

    root.mainloop()

# Main process
if __name__ == "__main__":
    # Step 1: Perform object detection
    detected_objects = perform_object_detection()

    # Step 2: Launch GUI with detected objects
    if detected_objects:
        launch_gui(detected_objects)
    else:
        print("No objects detected.")
