from tkinter import Tk, Label, Button, StringVar, Radiobutton
import cv2
from ultralytics import YOLO
import logging

# Suppress ultralytics logging
logging.getLogger("ultralytics").setLevel(logging.WARNING)

# Load your trained YOLOv8 model
model = YOLO('/home/gafur/Documents/senior-design/code/local_env/runs_strawberry_chobani_300/runs/detect/train/weights/best.pt')

# Nutritional facts data
nutritional_facts = {
    "strawberry-chobani": {"Calories": 140, "Protein": 12, "Fat": 2},
    "protein-chobani": {"Calories": 190, "Protein": 20, "Fat": 5},
}

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.5

# Perform object detection
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

# GUI for preference selection
def get_user_preference():
    root = Tk()
    root.title("Choose Your Preference")
    root.geometry("300x200")

    preference = StringVar(value="Calories")

    Label(root, text="Select your preference:", font=("Arial", 14)).pack(pady=10)

    Radiobutton(root, text="Low Calories", variable=preference, value="Calories").pack(anchor="w")
    Radiobutton(root, text="High Protein", variable=preference, value="Protein").pack(anchor="w")
    Radiobutton(root, text="Low Fat", variable=preference, value="Fat").pack(anchor="w")

    def submit_preference():
        root.destroy()

    Button(root, text="Submit", command=submit_preference).pack(pady=10)

    root.mainloop()
    return preference.get()

# GUI to show recommendations
def show_recommendations(detected_objects, preference):
    root = Tk()
    root.title("Detected Products and Recommendation")
    root.geometry("400x400")

    # Display detected products
    Label(root, text="Detected Products:", font=("Arial", 14)).pack(pady=10)
    recommended_product = None
    min_value = float("inf")

    for obj, facts in detected_objects.items():
        Label(root, text=obj, font=("Arial", 12), fg="blue").pack()
        if isinstance(facts, dict):
            for key, value in facts.items():
                Label(root, text=f"{key}: {value}", font=("Arial", 10)).pack()
            # Find the best recommendation
            if preference in facts:
                if facts[preference] < min_value:
                    min_value = facts[preference]
                    recommended_product = obj
        else:
            Label(root, text="Nutritional facts not available.", font=("Arial", 10)).pack()
        Label(root, text="").pack()  # Empty line for spacing

    # Display recommendation
    if recommended_product:
        Label(root, text=f"Recommended Product: {recommended_product}", font=("Arial", 14), fg="green").pack(pady=20)
    else:
        Label(root, text="No suitable recommendation found.", font=("Arial", 14), fg="red").pack(pady=20)

    root.mainloop()

# Main process
if __name__ == "__main__":
    # Step 1: Get user preference
    user_preference = get_user_preference()

    # Step 2: Perform object detection
    detected_objects = perform_object_detection()

    # Step 3: Show recommendations
    if detected_objects:
        show_recommendations(detected_objects, user_preference)
    else:
        print("No objects detected.")
