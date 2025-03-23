import cv2
import numpy as np
import tensorflow as tf
import time
import threading
import geo
import firebase_admin
from firebase_admin import credentials

# Load the TFLite model (only once)
interpreter = tf.lite.Interpreter(model_path="best_float16.tflite")
interpreter.allocate_tensors()

# Retrieve input/output tensor details (only once)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
input_size = (input_shape[1], input_shape[2])  # (height, width)
print(input_details)

# Global flag to trigger background task (avoid multiple threads for each detection)
triggered = False
trigger_lock = threading.Lock()

def preprocess_frame(frame):
    """Preprocess frame: resize, normalize, and ensure correct shape"""
    img_resized = cv2.resize(frame, (input_shape[1], input_shape[2]))
    img_resized = img_resized.astype(np.float32) / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)
    return img_resized

def triggred():
    """Background task for pothole detection"""
    print("Pothole")

def trigger_in_background():
    """Trigger background task only once to avoid multiple triggers"""
    global triggered
    if not triggered:
        triggered = True
        threading.Thread(target=triggred).start()

def reset_trigger():
    """Reset trigger flag after 1 second delay to allow new triggers"""
    global triggered
    time.sleep(1)
    triggered = False

def detect_objects(frame, confidence_threshold=0.75):
    """Run inference and return detections"""
    h, w, _ = frame.shape
    image_data = preprocess_frame(frame)
    
    interpreter.set_tensor(input_details[0]['index'], image_data)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    num_detections = output_data.shape[-1]
    boxes = output_data[0, :4, :].T  # Transposed boxes (x_center, y_center, width, height)
    scores = output_data[0, 4, :]
    classes = output_data[0, 5, :]

    for i in range(num_detections):
        if scores[i] > confidence_threshold:
            # Trigger background task
            trigger_in_background()
            
            # Draw bounding box on detected object (pothole)
            x_center, y_center, width, height = boxes[i]
            x_min = int((x_center - width / 2) * w)
            y_min = int((y_center - height / 2) * h)
            x_max = int((x_center + width / 2) * w)
            y_max = int((y_center + height / 2) * h)

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            label = f"Pothole {int(classes[i])}: {(scores[i] * 100):.2f}"
            cv2.putText(frame, label, (x_min, max(y_min - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

def add_to_firebase():
    geo_location = geo.get_current_location()
    if geo_location:
        lat, lon = geo_location
        cred = credentials.Certificate("firebase-adminsdk.json")
        firebase_admin.initialize_app(cred)
        

def process_frame():
    """Main frame processing loop"""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    while cap.isOpened():
        ret, frame = cap.read()  # 1ms
        if not ret:
            break

        # Detect objects in the frame
        frame = detect_objects(frame)
        
        # Display processed frame
        cv2.imshow("Real-time Detection", frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Start the background thread to reset trigger state
threading.Thread(target=reset_trigger, daemon=True).start()

# Start processing frames
process_frame()
