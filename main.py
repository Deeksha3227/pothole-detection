import cv2
import numpy as np
import tensorflow as tf
import requests
import threading
import time
import firebase_admin
from firebase_admin import credentials, db


cred = credentials.Certificate("cred.json")  
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://potholedetection-4f930-default-rtdb.firebaseio.com/"  # Replace with your database URL
})
ref = db.reference("location")

locations = []

previous_data = None

interpreter = tf.lite.Interpreter(model_path="best_float16.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
input_size = (input_shape[1], input_shape[2])

triggered = False

trigger_lock = threading.Lock()

def get_gps_data(): 
    global previous_data 
    gps_data = ref.get()

    if gps_data:
        latitude = gps_data.get("latitude", 0.0)
        longitude = gps_data.get("longitude", 0.0)

        if previous_data !=  {"lat": latitude, "lon": longitude}:  
            locations.append( {"lat": latitude, "lon": longitude} )
            previous_data = {"lat": latitude, "lon": longitude}
            print(f"New location added: {latitude}, {longitude}")
        else:
            print("Duplicate GPS data, not appending.")
    else:
        print("No GPS data found.")
   

def preprocess_frame(frame):
    """Preprocess frame: resize, normalize, and ensure correct shape"""
    img_resized = cv2.resize(frame, (input_shape[1], input_shape[2]))
    img_resized = img_resized.astype(np.float32) / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)
    return img_resized

def triggeredfun():
    """Background task for pothole detection"""
    get_gps_data()
    reset_trigger()

def trigger_in_background():
    """Trigger background task only once to avoid multiple triggers"""
    global triggered
    with trigger_lock:
        if not triggered:
            triggered = True
            threading.Thread(target=triggeredfun, daemon=True).start()


def reset_trigger():
    """Reset trigger flag after 0.5 seconds without blocking main thread"""
    def reset():
        time.sleep(0.5)
        global triggered
        with trigger_lock:
            triggered = False

    threading.Thread(target=reset, daemon=True).start()



def detect_objects(frame, confidence_threshold=0.75):
    """Run inference and return detections"""
    h, w, _ = frame.shape
    image_data = preprocess_frame(frame)
    
    interpreter.set_tensor(input_details[0]['index'], image_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    num_detections = output_data.shape[-1]
    boxes = output_data[0, :4, :].T
    scores = output_data[0, 4, :]
    classes = output_data[0, 5, :]  


    for i in range(num_detections):
        if scores[i] > confidence_threshold:
            trigger_in_background()
            x_center, y_center, width, height = boxes[i]
            x_min = int((x_center - width / 2) * w)
            y_min = int((y_center - height / 2) * h)
            x_max = int((x_center + width / 2) * w)
            y_max = int((y_center + height / 2) * h)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            label = f"Pothole {int(classes[i])}: {(scores[i] * 100):.2f}"
            cv2.putText(frame, label, (x_min, max(y_min - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

  

def process_frame():
    """Main frame processing loop"""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    while cap.isOpened():
        ret, frame = cap.read()  
        if not ret:
            break
        frame = detect_objects(frame)
        cv2.imshow("Real-time Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
