from flask import Flask, render_template, jsonify, request
import threading
import main

app = Flask(__name__)

detection_running = False
detection_lock = threading.Lock()

def start_detection():
    """Runs the detection process in a separate thread to avoid blocking Flask"""
    global detection_running
    main.process_frame()
    with detection_lock:
        detection_running = False

@app.route('/detect')
def startdetect():
    """Starts detection only if not already running"""
    global detection_running
    with detection_lock:
        if not detection_running:
            detection_running = True
            threading.Thread(target=start_detection, daemon=True).start()
    return render_template('index.html')

@app.route('/')
def index():
    return render_template('homepage.html')

@app.route('/locations', methods=['GET'])
def get_locations():
    return jsonify(main.locations)


if __name__ == '__main__':
    app.run(port=5000, debug=True)
