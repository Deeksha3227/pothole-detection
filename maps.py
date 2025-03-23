from flask import Flask, render_template, jsonify

app = Flask(__name__)

# Sample data: Multiple latitude and longitude points



locations = [
    {"name": "Location 1", "lat": 51.5074, "lon": -0.1278},  # London
    {"name": "Location 2", "lat": 48.8566, "lon": 2.3522},   # Paris
    {"name": "Location 3", "lat": 40.7128, "lon": -74.0060},  # New York
    {"name": "Location 4", "lat": 34.0522, "lon": -118.2437}  # Los Angeles
]

@app.route('/')
def index():
    return render_template('index.html', locations=locations)

@app.route('/locations', methods=['GET'])
def get_locations():
    return jsonify(locations)

if __name__ == '__main__':
    app.run(debug=True)
