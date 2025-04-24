from flask import Flask, render_template, Response, jsonify
import cv2
from ultralytics import YOLO
import threading
import numpy as np
import csv
import os
from datetime import datetime
import time

app = Flask(__name__)

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Use test video (30 fps)
cap = cv2.VideoCapture(1)
fps = 30
frame_interval = 1.0 / fps

# Shared state
dot_data = []
heatmap_grid = []
density = 0
real_density = 0
people_count = 0
lock = threading.Lock()

# Homography (image -> real world)
image_pts = np.array([[100, 100], [540, 100], [540, 380], [100, 380]], dtype='float32')
real_pts = np.array([[0, 0], [10, 0], [10, 5], [0, 5]], dtype='float32')  # 50 m²
H, _ = cv2.findHomography(image_pts, real_pts)
real_area_m2 = 50

# CSV log setup
LOG_FILE = 'logs.csv'
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'PeopleCount', 'PixelDensity', 'RealDensity'])

def log_data(timestamp, people_count, density, real_density):
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, people_count, density, real_density])

def generate_frames():
    global dot_data, heatmap_grid, density, real_density, people_count
    last_time = time.time()

    while True:
        current_time = time.time()
        if current_time - last_time < frame_interval:
            continue
        last_time = current_time

        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
            continue

        results = model(frame)
        dots = []
        grid_rows, grid_cols = 10, 10
        grid = [[0 for _ in range(grid_cols)] for _ in range(grid_rows)]

        for result in results:
            if hasattr(result, 'boxes'):
                for box in result.boxes:
                    if int(box.cls[0]) != 0 or box.conf[0] < 0.5:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0]
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    dots.append({"x": cx, "y": cy})
                    cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

                    gx = min(grid_cols - 1, cx * grid_cols // frame.shape[1])
                    gy = min(grid_rows - 1, cy * grid_rows // frame.shape[0])
                    grid[gy][gx] += 1

        px_density = len(dots) / (frame.shape[0] * frame.shape[1])
        real_density_val = len(dots) / real_area_m2
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        with lock:
            dot_data = dots
            heatmap_grid = grid
            density = px_density
            real_density = real_density_val
            people_count = len(dots)
            log_data(timestamp, people_count, density, real_density)

        height, width = frame.shape[:2]
        for i in range(1, grid_rows):
            y = int(i * height / grid_rows)
            cv2.line(frame, (0, y), (width, y), (0, 255, 0), 1)
        for j in range(1, grid_cols):
            x = int(j * width / grid_cols)
            cv2.line(frame, (x, 0), (x, height), (0, 255, 0), 1)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index3graphs.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def get_stats():
    with lock:
        return jsonify({
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "count": people_count,
            "density": density,
            "real_density": real_density,
            "dots": dot_data,
            "heatmap": heatmap_grid
        })

@app.route('/dots')
def get_dots():
    with lock:
        return jsonify(dot_data)

@app.route('/export_logs')
def export_logs():
    with open(LOG_FILE, 'r') as f:
        content = f.read()
    return Response(content, mimetype='text/csv',
                    headers={"Content-Disposition": "attachment;filename=logs.csv"})

if __name__ == '__main__':
    app.run(debug=True)
