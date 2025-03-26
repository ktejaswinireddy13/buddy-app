from flask import Flask, request, Response, jsonify
import requests
import time
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
import cv2
import numpy as np

app = Flask(__name__)

# Load YOLOv11s model
model = YOLO("model/yolo11s.pt")

GOOGLE_MAPS_API_KEY = "AIzaSyBQJ9Sq-jTvJ45e_JWE5UELySDiqMadzXY"

def clean_html(text):
    import re
    return re.sub(r'<.*?>', '', text)

def convert_to_steps(distance_text):
    import re
    match = re.search(r'([\d\.]+)\s*(m|mi|ft)', distance_text)
    if not match:
        return None

    value, unit = float(match.group(1)), match.group(2)

    if unit == "m":
        steps = int(value * 1.5)
    elif unit == "ft":
        steps = int(value / 2)
    elif unit == "mi":
        steps = int(value * 1609 * 1.5)

    return steps

def get_directions(origin, destination):
    url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {
        "origin": origin,
        "destination": destination,
        "mode": "walking",
        "key": GOOGLE_MAPS_API_KEY
    }
    response = requests.get(url, params=params)
    data = response.json()

    if data.get("status") == "OK":
        steps = []
        for leg in data["routes"][0]["legs"]:
            for step in leg["steps"]:
                instruction = clean_html(step["html_instructions"])
                distance = step["distance"]["text"]
                step_count = convert_to_steps(distance)

                if "left" in instruction.lower():
                    direction = "Turn slightly to your left."
                elif "right" in instruction.lower():
                    direction = "Turn slightly to your right."
                else:
                    direction = "Keep walking straight."

                if step_count:
                    steps.append({"instruction": direction, "steps": step_count})

        return steps
    return None

def generate_frames():
    cap = cv2.VideoCapture(0)  # Use webcam (Change this for mobile streaming)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Run YOLO on frame
        results = model(frame)
        
        # Draw bounding boxes
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = result.names[int(box.cls)]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, cls, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Convert frame to JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/navigate", methods=["GET", "POST"])
def navigate():
    image = request.files.get("image")
    destination = request.form.get("destination")

    if not destination:
        return jsonify({"error": "Destination not provided"}), 400

    origin = "New York"
    steps = get_directions(origin, destination)

    if not steps:
        return jsonify({"error": "Could not fetch directions"}), 500

    navigation_steps = []
    if image:
        image = Image.open(BytesIO(image.read()))
        obstacle, processed_image = detect_obstacle(image)

        if obstacle:
            navigation_steps.append({"instruction": f"Obstacle detected: {', '.join(obstacle)}. Adjust your path."})

    navigation_steps += steps

    return jsonify({"steps": navigation_steps, "processed_image": processed_image})


if __name__ == "__main__":
    app.run(debug=True)
