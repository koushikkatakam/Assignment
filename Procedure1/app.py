from dotenv import load_dotenv
import os
import cv2
import json
import numpy as np
import pandas as pd
from datetime import datetime
from ultralytics import YOLOWorld
import base64
import openai
import re
import tempfile

load_dotenv()  

openai.api_key = os.getenv("OPENAI_API_KEY")


excel_path = r'C:\Users\koush\Desktop\Assignment\Object list.xlsx'
df = pd.read_excel(excel_path)
object_names = df['Objects'].dropna().tolist()

prompt_map = {
    "Agricultural-Areas": "a large agricultural field with crops",
    "Auto": "a small three-wheeled auto rickshaw with a black canopy roof, used for passenger transport in cities",
    "Bridges": "a concrete road bridge over a river or road",
    "Bus": "a public transport bus on the road",
    "Car": "a small to medium-sized four-wheeled sedan or hatchback with curved aerodynamic shape, low ground clearance, closed doors and windows, used for private transport",
    "Electricity-Utilities": "a tall utility pole made of concrete or wood, carrying multiple electric power lines and transformers",
    "Fuel-Stations": "a roadside petrol pump station",
    "Km-Stone": "a distance milestone placed beside the road",
    "Left Turn": "a left turn sign on the road",
    "MedianOpening": "an opening in the road median",
    "Motorcycle": "a motorcycle on the road",
    "Residential-Areas": "a neighborhood with houses and buildings",
    "Rest-Areas": "a highway rest area with benches and trees",
    "Right Turn": "a right turn sign on the road",
    "ServiceRoad": "a narrow parallel road beside a main highway used for local access and slower traffic, usually separated by a divider or curb",
    "Sign-Boards": "a roadside direction signboard",
    "Street-Lights": "a tall street light pole along the road",
    "Traffic Signals": "a traffic light at a road intersection",
    "Truck": "a large delivery truck on the road"
}
prompts = [prompt_map.get(obj, f"a photo of a {obj.lower()}") for obj in object_names]

# --- Load YOLOWorld model ---
model = YOLOWorld(r"C:\Users\koush\Desktop\Assignment\yolov8x-worldv2.pt")
model.set_classes(prompts)

# --- Load video ---
video_path = r'C:\Users\koush\Desktop\Assignment\Videos\video.mp4'
cap = cv2.VideoCapture(video_path)

# --- Output directory ---
output_frame_dir = "output_frames"
os.makedirs(output_frame_dir, exist_ok=True)
detections_log = []

def image_to_base64(img_path):
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# --- Extract OCR info from OpenAI GPT Vision ---
def extract_metadata_openai(image_np):
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            cv2.imwrite(tmp.name, image_np)
            base64_img = image_to_base64(tmp.name)

        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": "Extract the values: LAT, LNG, ALT, GPSTIME, SURVEY START, SPEED, CHAINAGE, DIRECTION as JSON with lowercase keys like: latitude, longitude, altitude, gps_time, survey_start, speed, chainage, direction."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}}
                ]}
            ],
            max_tokens=300
        )

        raw = response.choices[0].message.content.strip()
        print("[GPT OCR]", raw)

        # --- Fix: Remove ```json and triple backticks ---
        clean = re.sub(r"```json|```", "", raw).strip()

        # Convert to dict
        metadata = json.loads(clean)
        return metadata

    except Exception as e:
        print(f"[ERROR] OpenAI OCR failed: {e}")
        return {}

# --- Process video frames ---
frame_index = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_index += 1
    if frame_index % 20 != 0:
        continue

    original_frame = frame.copy()
    height, _, _ = frame.shape
    ocr_crop = frame[:int(height * 0.2), :]

    metadata = extract_metadata_openai(ocr_crop)

    results = model(frame)
    result = results[0]

    for box in result.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls_id = box
        if conf < 0.5:
            continue

        class_id = int(cls_id)
        label = object_names[class_id] if class_id < len(object_names) else f"Class {class_id}"

        cv2.rectangle(original_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(original_frame, f"{label} ({conf:.2f})", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        detections_log.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "object": label,
            "confidence": round(conf, 2),
            "coordinates": [int(x1), int(y1), int(x2), int(y2)],
            "latitude": metadata.get("latitude"),
            "longitude": metadata.get("longitude"),
            "altitude": metadata.get("altitude"),
            "gps_time": metadata.get("gps_time"),
            "survey_start": metadata.get("survey_start"),
            "chainage": metadata.get("chainage"),
            "direction": metadata.get("direction"),
            "speed": metadata.get("speed"),
        })

    cv2.imshow("Detection", original_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.imwrite(os.path.join(output_frame_dir, f"frame_{frame_index}.jpg"), original_frame)
    print(f"[INFO] Frame {frame_index} done, Detections: {len(result.boxes)}")

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()

with open("detections_with_location1.json", "w") as f:
    json.dump(detections_log, f, indent=4)

print("\n✅ Detection complete. Output saved to: detections_with_location.json")
