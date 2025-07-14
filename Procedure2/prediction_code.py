from dotenv import load_dotenv
import os
import cv2
import json
import re
import base64
import tempfile
from datetime import datetime
from ultralytics import YOLO
import openai

# --- Load environment variables ---
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Custom-trained model path ---
model_path = r'C:\Users\koush\Desktop\Assignment_Details\Procedure2\best.pt'  # change to your path
model = YOLO(model_path)

# --- Input video path ---
video_path = r'C:\Users\koush\Desktop\Assignment\Videos\video.mp4'
cap = cv2.VideoCapture(video_path)

# --- Output directory ---
output_frame_dir = "output_frames_v1"
os.makedirs(output_frame_dir, exist_ok=True)

# --- Sample object names (replace with your training classes if needed) ---
object_names = model.names

# --- Convert image to base64 ---
def image_to_base64(img_path):
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# --- Extract OCR metadata using OpenAI ---
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

        # Remove ```json blocks if present
        clean = re.sub(r"```json|```", "", raw).strip()
        return json.loads(clean)

    except Exception as e:
        print(f"[ERROR] OpenAI OCR failed: {e}")
        return {}

# --- Detection + OCR Loop ---
detections_log = []
frame_index = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_index += 1
    if frame_index % 20 != 0:
        continue

    original_frame = frame.copy()
    h, _, _ = frame.shape
    ocr_crop = frame[:int(h * 0.2), :]

    metadata = extract_metadata_openai(ocr_crop)
    results = model(original_frame)
    result = results[0]

    for box in result.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls_id = box
        if conf < 0.5:
            continue

        cls_id = int(cls_id)
        label = object_names[cls_id] if cls_id < len(object_names) else f"Class {cls_id}"

        # Draw
        cv2.rectangle(original_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(original_frame, f"{label} ({conf:.2f})", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Save to log
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
            "speed": metadata.get("speed")
        })

    # Show live
    cv2.imshow("Detection", original_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Save frame
    cv2.imwrite(os.path.join(output_frame_dir, f"frame_{frame_index}.jpg"), original_frame)
    print(f"[INFO] Frame {frame_index} done, Detections: {len(result.boxes)}")

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()

# Save JSON log
with open("detections_with_location.json", "w") as f:
    json.dump(detections_log, f, indent=4)

print("\n✅ Done. Output saved to: detections_with_location.json.json")
