# Copyright 2025 Subhajeet Roy & Winix Technologies Pvt Ltd

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#      http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import cv2
import ultralytics
from ultralytics import YOLO

from app_modules.global_config_gui import global_config



def detect_object_in_image_yolo(image_path: str, output_path: str = None, model_name: str = "yolov8n.pt"):
    print("YOLO version:", ultralytics.__version__)
    TARGET_LABELS = global_config['selected_labels']
    print(TARGET_LABELS)
    model = YOLO(model_name)

    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error: Could not load image from {image_path}")
        return

    results = model(image)[0]  # Run inference
    annotated = image.copy()

    for *box, score, cls in results.boxes.data.tolist():
        label = model.names[int(cls)].lower()
        if label not in TARGET_LABELS:
            continue
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, f"{label} {score:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if output_path is None:
        output_path = image_path.rsplit('.', 1)[0] + "_yolo_detected.jpg"
    cv2.imwrite(output_path, annotated)
    print(f"✅ Saved output to: {output_path}")




def detect_object_in_video_yolo(input_video_path: str, output_video_path: str, model_name: str = "yolov8n.pt"):
    print("YOLO version:", ultralytics.__version__)
    TARGET_LABELS = global_config['selected_labels']
    print(TARGET_LABELS)
    model = YOLO(model_name)

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("❌ Error: Cannot open video file.")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        annotated = frame.copy()

        for *box, score, cls in results.boxes.data.tolist():
            label = model.names[int(cls)].lower()
            if label not in TARGET_LABELS:
                continue
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(annotated, f"{label} {score:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        out.write(annotated)
        frame_count += 1

    cap.release()
    out.release()
    print(f"✅ Processed {frame_count} frames.")
    print(f"🎥 Output saved to: {output_video_path}")
