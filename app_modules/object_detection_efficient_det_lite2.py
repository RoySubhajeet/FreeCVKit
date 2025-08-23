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
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from app_modules.global_config_gui import global_config

# Load the object detection model (EfficientDet Lite2 by default)
MODEL_PATH = './models/efficientdet-lite2.tflite'  # You can use lite0, lite1, etc.


# List of allowed labels


def detect_object_in_image(image_path, output_path=None):
    # Initialize the object detector
    TARGET_LABELS = global_config['selected_labels']
    print(TARGET_LABELS)
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.ObjectDetectorOptions(
        base_options=base_options,
        score_threshold=0.1
    )
    detector = vision.ObjectDetector.create_from_options(options)

    image = cv2.imread(image_path)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

    # Run detection
    detection_result = detector.detect(mp_image)

    for detection in detection_result.detections:
        category = detection.categories[0]
        label = category.category_name.lower()
        score = category.score

        if label not in TARGET_LABELS:
            continue

        # Draw bounding box
        bbox = detection.bounding_box
        start_point = (int(bbox.origin_x), int(bbox.origin_y))
        end_point = (int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height))
        cv2.rectangle(image, start_point, end_point, (0, 255, 0), 2)
        cv2.putText(image, f"{label} {score:.2f}", (start_point[0], start_point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if output_path is None:
        output_path = image_path.replace(".jpg", "_mp_detected.jpg")
    cv2.imwrite(output_path, image)
    print(f"✅ Saved output to: {output_path}")


def detect_object_in_video(input_video_path: str, output_video_path: str):
    TARGET_LABELS = global_config['selected_labels']
    print(TARGET_LABELS)
    # Initialize the object detector
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.ObjectDetectorOptions(
        base_options=base_options,
        score_threshold=0.2
    )
    detector = vision.ObjectDetector.create_from_options(options)

    # Open video file
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("❌ Error: Cannot open video file.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define video writer
    out = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (frame_width, frame_height)
    )

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        detection_result = detector.detect(mp_image)

        for detection in detection_result.detections:
            category = detection.categories[0]
            label = category.category_name.lower()
            score = category.score

            if label not in TARGET_LABELS:
                continue

            bbox = detection.bounding_box
            start_point = (int(bbox.origin_x), int(bbox.origin_y))
            end_point = (int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height))
            cv2.rectangle(frame, start_point, end_point, (0, 255, 255), 2)  # Yellow box
            cv2.putText(frame, f"{label} {score:.2f}", (start_point[0], start_point[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    print(f"✅ Processed {frame_count} frames.")
    print(f"🎥 Output saved to: {output_video_path}")
