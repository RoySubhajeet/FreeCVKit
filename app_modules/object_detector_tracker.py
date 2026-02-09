# Copyright 2025 Subhajeet Roy & Winix Technologies Pvt Ltd
# Licensed under the Apache License, Version 2.0

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import ultralytics
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from app_modules.centroid_tracker import CentroidTracker
from collections import defaultdict, deque
import cv2
import numpy as np

class ObjectDetector:
    def __init__(self, model_type: str, model_name: str, target_labels: set, score_threshold: float = 0.2):
        """
        Initialize the object detection class.
        :param model_type: 'efficientdet' or 'yolo'
        :param model_name: Model path ('./models/efficientdet-lite2.tflite' or 'yolov8n.pt')
        :param target_labels: Set of labels to detect, e.g., {"person"}
        :param score_threshold: Confidence score threshold for detection
        """
        self.model_type = model_type.lower()
        self.model_name = model_name
        self.target_labels = {label.lower() for label in target_labels}
        self.score_threshold = score_threshold

        if self.model_type == "efficientdet":
            base_options = python.BaseOptions(model_asset_path=self.model_name)
            options = vision.ObjectDetectorOptions(
                base_options=base_options,
                score_threshold=self.score_threshold
            )
            self.detector = vision.ObjectDetector.create_from_options(options)
            print(f"✅ Loaded EfficientDet model from {self.model_name}")

        elif self.model_type == "yolo":
            print("YOLO version:", ultralytics.__version__)
            self.detector = YOLO(self.model_name)
            print(f"✅ Loaded YOLO model from {self.model_name}")

        else:
            raise ValueError("❌ Unsupported model type. Use 'efficientdet' or 'yolo'.")

    def detect_image(self, image_path: str, output_path: str = None):
        """
        Detect objects in a single image and save output.
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Error: Could not load image from {image_path}")
            return

        annotated = image.copy()

        if self.model_type == "efficientdet":
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            detection_result = self.detector.detect(mp_image)

            for detection in detection_result.detections:
                category = detection.categories[0]
                label = category.category_name.lower()
                score = category.score
                if label not in self.target_labels:
                    continue
                bbox = detection.bounding_box
                start_point = (int(bbox.origin_x), int(bbox.origin_y))
                end_point = (int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height))
                cv2.rectangle(annotated, start_point, end_point, (0, 255, 0), 2)
                cv2.putText(annotated, f"{label} {score:.2f}", (start_point[0], start_point[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        elif self.model_type == "yolo":
            results = self.detector(image)[0]
            for *box, score, cls in results.boxes.data.tolist():
                label = self.detector.names[int(cls)].lower()
                if label not in self.target_labels:
                    continue
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, f"{label} {score:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if output_path is None:
            suffix = "_yolo_detected" if self.model_type == "yolo" else "_mp_detected"
            output_path = image_path.rsplit('.', 1)[0] + suffix + ".jpg"

        cv2.imwrite(output_path, annotated)
        print(f"✅ Saved output to: {output_path}")

    def detect_video(self, input_video_path: str, output_video_path: str):
        """
        Detect objects in a video and save annotated output.
        """
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print("❌ Error: Cannot open video file.")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated = frame.copy()

            if self.model_type == "efficientdet":
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                detection_result = self.detector.detect(mp_image)

                for detection in detection_result.detections:
                    category = detection.categories[0]
                    label = category.category_name.lower()
                    score = category.score
                    if label not in self.target_labels:
                        continue
                    bbox = detection.bounding_box
                    start_point = (int(bbox.origin_x), int(bbox.origin_y))
                    end_point = (int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height))
                    cv2.rectangle(annotated, start_point, end_point, (0, 255, 255), 2)
                    cv2.putText(annotated, f"{label} {score:.2f}", (start_point[0], start_point[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            elif self.model_type == "yolo":
                results = self.detector(frame)[0]
                for *box, score, cls in results.boxes.data.tolist():
                    label = self.detector.names[int(cls)].lower()
                    if label not in self.target_labels:
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


    def track_video_centroid_algo(self, input_video_path: str, output_video_path: str):
        """
        Track objects in a video using Centroid Tracking.
        Adds:
        - A horizontal line at the middle
        - Counters for up and down movements
        - Path/trajectory drawing for each tracked object
        """
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print("❌ Error: Cannot open video file.")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        # Initialize tracker and helpers
        tracker = CentroidTracker(maxDisappeared=40, maxDistance=50)
        frame_count = 0

        line_y = height // 2  # middle horizontal line
        total_up = 0
        total_down = 0

        # Store object paths (last 30 centroids per ID)
        object_paths = defaultdict(lambda: deque(maxlen=30))
        object_last_y = {}

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated = frame.copy()
            rects = []

            # --- Detection Phase ---
            if self.model_type == "yolo":
                results = self.detector(frame)[0]
                for *box, score, cls in results.boxes.data.tolist():
                    label = self.detector.names[int(cls)].lower()
                    if label not in self.target_labels:
                        continue
                    x1, y1, x2, y2 = map(int, box)
                    rects.append((x1, y1, x2, y2))
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)

            elif self.model_type == "efficientdet":
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                detection_result = self.detector.detect(mp_image)
                for detection in detection_result.detections:
                    category = detection.categories[0]
                    label = category.category_name.lower()
                    if label not in self.target_labels:
                        continue
                    bbox = detection.bounding_box
                    x1, y1 = int(bbox.origin_x), int(bbox.origin_y)
                    x2, y2 = int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height)
                    rects.append((x1, y1, x2, y2))
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)

            # --- Tracking Phase ---
            objects = tracker.update(rects)

            # Draw the horizontal middle line
            cv2.line(annotated, (0, line_y), (width, line_y), (0, 0, 255), 2)

            # Loop through tracked objects
            for objectID, centroid in objects.items():
                cx, cy = centroid
                object_paths[objectID].append((cx, cy))

                # Check if object crossed the line
                if objectID in object_last_y:
                    prev_y = object_last_y[objectID]
                    if prev_y < line_y and cy >= line_y:
                        total_down += 1
                    elif prev_y > line_y and cy <= line_y:
                        total_up += 1
                object_last_y[objectID] = cy

                # Draw ID and centroid
                cv2.putText(annotated, f"ID {objectID}", (cx - 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(annotated, (cx, cy), 4, (0, 255, 0), -1)

                # Draw trajectory path
                path_points = list(object_paths[objectID])
                for i in range(1, len(path_points)):
                    if path_points[i - 1] is None or path_points[i] is None:
                        continue
                    cv2.line(annotated, path_points[i - 1], path_points[i], (255, 0, 0), 2)

            # Display counters at top
            cv2.putText(annotated, f"Up: {total_up}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated, f"Down: {total_down}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            out.write(annotated)
            frame_count += 1

        cap.release()
        out.release()
        print(f"✅ Processed {frame_count} frames with tracking.")
        print(f"🎥 Output saved to: {output_video_path}")
        print(f"⬆️ Total Up: {total_up}, ⬇️ Total Down: {total_down}")

    def track_video_deep_sort_algo(self, input_video_path: str, output_video_path: str):
        """
        Track objects using DeepSORT in a video with up/down and left/right counting.
        """
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print("❌ Error: Cannot open video file.")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        # Initialize DeepSORT
        tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0, max_cosine_distance=0.3)

        # Horizontal and vertical reference lines
        line_y = height // 2  # horizontal line in the middle
        line_x = width // 2   # vertical line in the middle

        count_up, count_down = 0, 0
        count_left, count_right = 0, 0
        track_memory = {}

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated = frame.copy()
            detections = []

            # --------- DETECTION PHASE ----------
            if self.model_type == "yolo":
                results = self.detector(frame)[0]
                for *box, score, cls in results.boxes.data.tolist():
                    label = self.detector.names[int(cls)].lower()
                    if label not in self.target_labels:
                        continue
                    x1, y1, x2, y2 = map(int, box)
                    detections.append(([x1, y1, x2 - x1, y2 - y1], score, label))

            elif self.model_type == "efficientdet":
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                detection_result = self.detector.detect(mp_image)
                for detection in detection_result.detections:
                    category = detection.categories[0]
                    label = category.category_name.lower()
                    if label not in self.target_labels:
                        continue
                    bbox = detection.bounding_box
                    x1, y1 = int(bbox.origin_x), int(bbox.origin_y)
                    x2, y2 = int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height)
                    score = category.score
                    detections.append(([x1, y1, x2 - x1, y2 - y1], score, label))

            # --------- TRACKING PHASE ----------
            tracks = tracker.update_tracks(detections, frame=frame)

            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                l, t, w, h = track.to_ltwh()
                cx, cy = int(l + w / 2), int(t + h / 2)

                # Draw bounding box and ID
                cv2.rectangle(annotated, (int(l), int(t)), (int(l + w), int(t + h)), (0, 255, 0), 2)
                cv2.putText(annotated, f"ID {track_id}", (int(l), int(t) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Track path
                if track_id not in track_memory:
                    track_memory[track_id] = []
                track_memory[track_id].append((cx, cy))
                for i in range(1, len(track_memory[track_id])):
                    cv2.line(annotated, track_memory[track_id][i - 1],
                             track_memory[track_id][i], (255, 0, 0), 2)

                # Count Up/Down movements
                if len(track_memory[track_id]) >= 2:
                    prev_y = track_memory[track_id][-2][1]
                    curr_y = track_memory[track_id][-1][1]
                    if prev_y < line_y <= curr_y:
                        count_down += 1
                    elif prev_y > line_y >= curr_y:
                        count_up += 1

                # Count Left/Right movements
                if len(track_memory[track_id]) >= 2:
                    prev_x = track_memory[track_id][-2][0]
                    curr_x = track_memory[track_id][-1][0]
                    if prev_x < line_x <= curr_x:
                        count_right += 1
                    elif prev_x > line_x >= curr_x:
                        count_left += 1

            # Draw horizontal and vertical lines
            cv2.line(annotated, (0, line_y), (width, line_y), (0, 0, 255), 2)  # horizontal
            cv2.line(annotated, (line_x, 0), (line_x, height), (255, 0, 0), 2)  # vertical

            # Display counts
            cv2.putText(annotated, f"Up: {count_up}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(annotated, f"Down: {count_down}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(annotated, f"Left: {count_left}", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(annotated, f"Right: {count_right}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            out.write(annotated)

        cap.release()
        out.release()
        print(f"✅ Processed video saved at: {output_video_path}")



