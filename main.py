# Copyright 2025 Subhajeet Roy & Winix Technologies Pvt Ltd
from app_modules.global_config_gui import ConfigApp
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#      http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from app_modules.object_detector_tracker import ObjectDetector
from app_modules.pose_landmarker_estimation import process_video_with_pose_estimation, \
    process_image_with_pose_estimation
from app_modules.utils import get_file_name_from_path
app = ConfigApp()

# Run the GUI
if __name__ == "__main__":

    app.create_gui()

# After the GUI closes, you can still access global_config
print("\nFinal Global Configuration after GUI closed:", app.config)

def perform_action_based_on_config():
    print("\n--- Performing Action Based on Configuration ---")

    # Ensure defaults if user didn't select explicitly
    if not app.config["model_type"]:
        app.config["model_type"] = app.options["model_type"][app.config["cv_task"]][0]

    if not app.config["tracking_algo"]:
        app.config["tracking_algo"] = app.options["tracking_algo"][app.config["cv_task"]][0]

    cv_task = app.config["cv_task"]
    input_source = app.config["input_source"]
    file_path = app.config["file_path"]
    model_type = app.config["model_type"]
    sensitivity = app.config["detection_sensitivity"]

    print(f"Final Global Configuration after GUI closed: {app.config}")

    # ---------------- Pose Estimation ---------------- #
    if cv_task == app.options["cv_task"][0]:
        print(f"Task: Pose Estimation. Model: {model_type}. Sensitivity: {sensitivity}")

        if input_source == "Select image from gallery to analyze" and file_path:
            print(f"Analyzing image: {file_path}")
            file_name = get_file_name_from_path(file_path)
            process_image_with_pose_estimation(file_path, "./generated/" + file_name)

        elif input_source == "Select video from gallery" and file_path:
            print(f"Analyzing video: {file_path}")
            file_name = get_file_name_from_path(file_path)
            process_video_with_pose_estimation(file_path, "./generated/" + file_name)

        elif input_source == "Live video":
            print("Live video selected (currently not supported for processing).")
        else:
            print("No valid input source or file selected for Pose Estimation.")

    # ---------------- Object Detection ---------------- #
    elif cv_task == app.options["cv_task"][1]:
        print(f"Task: Object Detection. Model: {model_type}. Sensitivity: {sensitivity}")

        if model_type == "EfficientDet-Lite":
            detector = ObjectDetector(model_type="efficientdet",
                                      model_name="./models/efficientdet-lite2.tflite",
                                      target_labels=set(app.config["selected_labels"]))

        elif model_type == "YOLO":
            detector = ObjectDetector(model_type="yolo",
                                      model_name="yolov8n.pt",
                                      target_labels=app.config["selected_labels"])
        else:
            print("Invalid model type for Object Detection.")
            return

        if input_source == "Select image from gallery to analyze" and file_path:
            print(f"Analyzing image: {file_path}")
            file_name = get_file_name_from_path(file_path)
            detector.detect_image(file_path, "./generated/" + file_name)

        elif input_source == "Select video from gallery" and file_path:
            print(f"Analyzing video: {file_path}")
            file_name = get_file_name_from_path(file_path)
            detector.detect_video(file_path, "./generated/" + file_name)

        elif input_source == "Live video":
            print("Live video selected (currently not supported for processing).")
        else:
            print("No valid input source or file selected for Object Detection.")

    # ---------------- Object Tracking ---------------- #
    elif cv_task == app.options["cv_task"][2]:
        print(f"Task: Object Tracking. Model: {model_type}. Tracking Algorithm: {app.config['tracking_algo']}")
        tracking_type = app.config["tracking_algo"]
        if input_source == "Select video from gallery" and file_path:
            print(f"Starting tracking on video: {file_path}")
            print(f"Starting tracking on video: {file_path}")
            file_name = get_file_name_from_path(file_path)

            tracker = ObjectDetector(model_type=model_type,
                                     model_name="yolov8n.pt" if model_type == "YOLO" else "./models/efficientdet-lite2.tflite",
                                     target_labels=app.config["selected_labels"])
            if tracking_type=="Centroid-Tracking":
                tracker.track_video_centroid_algo(file_path, "./generated/" + file_name)
            elif tracking_type=="Deep SORT":
                tracker.track_video_deep_sort_algo(file_path, "./generated/" + file_name)
        else:
            print("No valid input source or file selected for Object Tracking.")

    else:
        print("No specific CV task defined.")



perform_action_based_on_config()
