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

from app_modules.global_config_gui import create_gui, global_config
from app_modules.object_detection_efficient_det_lite2 import detect_object_in_image, detect_object_in_video
from app_modules.object_detection_yolo import detect_object_in_image_yolo, detect_object_in_video_yolo
from app_modules.pose_landmarker_estimation import process_video_with_pose_estimation, \
    process_image_with_pose_estimation
from app_modules.utils import get_file_name_from_path

# Run the GUI
if __name__ == "__main__":
    create_gui()

# After the GUI closes, you can still access global_config
print("\nFinal Global Configuration after GUI closed:", global_config)

# Example of how you might use the config later in your app
def perform_action_based_on_config():
    print("\n--- Performing Action Based on Configuration ---")
    if global_config["cv_task"] == "Pose Estimation and Landmark tracking":
        print(f"Task: Pose Estimation. Model: {global_config['model_type']}. Sensitivity: {global_config['detection_sensitivity']}")
        if global_config["input_source"] == "Select image from gallery to analyze" and global_config["file_path"]:
            print(f"Analyzing image: {global_config['file_path']}")
            file_name = get_file_name_from_path(global_config['file_path'])
            process_image_with_pose_estimation(global_config['file_path'],"./generated/"+file_name)
        elif global_config["input_source"] == "Select video from gallery" and global_config["file_path"]:
            print(f"Analyzing video: {global_config['file_path']}")
            file_name=get_file_name_from_path(global_config['file_path'])
            process_video_with_pose_estimation(global_config['file_path'],"./generated/"+file_name)
        elif global_config["input_source"] == "Live video":
            print("Live video selected (currently not supported for processing).")
        else:
            print("No valid input source or file selected for Pose Estimation.")
    elif global_config["cv_task"] == "Object Detection Object tracking":
        print(
            f"Task: Object Detection. Model: {global_config['model_type']}. Sensitivity: {global_config['detection_sensitivity']}")
        if global_config['model_type']=="EfficientDet-Lite":

            if global_config["input_source"] == "Select image from gallery to analyze" and global_config["file_path"]:
                print(f"Analyzing image: {global_config['file_path']}")
                file_name = get_file_name_from_path(global_config['file_path'])
                detect_object_in_image(global_config['file_path'],"./generated/"+file_name)
            elif global_config["input_source"] == "Select video from gallery" and global_config["file_path"]:
                print(f"Analyzing video: {global_config['file_path']}")
                file_name = get_file_name_from_path(global_config['file_path'])
                detect_object_in_video(global_config['file_path'], "./generated/" + file_name)
            elif global_config["input_source"] == "Live video":
                print("Live video selected (currently not supported for processing).")
            else:
                print("No valid input source or file selected for Object Detection.")
        elif global_config['model_type'] == "YOLO":
            if global_config["input_source"] == "Select image from gallery to analyze" and global_config["file_path"]:
                print(f"Analyzing image: {global_config['file_path']}")
                file_name = get_file_name_from_path(global_config['file_path'])
                detect_object_in_image_yolo(global_config['file_path'],"./generated/"+file_name)
            elif global_config["input_source"] == "Select video from gallery" and global_config["file_path"]:
                print(f"Analyzing video: {global_config['file_path']}")
                file_name = get_file_name_from_path(global_config['file_path'])
                detect_object_in_video_yolo(global_config['file_path'], "./generated/" + file_name)
            elif global_config["input_source"] == "Live video":
                print("Live video selected (currently not supported for processing).")
            else:
                print("No valid input source or file selected for Object Detection.")

    else:
        print("No specific CV task defined.")

# Call an example function to demonstrate using the saved config
perform_action_based_on_config()
