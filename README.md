# FreeCVKit: A Free & Open-Source Python Toolkit for Real-time Computer Vision 

**FreeCVKit** is a powerful and easy-to-use Python library designed to streamline real-time computer vision tasks. Built on the foundational power of Google's MediaPipe and Yolov8n, it provides a simple API for common applications like **pose estimation** and **object detection**. By handling the complex back-end operations, **FreeCVKit** allows developers, researchers, and hobbyists to quickly and efficiently add advanced vision capabilities to their projects, with a focus on clear visualization of results.

## Key Features

- **Real-time Processing:** Optimized for high-performance, low-latency processing of live video streams and webcam feeds.
- **MediaPipe Integration:** Direct and simplified access to state-of-the-art MediaPipe models, including:
  - **Pose Estimation & Landmark Tracking:** Accurately detects and tracks human skeletal landmarks, excluding facial landmarks and their connections for cleaner body-focused analysis.
  - **Object Detection & Tracking:** Utilizes efficient models (e.g., EfficientDet-Lite, YOLO) to identify and track specified objects in frames.
- **Intuitive Visualization:** Automatically draws keypoint landmarks, connecting lines, and bounding boxes directly onto images and video frames.
- **Flexible I/O:** Supports a wide range of inputs, including video files (`.mp4`), static images (`.jpg`, `.png`), and a placeholder for future live video integration.
- **Configurable via GUI:** An interactive Tkinter GUI allows users to easily set up CV tasks, model types, detection sensitivity, input sources, and object detection labels.
- **Clean & Modular API:** Designed for seamless integration. Easily import core functions to build your own custom applications.

## Getting Started

Follow these steps to get **FreeCVKit** up and running on your local machine.

### Prerequisites

- Python 3.10
- `pip` package installer

### Installation

1. **Clone the repository:**

   ```
   git clone https://github.com/RoySubhajeet/FreeCVKit.git
   cd FreeCVKit
   ```

2. **Install dependencies:**

   ```
   pip install -r requirements.txt
   ```

   *(Note: `tk` is usually included with Python, but ensure it's available for `tkinter`.)*

## Configuration via GUI

**FreeCVKit** includes a user-friendly graphical interface to configure your computer vision task.

1. **Run the configuration GUI:**

   ```
   python main.py
   ```

   *(Replace `your_gui_script_name.py` with the actual name of your Python file containing the `create_gui()` function.)*

2. **Select your options:**

   - **1) Select CV Task:** Choose between main_options[0] or main_options[1]
   - **2) Select Model Type:** This dropdown will dynamically update based on your CV Task selection (e.g., "MediaPipe Pose landmarker" for Pose Estimation, or "EfficientDet-Lite", "YOLO" for Object Detection).
   - **Select Labels to Filter:** (Appears only for main_options[1]) Choose up to four labels like "person", "car", "traffic signal", "dog". "person" is pre-selected by default.
   - **Select Detection Sensitivity:** Adjust the sensitivity of the detection model.
   - **3) Select Input Source:** Choose to analyze an image or video from your gallery, or select "Live video" (currently not supported).
     - If "Select image from gallery to analyze" or "Select video from gallery" is chosen, a file browser will open for you to pick your input file. The selected file path will be displayed.

3. **Save Configuration:** Click the "Save Configuration" button. The dialog will close, and your selections will be saved into the `global_config` dictionary, ready for use by the processing functions. All fields are validated to ensure no empty selections.

## 

## Technical Stack

- **Python**
- **MediaPipe**
- **OpenCV**
- **NumPy**
- **Tkinter** (for GUI)

## 📄 License

This project is licensed under the **Apache License, Version 2.0**. See the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

```
Copyright 2025 Subhajeet Roy & Winix Technologies Pvt Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

## About the Creator

**Subhajeet Roy** is a  CTO and technical solutions architect with over 21 years of experience in AI, Computer Vision, and Embedded Systems. He has a proven track record of leading the development of complex, real-world applications, including AI-powered pose estimation platforms and custom Edge AI frameworks. **FreeCVKit** is his contribution to the open-source community, sharing valuable tools for developers to accelerate their computer vision projects.

Connect with me on [LinkedIn](https://www.linkedin.com/in/subhajeetroy/) and explore more projects on [GitHub](https://github.com/RoySubhajeet). For more about Winix Technologies, visit https://winixtechnologies.com.

## Contribution & Support

We welcome and appreciate all contributions! Whether you want to fix a bug, add a new feature, or improve the documentation, please check out our [Contributing Guidelines](https://www.google.com/search?q=CONTRIBUTING.md) for more details.
