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
import os
from mediapipe.framework.formats import landmark_pb2


# Initialize MediaPipe drawing utilities
mp_drawing = mp.solutions.drawing_utils
# Initialize MediaPipe Pose solution
mp_pose = mp.solutions.pose

# --- Global Definitions for Facial Landmark Exclusion ---
# Define facial landmark indices (Nose, Eyes, Ears, Mouth)
# These indices correspond to the facial landmarks in MediaPipe's pose model.
# MediaPipe Pose landmarks are typically 0-10 for the face region.
FACIAL_LANDMARK_INDICES = frozenset([
    mp_pose.PoseLandmark.NOSE,
    mp_pose.PoseLandmark.LEFT_EYE_INNER, mp_pose.PoseLandmark.LEFT_EYE,
    mp_pose.PoseLandmark.LEFT_EYE_OUTER, mp_pose.PoseLandmark.RIGHT_EYE_INNER,
    mp_pose.PoseLandmark.RIGHT_EYE, mp_pose.PoseLandmark.RIGHT_EYE_OUTER,
    mp_pose.PoseLandmark.LEFT_EAR, mp_pose.PoseLandmark.RIGHT_EAR,
    mp_pose.PoseLandmark.MOUTH_LEFT, mp_pose.PoseLandmark.MOUTH_RIGHT
])

# Create a new set of connections that explicitly excludes facial connections.
# A connection is excluded if either of its endpoints is a facial landmark.
NON_FACIAL_POSE_CONNECTIONS = frozenset([
    connection for connection in mp_pose.POSE_CONNECTIONS
    if connection[0] not in FACIAL_LANDMARK_INDICES and
       connection[1] not in FACIAL_LANDMARK_INDICES
])

# Define drawing specifications for non-facial landmarks (points)
# This will be used to draw white circles for body landmarks.
NON_FACIAL_LANDMARK_DRAWING_SPEC = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=3)

# Define drawing specifications for facial landmarks (points)
# This will make facial landmarks completely invisible.
FACIAL_LANDMARK_DRAWING_SPEC = mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=0, circle_radius=0)

# Define drawing specifications for non-facial connections (lines)
# This will be used to draw sky blue lines for body connections.
NON_FACIAL_CONNECTION_DRAWING_SPEC = mp_drawing.DrawingSpec(color=(64, 224, 208), thickness=2)




def process_image_with_pose_estimation(image_path: str, output_path: str):
    """
    Performs pose estimation on a single image and saves the result,
    excluding facial landmarks and their connecting lines,
    while keeping body landmarks and connections intact.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error: Could not load image from {image_path}")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1, enable_segmentation=False) as pose:
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            # Clone all landmarks (keeping original indices for correct connections)
            landmarks = results.pose_landmarks.landmark
            modified_landmarks = landmark_pb2.NormalizedLandmarkList()
            for idx, landmark in enumerate(landmarks):
                lm = landmark_pb2.NormalizedLandmark()
                lm.x, lm.y, lm.z, lm.visibility = landmark.x, landmark.y, landmark.z, landmark.visibility
                if idx in [f.value for f in FACIAL_LANDMARK_INDICES]:
                    # Move facial landmarks far outside visible range so they won't be drawn
                    lm.x, lm.y, lm.z = -1.0, -1.0, -1.0
                modified_landmarks.landmark.append(lm)

            # Draw landmarks and connections (body only)
            mp.solutions.drawing_utils.draw_landmarks(
                image=image,
                landmark_list=modified_landmarks,
                connections=NON_FACIAL_POSE_CONNECTIONS,  # Only body connections
                landmark_drawing_spec=NON_FACIAL_LANDMARK_DRAWING_SPEC,
                connection_drawing_spec=NON_FACIAL_CONNECTION_DRAWING_SPEC
            )
        else:
            print(f"⚠️ No pose landmarks detected in {os.path.basename(image_path)}")

    cv2.imwrite(output_path, image)
    print(f"✅ Processed image saved to: {output_path}")



def process_video_with_pose_estimation(video_path: str, output_path: str):
    """
    Performs pose estimation on a video and saves the result,
    excluding facial landmarks and their connecting lines,
    while keeping body landmarks and connections visible.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video not found or can't be opened: {video_path}")

    # Get video properties
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    with mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                # Clone landmarks and hide facial ones by moving them out of frame
                landmarks = results.pose_landmarks.landmark
                modified_landmarks = landmark_pb2.NormalizedLandmarkList()
                for idx, landmark in enumerate(landmarks):
                    lm = landmark_pb2.NormalizedLandmark()
                    lm.x, lm.y, lm.z, lm.visibility = landmark.x, landmark.y, landmark.z, landmark.visibility
                    if idx in [f.value for f in FACIAL_LANDMARK_INDICES]:
                        lm.x, lm.y, lm.z = -1.0, -1.0, -1.0
                    modified_landmarks.landmark.append(lm)

                # Draw body landmarks and connections only
                mp.solutions.drawing_utils.draw_landmarks(
                    image=frame,
                    landmark_list=modified_landmarks,
                    connections=NON_FACIAL_POSE_CONNECTIONS,
                    landmark_drawing_spec=NON_FACIAL_LANDMARK_DRAWING_SPEC,
                    connection_drawing_spec=NON_FACIAL_CONNECTION_DRAWING_SPEC
                )

            out.write(frame)

    cap.release()
    out.release()
    print(f"✅ Processed video saved to: {output_path}")
