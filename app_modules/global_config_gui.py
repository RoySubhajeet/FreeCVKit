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

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog # Import filedialog for file browsing



import tkinter as tk
from tkinter import ttk, messagebox, filedialog


class ConfigApp:
    def __init__(self):
        # ---- GLOBAL CONFIGS ----
        self.config = {
            "cv_task": "Pose Estimation and Landmark tracking",
            "model_type": None,
            "tracking_algo": None,
            "detection_sensitivity": "Medium",
            "input_source": None,
            "file_path": None,
            "selected_labels": []
        }

        # ---- OPTIONS ----
        self.options = {
            "cv_task": ["Pose Estimation and Landmark tracking", "Object Detection", "Object Tracking"],
            "model_type": {
                "Pose Estimation and Landmark tracking": ["MediaPipe Pose landmarker"],
                "Object Detection": ["EfficientDet-Lite", "YOLO"],
                "Object Tracking": ["EfficientDet-Lite", "YOLO"]
            },
            "tracking_algo": {
                "Pose Estimation and Landmark tracking": ["None"],
                "Object Detection": ["None"],
                "Object Tracking": ["Centroid-Tracking","Deep SORT" ,"Other tracking"]
            },
            "detection_sensitivity": ["Low", "Medium", "High", "Very High"],
            "input_source": ["Select image from gallery to analyze", "Select video from gallery", "Live video"],
            "labels": ["person", "car", "ball", "dog"]
        }

        # ---- UI References ----
        self.root = None
        self.widgets = {}
        self.checkbox_vars = {}

    # ---------------- GUI CREATION ---------------- #
    def create_gui(self):
        self.root = tk.Tk()
        self.root.title("App Configuration")
        self.root.geometry("500x650")
        self.root.resizable(False, False)

        style = ttk.Style()
        style.theme_use('clam')

        row = 0
        # Dynamic dropdown creation
        for key in ["cv_task", "model_type", "tracking_algo", "detection_sensitivity", "input_source"]:
            self._create_dropdown(key, row)
            row += 1

        # Labels (checkboxes)
        self._create_label_checkboxes(row)
        row += 1

        # File path display
        self.widgets["file_path_label"] = ttk.Label(self.root, text="File Path: None", font=('Arial', 9), wraplength=450)
        self.widgets["file_path_label"].pack(pady=5)

        # Save button
        save_btn = ttk.Button(self.root, text="Save Configuration", command=self.save_configuration)
        save_btn.pack(pady=20)

        # Config display
        self.widgets["config_display"] = ttk.Label(self.root, text="Current Config: (Not saved yet)",
                                                   font=('Arial', 10), wraplength=450, justify=tk.LEFT)
        self.widgets["config_display"].pack(pady=10)

        self._update_dropdowns()
        self.root.mainloop()

    # ---------------- HELPER FUNCTIONS ---------------- #
    def _create_dropdown(self, key, row):
        """Create a dropdown dynamically."""
        ttk.Label(self.root, text=f"{row + 1}) Select {key.replace('_', ' ').title()}:",
                  font=('Arial', 10, 'bold')).pack(pady=10)

        combo = ttk.Combobox(self.root, state="readonly", width=40)
        combo.pack(pady=5)

        combo.bind("<<ComboboxSelected>>", lambda e, k=key: self._on_dropdown_change(k))
        self.widgets[key] = combo

    def _create_label_checkboxes(self, row):
        """Create label checkboxes dynamically."""
        frame = ttk.Frame(self.root)
        frame.pack(pady=5)
        ttk.Label(frame, text="Select Labels to Filter:", font=('Arial', 10, 'bold')).pack(pady=5)
        inner_frame = ttk.Frame(frame)
        inner_frame.pack()

        for label in self.options["labels"]:
            var = tk.BooleanVar(value=(label == "person"))
            self.checkbox_vars[label] = var
            cb = ttk.Checkbutton(inner_frame, text=label.capitalize(), variable=var)
            cb.pack(side=tk.LEFT, padx=5)

        self.widgets["labels_frame"] = frame

    def _update_dropdowns(self):
        """Update model type and tracking algorithm dynamically based on cv_task."""
        cv_task = self.config["cv_task"]

        # Set CV Task options and current value
        self.widgets["cv_task"]["values"] = self.options["cv_task"]
        self.widgets["cv_task"].set(cv_task)

        # Update dependent dropdowns
        self.widgets["model_type"]["values"] = self.options["model_type"].get(cv_task, [])
        self.widgets["model_type"].set(self.options["model_type"][cv_task][0])

        self.widgets["tracking_algo"]["values"] = self.options["tracking_algo"].get(cv_task, [])
        self.widgets["tracking_algo"].set(self.options["tracking_algo"][cv_task][0])

        self.widgets["detection_sensitivity"]["values"] = self.options["detection_sensitivity"]
        self.widgets["detection_sensitivity"].set(self.config["detection_sensitivity"])

        self.widgets["input_source"]["values"] = self.options["input_source"]
        self.widgets["input_source"].set("Select an option")

        # Hide labels unless in detection/tracking
        if cv_task in ["Object Detection", "Object Tracking"]:
            self.widgets["labels_frame"].pack(pady=5)
        else:
            self.widgets["labels_frame"].pack_forget()

    def _on_dropdown_change(self, key):
        """Handle dropdown changes dynamically."""
        selected_value = self.widgets[key].get()
        self.config[key] = selected_value

        if key == "cv_task":
            self._update_dropdowns()

        if key == "input_source":
            self._handle_input_source_selection(selected_value)

    def _handle_input_source_selection(self, selected_source):
        """Handle file browsing for image/video input sources."""
        self.config["file_path"] = None
        self.widgets["file_path_label"].config(text="File Path: None")

        if selected_source == "Select image from gallery to analyze":
            filetypes = [("Image Files", "*.png *.jpg *.jpeg *.gif *.bmp")]
        elif selected_source == "Select video from gallery":
            filetypes = [("Video Files", "*.mp4 *.avi *.mov *.mkv")]
        else:
            if selected_source == "Live video":
                messagebox.showinfo("Feature Not Supported", "Live video is not supported for now.")
            return

        file_path = filedialog.askopenfilename(title="Select File", filetypes=filetypes)
        if file_path:
            self.config["file_path"] = file_path
            self.widgets["file_path_label"].config(text=f"File Path: {file_path}")
        else:
            self.widgets["input_source"].set("Select an option")

    def save_configuration(self):
        """Validate and save configuration."""
        for key in ["cv_task", "model_type", "detection_sensitivity", "input_source"]:
            if not self.widgets[key].get() or self.widgets[key].get() == "Select an option":
                messagebox.showerror("Validation Error", f"Please select {key.replace('_', ' ')}.")
                return

        if self.config["input_source"] in ["Select image from gallery to analyze", "Select video from gallery"]:
            if not self.config["file_path"]:
                messagebox.showerror("Validation Error", "Please select a valid file.")
                return

        # Save label selections for detection/tracking
        if self.config["cv_task"] in ["Object Detection", "Object Tracking"]:
            self.config["selected_labels"] = [label for label, var in self.checkbox_vars.items() if var.get()]
            if not self.config["selected_labels"]:
                messagebox.showerror("Validation Error", "Please select at least one label.")
                return
        else:
            self.config["selected_labels"] = []

        # Update display
        config_text = "Current Config:\n" + "\n".join([f"{k}: {v}" for k, v in self.config.items()])
        self.widgets["config_display"].config(text=config_text)

        messagebox.showinfo("Configuration Saved", "Your selections have been saved successfully!")
        print("Updated Global Configuration:", self.config)
        self.root.destroy()



