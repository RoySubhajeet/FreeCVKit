import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog # Import filedialog for file browsing

# Global configuration dictionary
# This dictionary will store the selected options
global_config = {
    "cv_task": "None",
    "model_type": "None",
    "detection_sensitivity": "None",
    "input_source": "None",
    "file_path": None, # To store selected image/video path
    "selected_labels": [] # Modified: New key to store selected object detection labels
}

# Global references for GUI elements
cv_task_combobox = None
model_type_combobox = None
sensitivity_combobox = None
input_source_combobox = None
file_path_label = None
root_window = None
config_display_label = None

# Modified: Global variables for checkboxes and their container
checkbox_frame = None
checkbox_vars = {} # Dictionary to hold tk.BooleanVar for each checkbox

# Modified: Function to convert checkbox states to a list of selected labels
def get_selected_labels_list():
    """
    Converts the state of the object detection checkboxes into a list of selected labels.
    """
    selected_labels = []
    for label, var in checkbox_vars.items():
        if var.get():
            selected_labels.append(label)
    return selected_labels

# Modified: update_model_options now also manages checkbox visibility
def update_model_options(*args):
    """
    Updates the options in the 'Model Type' dropdown based on the
    selection in the 'CV Task' dropdown and manages checkbox visibility.
    """
    selected_task = cv_task_combobox.get()

    if selected_task == "Pose Estimation and Landmark tracking":
        model_type_combobox['values'] = ["MediaPipe Pose landmarker"]
        model_type_combobox.set("MediaPipe Pose landmarker")
        # Modified: Hide checkboxes if not object detection
        if checkbox_frame:
            checkbox_frame.pack_forget()
    elif selected_task == "Object Detection Object tracking":
        model_type_combobox['values'] = ["EfficientDet-Lite", "YOLO"]
        model_type_combobox.set("EfficientDet-Lite") # Set a default for this task
        # Modified: Show checkboxes if object detection
        if checkbox_frame:
            checkbox_frame.pack(pady=5)
            # Modified: Ensure 'person' is selected by default when shown
            checkbox_vars["person"].set(True)
    else:
        model_type_combobox['values'] = []
        model_type_combobox.set("") # Clear selection if no task is chosen
        # Modified: Hide checkboxes if no task is chosen
        if checkbox_frame:
            checkbox_frame.pack_forget()

def handle_input_source_selection(*args):
    """
    Handles the 'Input Source' selection, either opening a file browser
    or displaying a 'Not supported' message.
    """
    selected_source = input_source_combobox.get()
    global_config["file_path"] = None # Reset file path on new selection
    file_path_label.config(text="File Path: None") # Clear display

    if selected_source == "Select image from gallery to analyze":
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.gif *.bmp")]
        )
        if file_path:
            global_config["file_path"] = file_path
            file_path_label.config(text=f"File Path: {file_path}")
        else:
            input_source_combobox.set("Select an option") # Reset if user cancels
    elif selected_source == "Select video from gallery":
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
        )
        if file_path:
            global_config["file_path"] = file_path
            file_path_label.config(text=f"File Path: {file_path}")
        else:
            input_source_combobox.set("Select an option") # Reset if user cancels
    elif selected_source == "Live video":
        messagebox.showinfo("Feature Not Supported", "Live video is not supported for now.")
        input_source_combobox.set("Select an option") # Reset to placeholder

def save_configuration():
    """
    Retrieves selected values from all dropdowns and checkboxes, validates them,
    saves them to the global_config dictionary, and closes the GUI.
    """
    # Perform validation for all required fields
    if not cv_task_combobox.get():
        messagebox.showerror("Validation Error", "Please select a CV Task.")
        return
    if not model_type_combobox.get():
        messagebox.showerror("Validation Error", "Please select a Model Type.")
        return
    if not sensitivity_combobox.get():
        messagebox.showerror("Validation Error", "Please select a Detection Sensitivity.")
        return

    selected_input_source = input_source_combobox.get()
    if not selected_input_source or selected_input_source == "Select an option":
        messagebox.showerror("Validation Error", "Please select an Input Source.")
        return

    # Specific validation for file path if gallery option is chosen
    if selected_input_source in ["Select image from gallery to analyze", "Select video from gallery"]:
        if not global_config["file_path"]:
            messagebox.showerror("Validation Error", "Please select a file from the gallery.")
            return

    # Modified: Validation for object detection labels if the task is selected
    if cv_task_combobox.get() == "Object Detection Object tracking":
        selected_labels = get_selected_labels_list()
        if not selected_labels:
            messagebox.showerror("Validation Error", "Please select at least one label for Object Detection.")
            return
        global_config["selected_labels"] = selected_labels
    else:
        global_config["selected_labels"] = [] # Clear if not object detection

    # If all validations pass, save the configuration
    global_config["cv_task"] = cv_task_combobox.get()
    global_config["model_type"] = model_type_combobox.get()
    global_config["detection_sensitivity"] = sensitivity_combobox.get()
    global_config["input_source"] = selected_input_source

    # Modified: Update the display label to show the current configuration including selected labels
    config_display_label.config(text=f"Current Config:\n"
                                     f"CV Task: {global_config['cv_task']}\n"
                                     f"Model Type: {global_config['model_type']}\n"
                                     f"Sensitivity: {global_config['detection_sensitivity']}\n"
                                     f"Input Source: {global_config['input_source']}\n"
                                     f"File Path: {global_config['file_path']}\n"
                                     f"Selected Labels: {global_config['selected_labels']}") # Modified

    messagebox.showinfo("Configuration Saved", "Your selections have been saved successfully!")
    print("Updated Global Configuration:", global_config) # For console verification

    # Close the dialog
    if root_window:
        root_window.destroy()

def create_gui():
    """
    Creates the main Tkinter GUI window with dropdowns, checkboxes, and a save button.
    """
    global root_window, cv_task_combobox, model_type_combobox, sensitivity_combobox, \
           input_source_combobox, file_path_label, config_display_label, checkbox_frame # Modified

    root_window = tk.Tk()
    root_window.title("App Configuration")
    root_window.geometry("500x650") # Modified: Increased window size to accommodate new elements
    root_window.resizable(False, False) # Make window not resizable

    style = ttk.Style()
    style.theme_use('clam')

    # --- CV Task Dropdown ---
    cv_task_label = ttk.Label(root_window, text="1) Select CV Task:", font=('Arial', 10, 'bold'))
    cv_task_label.pack(pady=10)

    cv_task_options = ["Pose Estimation and Landmark tracking", "Object Detection Object tracking"]
    cv_task_combobox = ttk.Combobox(root_window, values=cv_task_options, state="readonly", width=40)
    cv_task_combobox.set(cv_task_options[0])
    cv_task_combobox.pack(pady=5)
    cv_task_combobox.bind("<<ComboboxSelected>>", update_model_options)

    # --- Model Type Dropdown (Dynamically updated) ---
    model_label = ttk.Label(root_window, text="2) Select Model Type:", font=('Arial', 10, 'bold'))
    model_label.pack(pady=10)

    model_type_combobox = ttk.Combobox(root_window, values=[], state="readonly", width=40)
    model_type_combobox.pack(pady=5)
    update_model_options() # Initial call to populate based on default CV task

    # Modified: Object Detection Labels Checkboxes (New Section)
    checkbox_frame = ttk.Frame(root_window)
    # This frame will be packed/unpacked to show/hide checkboxes

    labels_label = ttk.Label(checkbox_frame, text="Select Labels to Filter:", font=('Arial', 10, 'bold'))
    labels_label.pack(pady=5)

    # Modified: Frame to hold checkboxes horizontally
    checkbox_inner_frame = ttk.Frame(checkbox_frame) # Modified
    checkbox_inner_frame.pack() # Modified

    object_labels = ["person", "car", "traffic signal", "dog"]
    for label in object_labels:
        var = tk.BooleanVar(value=False)
        checkbox_vars[label] = var
        # Modified: Pack checkboxes side by side
        cb = ttk.Checkbutton(checkbox_inner_frame, text=label.capitalize(), variable=var) # Modified
        cb.pack(side=tk.LEFT, padx=5) # Modified: Use side=tk.LEFT for horizontal alignment

    # Modified: Pre-select 'person' by default when the frame is initially created
    checkbox_vars["person"].set(True)

    # Modified: Initially hide checkboxes if Pose Estimation is the default task
    if cv_task_combobox.get() != "Object Detection Object tracking":
        checkbox_frame.pack_forget()


    # --- Detection Sensitivity Dropdown ---
    sensitivity_label = ttk.Label(root_window, text="Select Detection Sensitivity:", font=('Arial', 10))
    sensitivity_label.pack(pady=10)

    sensitivity_options = ["Low", "Medium", "High", "Very High"]
    sensitivity_combobox = ttk.Combobox(root_window, values=sensitivity_options, state="readonly", width=40)
    sensitivity_combobox.set(sensitivity_options[1])
    sensitivity_combobox.pack(pady=5)

    # --- Input Source Dropdown ---
    input_source_label = ttk.Label(root_window, text="3) Select Input Source:", font=('Arial', 10, 'bold'))
    input_source_label.pack(pady=10)

    input_source_options = ["Select image from gallery to analyze", "Select video from gallery", "Live video"]
    input_source_combobox = ttk.Combobox(root_window, values=input_source_options, state="readonly", width=40)
    input_source_combobox.set("Select an option")
    input_source_combobox.pack(pady=5)
    input_source_combobox.bind("<<ComboboxSelected>>", handle_input_source_selection)

    # --- File Path Display Label ---
    file_path_label = ttk.Label(root_window, text="File Path: None", font=('Arial', 9), wraplength=450)
    file_path_label.pack(pady=5)

    # Modified: Moved Save button after checkboxes (and other elements)
    # --- Save Button ---
    save_button = ttk.Button(root_window, text="Save Configuration", command=save_configuration)
    save_button.pack(pady=20) # Modified: Moved here

    # --- Configuration Display Label ---
    config_display_label = ttk.Label(root_window, text="Current Config: (Not saved yet)", font=('Arial', 10), wraplength=450, justify=tk.LEFT)
    config_display_label.pack(pady=10)

    root_window.mainloop()

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
        elif global_config["input_source"] == "Select video from gallery" and global_config["file_path"]:
            print(f"Analyzing video: {global_config['file_path']}")
        elif global_config["input_source"] == "Live video":
            print("Live video selected (currently not supported for processing).")
        else:
            print("No valid input source or file selected for Pose Estimation.")
    elif global_config["cv_task"] == "Object Detection Object tracking":
        print(f"Task: Object Detection. Model: {global_config['model_type']}. Sensitivity: {global_config['detection_sensitivity']}")
        print(f"Labels to filter: {global_config['selected_labels']}") # Modified
        if global_config["input_source"] == "Select image from gallery to analyze" and global_config["file_path"]:
            print(f"Analyzing image: {global_config['file_path']}")
        elif global_config["input_source"] == "Select video from gallery" and global_config["file_path"]:
            print(f"Analyzing video: {global_config['file_path']}")
        elif global_config["input_source"] == "Live video":
            print("Live video selected (currently not supported for processing).")
        else:
            print("No valid input source or file selected for Object Detection.")
    else:
        print("No specific CV task defined.")

# Call an example function to demonstrate using the saved config
perform_action_based_on_config()
