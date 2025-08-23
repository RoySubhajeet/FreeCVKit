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

import os

def get_file_name_from_path(file_path: str) -> str:
    """
    Extracts and returns the file name (including extension) from a given file path.

    Args:
        file_path (str): The full or relative path to a file.

    Returns:
        str: The name of the file.
    """
    # os.path.basename() extracts the last component of a pathname,
    # which is typically the file name or the last directory name.
    file_name = os.path.basename(file_path)
    return file_name

# --- Example Usage ---
if __name__ == "__main__":
    # Example 1: Full path with a common file type
    path1 = "/home/user/documents/report.pdf"
    name1 = get_file_name_from_path(path1)
    print(f"Path: '{path1}' -> File Name: '{name1}'")

    # Example 2: Relative path
    path2 = "images/photo.jpg"
    name2 = get_file_name_from_path(path2)
    print(f"Path: '{path2}' -> File Name: '{name2}'")

    # Example 3: Path with no directory (just a file name)
    path3 = "my_script.py"
    name3 = get_file_name_from_path(path3)
    print(f"Path: '{path3}' -> File Name: '{name3}'")

    # Example 4: Path to a directory (basename will return the directory name)
    path4 = "/var/log/"
    name4 = get_file_name_from_path(path4)
    print(f"Path: '{path4}' -> File Name: '{name4}'") # Note: Returns directory name for directories

    # Example 5: Path with spaces and special characters
    path5 = "C:\\Users\\Public\\My Documents\\Project Alpha (2023).docx"
    name5 = get_file_name_from_path(path5)
    print(f"Path: '{path5}' -> File Name: '{name5}'")