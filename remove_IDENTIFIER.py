import os

def remove_identifier_files(root_dir):
    # Walk through all directories and subdirectories
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            # Check if the file is named '.IDENTIFIER'
            if filename == '.IDENTIFIER':
                file_path = os.path.join(dirpath, filename)
                try:
                    os.remove(file_path)
                    print(f"Removed: {file_path}")
                except Exception as e:
                    print(f"Failed to remove {file_path}: {e}")
    return
# Set the root directory to start the search
root_directory = '.'  # Change this to your target directory

# Call the function
remove_identifier_files(root_directory)
