import os
import shutil
import random

def divide_data(parent_folder, output_folder_name):
    """
    Divides the data in the parent folder into train and test sets,
    and copies the files to the appropriate folders.

    Args:
        parent_folder (str): The path to the parent folder containing the data.
        output_folder_name (str): The name of the output folder to create.
    """
    # Create the output folder in the same directory as the parent folder
    output_folder = os.path.join(os.path.dirname(parent_folder), output_folder_name)
    os.makedirs(output_folder, exist_ok=True)

    # Create the Train and Test folders
    train_folder = os.path.join(output_folder, "Train")
    test_folder = os.path.join(output_folder, "Test")
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Copy the files to the appropriate folders
    for subfolder in os.listdir(parent_folder):
        subfolder_path = os.path.join(parent_folder, subfolder)
        if os.path.isdir(subfolder_path):
            # Create the subfolder structure in the Train and Test folders
            train_subfolder = os.path.join(train_folder, subfolder)
            test_subfolder = os.path.join(test_folder, subfolder)
            os.makedirs(train_subfolder, exist_ok=True)
            os.makedirs(test_subfolder, exist_ok=True)

            # Get the list of files in the current subfolder
            files = os.listdir(subfolder_path)
            random.shuffle(files)

            # Calculate the number of files for the train and test sets
            train_files_count = int(0.8 * len(files))

            # Copy files to the Train folder
            for file in files[:train_files_count]:
                src_file = os.path.join(subfolder_path, file)
                dst_file = os.path.join(train_subfolder, file)
                shutil.copy(src_file, dst_file)

            # Copy the remaining files to the Test folder
            for file in files[train_files_count:]:
                src_file = os.path.join(subfolder_path, file)
                dst_file = os.path.join(test_subfolder, file)
                shutil.copy(src_file, dst_file)

