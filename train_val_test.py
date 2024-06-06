import os
import shutil
import random

def divide_data(parent_folder, output_folder_name):
    """
    Divides the data in the parent folder into train, validation, and test sets,
    and copies the files to the appropriate folders.

    Args:
        parent_folder (str): The path to the parent folder containing the data.
        output_folder_name (str): The name of the output folder to create.
    """
    # Create the output folder in the same directory as the parent folder
    output_folder = os.path.join(os.path.dirname(parent_folder), output_folder_name)
    os.makedirs(output_folder, exist_ok=True)

    # Create the Train, Validation, and Test folders
    train_folder = os.path.join(output_folder, "Train")
    val_folder = os.path.join(output_folder, "Validation")
    test_folder = os.path.join(output_folder, "Test")
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Copy the files to the appropriate folders
    for subfolder in os.listdir(parent_folder):
        subfolder_path = os.path.join(parent_folder, subfolder)
        if os.path.isdir(subfolder_path):
            # Create the subfolder structure in the Train, Validation, and Test folders
            train_subfolder = os.path.join(train_folder, subfolder)
            val_subfolder = os.path.join(val_folder, subfolder)
            test_subfolder = os.path.join(test_folder, subfolder)
            os.makedirs(train_subfolder, exist_ok=True)
            os.makedirs(val_subfolder, exist_ok=True)
            os.makedirs(test_subfolder, exist_ok=True)

            # Get the list of files in the current subfolder
            files = os.listdir(subfolder_path)
            random.shuffle(files)

            # Calculate the number of files for the train, validation, and test sets
            total_files = len(files)
            train_files_count = int(0.7 * total_files)
            val_files_count = int(0.15 * total_files)
            test_files_count = total_files - train_files_count - val_files_count

            # Copy files to the Train folder
            train_files = files[:train_files_count]
            for file in train_files:
                src_file = os.path.join(subfolder_path, file)
                dst_file = os.path.join(train_subfolder, file)
                shutil.copy(src_file, dst_file)

            # Copy files to the Validation folder
            val_files = files[train_files_count:train_files_count+val_files_count]
            for file in val_files:
                src_file = os.path.join(subfolder_path, file)
                dst_file = os.path.join(val_subfolder, file)
                shutil.copy(src_file, dst_file)

            # Copy the remaining files to the Test folder
            test_files = files[train_files_count+val_files_count:]
            for file in test_files:
                src_file = os.path.join(subfolder_path, file)
                dst_file = os.path.join(test_subfolder, file)
                shutil.copy(src_file, dst_file)