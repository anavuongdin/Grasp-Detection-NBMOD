import os
import time
import subprocess

# Replace 'path_to_folder' with the path to your target folder
folder_path = '/LOCAL2/anguyen/faic/lthieu/data'

# Replace 'run.sh' with the name of your bash script
bash_script = '/LOCAL2/anguyen/faic/vdan/grasping/Grasp-Detection-NBMOD/run.sh'

# Set a list to keep track of the files in the folder
files_in_folder = os.listdir(folder_path)

# Replace 'batch_number' with the desired batch number for the 'run.sh' script
batch_number = 8

while True:
    # Get the list of files in the folder
    files = os.listdir(folder_path)

    # Check for new files
    new_files = [file for file in files if file not in files_in_folder]

    # Update the list of files_in_folder
    files_in_folder = files

    # If new files are found, execute the bash script for each new file
    for new_file in new_files:
        # Execute the bash script with the specified batch number
        subprocess.run(['bash', bash_script, str(batch_number)])

    batch_number += 1
    # Wait for a specified time before checking again (e.g., 5 seconds)
    time.sleep(5)
