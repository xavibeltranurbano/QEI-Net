import zipfile
import os

def unzip_file(zip_path, extract_to):
    # Check if the zip file exists
    if not os.path.isfile(zip_path):
        print(f"The file {zip_path} does not exist.")
        return

    # Create the directory if it doesn't exist
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    # Unzipping process
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        print(f"Extracted all files in {zip_path} to {extract_to}")

# Replace with your file paths
zip_file_path = '/home/xurbano/QEI-ASL/QEI-Dataset.zip'
extract_path = '/home/xurbano/QEI-ASL'

unzip_file(zip_file_path, extract_path)
