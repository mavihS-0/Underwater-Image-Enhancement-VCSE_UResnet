import os

# Specify the directory containing the images
folder_path = 'UResnet\data\label\input'

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    # Check if the file is an image (add more formats if needed)
    if filename.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')):
        # Extract the file name and extension
        name, ext = os.path.splitext(filename)
        # Define the new name by adding '_label' before the extension
        new_name = f"processed_{name}{ext}"
        # Rename the file
        os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_name))

print("Renaming completed!")
