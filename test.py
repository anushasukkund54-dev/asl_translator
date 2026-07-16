import os

# Point this to the main folder containing the letter sub-folders (A, B, C...)
# Update this path exactly to where your folders are:
path = r"C:/Users/anush/Downloads/dataset/asl_alphabet_train/asl_alphabet_train"

print("--- Counting Images per Gesture ---")
for folder_name in os.listdir(path):
    folder_path = os.path.join(path, folder_name)
    if os.path.isdir(folder_path):
        num_files = len(os.listdir(folder_path))
        print(f"Gesture {folder_name}: {num_files} images")