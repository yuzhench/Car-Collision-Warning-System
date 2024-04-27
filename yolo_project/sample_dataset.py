import os
import shutil
import csv
import sys
import random

# Set this flag to True if you want to clean label files
CLEAN_LABELS = False

# Set sampling method: "front", "reversed", "random"
SAMPLING_METHOD = "random"

# Set of classes we are interested in
interested_classes = {'1', '5', '6', '13', '14'}



def create_directory(path):
    if os.path.exists(path):
        print(f"Directory {path} already exists. Exiting.")
        sys.exit(1)
    else:
        os.makedirs(path)

def main():
    # Load the CSV file
    print("Loading Train.csv...")
    print("- CLEAN_LABLES flag set to", CLEAN_LABELS)
    print("- Sampling Method set to", SAMPLING_METHOD)
    train_df = []
    with open('train.csv', 'r') as csv_file:
        reader = csv.reader(csv_file)
        train_df = list(reader)
    
    # Directory paths
    original_images_dir = './images'
    original_labels_dir = './labels'

    # Create a dictionary to store class mappings
    class_to_files = {class_id: [] for class_id in interested_classes}

    # Read the .txt files and map class labels to image files
    for row in train_df:
        image, label = row
        label_path = os.path.join(original_labels_dir, label)
        with open(label_path, 'r') as file:
            data = file.readlines()
            for line in data:
                class_id = line.split()[0]
                if class_id in interested_classes:
                    if image not in class_to_files[class_id]:
                        class_to_files[class_id].append(image)

    # Ask user for the number of examples to extract
    n = int(input("Enter the number of examples to extract per class: "))

    # Determine the order of sampling based on the selected method
    for class_id in class_to_files:
        if SAMPLING_METHOD == "reversed":
            class_to_files[class_id].reverse()
        elif SAMPLING_METHOD == "random":
            random.shuffle(class_to_files[class_id])

    # Check available samples and determine the actual number of samples to copy
    min_samples = min([len(files) for files in class_to_files.values() if len(files) >= n], default=0)
    samples_to_copy = min(n, min_samples)

    # Create directories for the new subset
    subset_dir = f'./sampling_{samples_to_copy}_{SAMPLING_METHOD}'
    subset_images_dir = os.path.join(subset_dir, 'images')
    subset_labels_dir = os.path.join(subset_dir, 'labels')

    # Check if the directory already exists
    if os.path.exists(subset_dir):
        print(f"Directory {subset_dir} already exists. Exiting.")
        sys.exit(1)

    # Create new directories since they don't exist
    os.makedirs(subset_images_dir)
    os.makedirs(subset_labels_dir)

    # Initialize a list to keep track of copied files
    copied_files = []
    class_counts = {class_id: 0 for class_id in interested_classes}  # Track counts for each class

    # Copy files and possibly clean label files
    for class_id, files in class_to_files.items():
        count = 0
        for file in files:
            if count >= samples_to_copy:
                break
            image_path = os.path.join(original_images_dir, file)
            label_path = os.path.join(original_labels_dir, file.replace('.jpg', '.txt'))
            new_image_path = os.path.join(subset_images_dir, file)
            new_label_path = os.path.join(subset_labels_dir, file.replace('.jpg', '.txt'))
            
            if file not in copied_files:
                shutil.copy(image_path, new_image_path)
                if CLEAN_LABELS:
                    with open(label_path, 'r') as original, open(new_label_path, 'w') as modified:
                        for line in original:
                            if line.split()[0] in interested_classes:
                                modified.write(line)
                else:
                    shutil.copy(label_path, new_label_path)
                
                copied_files.append(file)
                class_counts[class_id] += 1
                count += 1

    # Generate new train.csv file
    with open(os.path.join(subset_dir, 'train.csv'), 'w') as f:
        csv_writer = csv.writer(f)
        for file in copied_files:
            csv_writer.writerow([file, file.replace('.jpg', '.txt')])

    # Print the class sample counts
    print("Sampled dataset created at", subset_dir)
    print("Sampled counts per class:")
    for class_id, count in class_counts.items():
        print(f"Class {class_id}: {count} samples")

if __name__ == "__main__":
    main()
