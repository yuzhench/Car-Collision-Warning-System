import csv

# Define the range of image and text file numbers
start_num = 1
end_num = 2000

# Define the CSV file path
csv_file = "data/bigger_dataset.csv"

# Open the CSV file in write mode
with open(csv_file, 'w', newline='') as csvfile:
    # Create a CSV writer object
    csv_writer = csv.writer(csvfile)
    
    # Write the header row
    csv_writer.writerow(['image', 'text'])
    
    # Loop through the range of numbers
    for num in range(start_num, end_num + 1):
        # Generate the image and text file paths
        image_path = f"{num:06d}.jpg"
        text_path = f"{num:06d}.txt"
        
        # Write the image and text file paths to the CSV file
        csv_writer.writerow([image_path, text_path])

print(f"CSV file '{csv_file}' has been created.")
