import torch
import torchvision.transforms as transforms
from PIL import Image
from model import Yolov1
from utils import cellboxes_to_boxes, non_max_suppression, plot_image, generate_box_center_list

# Load the model architecture
# model = Yolov1(split_size=7, num_boxes=2, num_classes=20)
model = Yolov1()

# Load the trained weights
checkpoint = torch.load("overfit_2000.pth.tar")
model.load_state_dict(checkpoint['state_dict'])
model.eval()



# Define transformations for preprocessing the image
transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
])

# Load the image
image = Image.open("traffic_cross_frames_cutted/frame_200.jpg")

image = image.convert("RGB")
# Preprocess the image
input_image = transform(image)
input_image = input_image.unsqueeze(0)  # Add batch dimension

# Perform inference
with torch.no_grad():
    predictions = model(input_image)
    # print(predictions.shape)

# Convert cellboxes to regular bounding boxes
bounding_boxes = cellboxes_to_boxes(predictions)
bounding_boxes = non_max_suppression(bounding_boxes[0], iou_threshold=0.5, threshold=0.3, box_format="midpoint")

# generate_box_center_list(bounding_boxes)
print(len(bounding_boxes))
print(bounding_boxes)


# Visualize the results
plot_image(image, bounding_boxes)
