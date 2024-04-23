import torch
import torchvision.transforms as transforms
from PIL import Image
from model_error import Yolov1
from utils import cellboxes_to_boxes, non_max_suppression, plot_image

# Load the model architecture
model = Yolov1(split_size=7, num_boxes=2, num_classes=20)

# Load the trained weights
checkpoint = torch.load("overfit.pth.tar")
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Define transformations for preprocessing the image
transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
])

# Load the image
image = Image.open("car.jpg")

# Preprocess the image
input_image = transform(image)
input_image = input_image.unsqueeze(0)  # Add batch dimension

# Perform inference
with torch.no_grad():
    predictions = model(input_image)

# Convert cellboxes to regular bounding boxes
bounding_boxes = cellboxes_to_boxes(predictions)
bounding_boxes = non_max_suppression(bounding_boxes[0], iou_threshold=0.5, threshold=0.4, box_format="midpoint")

# Visualize the results
plot_image(image, bounding_boxes)