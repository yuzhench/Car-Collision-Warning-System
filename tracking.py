import json
import numpy as np
import cv2
from PIL import Image, ImageDraw
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io

#define the constant value here 
WIDTH = 200
HEIGHT = 200
BOX_LENGTH = 3
TIME_PERIOD = 0.03

# read the point data from the json file 
f = open("position.json")
data = json.load(f)
positions = data["position"]


def draw_point(point,image):
    if image is None:
        background = Image.new('RGB',(WIDTH,HEIGHT), color="grey")
    else:
        background = image
     
    x,y=point
    draw = ImageDraw.Draw(background)

    center = point
    radius = 2

    # Draw the circle
    draw.ellipse((center[1]-radius, center[0]-radius,
                center[1]+radius, center[0]+radius),
                outline='white')

    box_x_y = [(y-BOX_LENGTH,x-BOX_LENGTH), (y+BOX_LENGTH,x+BOX_LENGTH)]

    draw.rectangle(box_x_y,outline="white",width=2)

    # background.show()
    # background.save(direction)
    return background

def velocity_calculate(point1, point2):
    point1 = np.array(point1)
    point2 = np.array(point2)

    vector = point2 - point1
    distance = np.linalg.norm(vector)
    return distance

def draw_vector(start_point, direction, length, image, color='blue'):
    end_point = start_point + length * direction
    draw = ImageDraw.Draw(image)
    draw.line([tuple(start_point), tuple(end_point)], fill=color, width=2)

    # scaling_factor = 2
    # arrowhead_points = [(end_point[0] + scaling_factor * (start_point[0] - end_point[0]), 
    #                     end_point[1] + scaling_factor * (start_point[1] - end_point[1])),
    #                     (end_point[0] + scaling_factor * (start_point[0] - end_point[0]) + 0.05 * (start_point[1] - end_point[1]), 
    #                     end_point[1] + scaling_factor * (start_point[1] - end_point[1]) - 0.05 * (start_point[0] - end_point[0])),
    #                     (end_point[0], end_point[1])]
    # draw.polygon(arrowhead_points, fill=color)
    # image.show()
    return image


def find_appoximate_curve(degrees_threshold,points,image):
    y = points[:, 0]
    x = points[:, 1]
    degrees = range(1,degrees_threshold)
    mse_values = []
    for degree in degrees:
        # Fit the polynomial
        coefficients = np.polyfit(x, y, degree)
        
        # Evaluate the polynomial
        y_fit = np.polyval(coefficients, x)
        
        # Calculate mean squared error
        mse = mean_squared_error(y, y_fit)
        mse_values.append(mse)

    best_degree = degrees[np.argmin(mse_values)]

    best_coefficients = np.polyfit(x, y, best_degree)
    x_curve = np.linspace(min(x), max(x), 100)
    y_curve = np.polyval(best_coefficients, x_curve)

    # Plot the data points and the best curve fit
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.plot(x_curve, y_curve, color='red', label=f'Best Fit (Degree {best_degree})')
    ax.text(0.8,0.8, f'velocity', fontsize=12, color='blue',ha='left', transform=ax.transAxes)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Best Curve Fitting')
    ax.legend()
    ax.grid(True)
    
    # Convert the plot to an image object
    canvas = FigureCanvas(fig)
    buf = io.BytesIO()
    canvas.print_png(buf)
    buf.seek(0)
    pil_image = Image.open(buf)

    # Convert image to RGB mode if it's RGBA
    if pil_image.mode == 'RGBA':
        pil_image = pil_image.convert('RGB')
    
    return pil_image


def main():
    folder_dir = "/home/yuzhen/Desktop/EECS442/final_project/frames"
    old_point = None
    curr_point = None
    curr_points = []
    curr_img = None
    for index, position in enumerate (positions):
        curr_points.append(position)
        curr_point = np.array([position[1], position[0]])
        file_direction = os.path.join(folder_dir, f"{index}.jpg")
        print(file_direction)
        image = draw_point(position,curr_img)
        curr_img = image
        if old_point is not None:
            start_point = np.array([old_point[1], old_point[0] ])
            direction = curr_point - start_point
            distance = np.linalg.norm(direction)
            unit_direction = direction / distance
            final_image = draw_vector(curr_point, unit_direction,distance*2,image)
            degrees_threshold = 4
            final_image = find_appoximate_curve(degrees_threshold,np.array(curr_points),final_image)
            final_image.save(file_direction)
             
        else:
            image.save(file_direction)
        old_point = position


if __name__ == "__main__":
    main()

 

