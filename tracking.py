import json
import numpy as np
import cv2
from PIL import Image, ImageDraw
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
from scipy.integrate import quad

#define the constant value here 
WIDTH = 200
HEIGHT = 200
BOX_LENGTH = 3
TIME_PERIOD = 0.03
SAFETY_DISATNT = 8


 


def draw_point(row,image):
    if image is None:
        background = Image.new('RGB',(WIDTH,HEIGHT), color="grey")
    else:
        background = image
    
    all_points = []
    for i in range(0, len(row), 2):
        one_point = row[i], row[i+1]
        all_points.append(one_point)

    for point in all_points:
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
    for i in range(0,start_point.shape[0]):
        draw.line(((start_point[i][1],start_point[i][0]), (end_point[i][1],end_point[i][0])), fill=color, width=2)

    # scaling_factor = 2
    # arrowhead_points = [(end_point[0] + scaling_factor * (start_point[0] - end_point[0]), 
    #                     end_point[1] + scaling_factor * (start_point[1] - end_point[1])),
    #                     (end_point[0] + scaling_factor * (start_point[0] - end_point[0]) + 0.05 * (start_point[1] - end_point[1]), 
    #                     end_point[1] + scaling_factor * (start_point[1] - end_point[1]) - 0.05 * (start_point[0] - end_point[0])),
    #                     (end_point[0], end_point[1])]
    # draw.polygon(arrowhead_points, fill=color)
    # image.show()
    return image


def find_appoximate_curve(degrees_threshold,All_points,image):
    fig, ax = plt.subplots()
    ax.imshow(image)

    functions = []

    for i in range(0, All_points.shape[1], 2):
    
        points = All_points[:,i:i+2]
    
        y = points[:,0]
        x = points[:,1]
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


        #collect the functions 
        poly_function = np.poly1d(best_coefficients)
        functions.append(poly_function)
        # print("the function expression is: ", poly_function)

        x_curve_pred = np.linspace(max(x), WIDTH-1, 100)
        y_curve_pred = np.minimum (np.polyval(best_coefficients, x_curve_pred), HEIGHT-1)

        # Plot the data points and the best curve fit
        # fig, ax = plt.subplots()
        # ax.imshow(image)
        ax.plot(x_curve, y_curve, color='red', label=f'Best Fit (Degree {best_degree})')
        ax.plot(x_curve_pred, y_curve_pred, color='orange', label=f'prediction (Degree {best_degree})')

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

    # pil_image.show()
    
    return pil_image, functions

def check_intersection(func1, func2):
    x_range = np.linspace(0, WIDTH)

    y1 = func1(x_range)
    y2 = func2(x_range)

    intersection = any(np.sign(y1) != np.sign(y2))
    return intersection

def collision_detection(function_list, All_latest_points, All_average_speed = 10): #assume All_latest_points is a 2 collom matrix 
    for i in range(len(function_list)):
        for j in range (i+1, len(function_list)) :
            function_1 = function_list[i]
            function_2 = function_list[j]

            if check_intersection(function_1, function_2):
                print("intersection")
                #starting to check the distance between this two points 
                point_1_x = All_latest_points[i][0]
                point_2_x = All_latest_points[j][0]
                 
                

                

def calculate_average_speed(function, curr_points): #assume here the curr_point is a 2D colume matrix
    start_x = curr_points[0][0]
    end_x = curr_points[curr_points.shape[0]][0]
    curve_length = quad(function,start_x,end_x)
    return curve_length
    

def main():
    # read the point data from the json file 
    f = open("position.json")
    data = json.load(f)
    positions = data["position"]
    positions_2 = data["position_2"]

    mat_pos_1 = np.array(positions)
    mat_pos_2 = np.array(positions_2)
    mat_pos = np.concatenate((mat_pos_1, mat_pos_2),axis=1)
    
    folder_dir = "/home/yuzhen/Desktop/EECS442/final_project/frames"
    old_point = None
    curr_point = None
    All_curr_points = []
    curr_points = []
    curr_img = None
    all_position = [positions, positions_2]


    
    for index, row in enumerate (mat_pos):
        curr_points.append(row)
        curr_point = row
        file_direction = os.path.join(folder_dir, f"{index}.jpg")
        print(file_direction)
        image = draw_point(row,curr_img)
        curr_img = image
        if old_point is not None:
            start_point = old_point.reshape(-1,2) ## each row is a point coordination
            curr_point = curr_point.reshape(-1,2)
            print ("start_point: ",start_point)
            print ("curr_point: ", curr_point)
            # exit(0)
            direction = curr_point - start_point # here is a row
            distance = [np.linalg.norm((pair[0],pair[1])) for pair in direction]
            distance = (np.array(distance)).reshape(-1,1)
            unit_direction = direction / distance
            final_image = draw_vector(curr_point, unit_direction,distance,image)
            degrees_threshold = 4
            final_image, functions = find_appoximate_curve(degrees_threshold,np.array(curr_points),final_image)
            ##detect the function cross:
            collision_detection(functions,curr_point)
            final_image.save(file_direction) 
        # else:
        #     image.save(file_direction)
        old_point = row

    #reset for the next for loop 
    old_point = None
    curr_point = None
    curr_points = []




    # for obj_index, temp_positions in enumerate(all_position):
    #     for index, position in enumerate (temp_positions):
    #         curr_points.append(position)
    #         curr_point = np.array([position[1], position[0]])
    #         file_direction = os.path.join(folder_dir, f"{obj_index}_{index}.jpg")
    #         print(file_direction)
    #         image = draw_point(position,curr_img)
    #         curr_img = image
    #         if old_point is not None:
    #             start_point = np.array([old_point[1], old_point[0] ])
    #             direction = curr_point - start_point
    #             distance = np.linalg.norm(direction)
    #             unit_direction = direction / distance
    #             final_image = draw_vector(curr_point, unit_direction,distance*2,image)
    #             degrees_threshold = 4
    #             final_image = find_appoximate_curve(degrees_threshold,np.array(curr_points),final_image)
    #             final_image.save(file_direction) 
                
    #         # else:
    #         #     image.save(file_direction)
    #         old_point = position

    #     #reset for the next for loop 
    #     old_point = None
    #     curr_point = None
    #     curr_points = []


if __name__ == "__main__":
    main()

 

