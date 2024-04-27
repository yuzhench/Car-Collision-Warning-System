import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter

def iou_xywh(box_pred, box_gt):
    """
    DH
    """
    return iou_corners(xywh2corner(box_pred), xywh2corner(box_gt))

def xywh2corner(box):
    """
    DH
    box: Nx4: [x1,y1,x2,y2] top left & bottom right x,y, coordinates
    """
    corners = torch.zeros_like(box, dtype=torch.float32)
    corners[:,0] = box[:,0] - box[:,2] / 2  # x1 = x - w/2
    corners[:,1] = box[:,1] - box[:,3] / 2  # y1 = y - h/2
    corners[:,2] = box[:,0] + box[:,2] / 2  # x2 = x + w/2
    corners[:,3] = box[:,1] + box[:,3] / 2  # y2 = y + h/2
    return corners

def area_reg_box(box):
    return abs((box[:,3]-box[:,1]) * (box[:,2]-box[:,0]))

def iou_corners(box_pred, box_gt):
    """
    DH
    Algo: max of x1 y1 each, min of x2, y2 each, compute area
    """
    intersect_box = torch.zeros_like(box_pred, dtype=torch.float32)
    intersect_box[:,0] = torch.maximum(box_pred[:,0], box_gt[:,0]) # max of x1
    intersect_box[:,1] = torch.maximum(box_pred[:,1], box_gt[:,1]) # max of y1
    intersect_box[:,2] = torch.minimum(box_pred[:,2], box_gt[:,2]) # min of x2
    intersect_box[:,3] = torch.minimum(box_pred[:,3], box_gt[:,3]) # min of y2
    
    inter_area = area_reg_box(intersect_box)
    pred_area = area_reg_box(box_pred)
    gt_area = area_reg_box(box_gt)
    return inter_area / (pred_area + gt_area - inter_area + 1e-6)



# def get_line_ab(endpoints):
#     """
#     Endpoints: (X)x(x1,y1,x2,y2)
#     Assuming ax + y + b = 0 for a line
#     so a = (y1-y2)/(x1-x2) with offset to prevent /= 0
#         b = 0-a*x1-y1

#     Output (X)x1
#     """
#     # print("AB Input", endpoints.shape)
#     a1 = (endpoints[...,1] - endpoints[...,3]) / (endpoints[...,0] - endpoints[...,2] + 1e-9)
#     b1 = -endpoints[...,1] - a1 * endpoints[...,0]
#     # print("AB Output", a1.shape)
#     return(a1,b1)

def get_line_ab(endpoints):
    """
    Calculate line coefficients a, b for the equation y = ax + b
    Also returns coefficients c, d for the general form ax + by = c
    """
    dx = endpoints[..., 2] - endpoints[..., 0] + 1e-9  # Avoid division by zero
    dy = endpoints[..., 3] - endpoints[..., 1]
    a = dy / dx  # Slope
    b = endpoints[..., 1] - a * endpoints[..., 0]  # Intercept

    # Coefficients for general line form ax + by = c
    a_general = -dy
    b_general = dx
    c_general = a_general * endpoints[..., 0] + b_general * endpoints[..., 1]

    return a, b, a_general, b_general, c_general

def in_segment(endpoints, x,y):
    """
    Test if point in within line segment
    Input shape: (X)x4, (X), (X)
    Output: (X)
    """
    ymax = torch.maximum(endpoints[...,1], endpoints[...,3])
    ymin = torch.minimum(endpoints[...,1], endpoints[...,3])
    xmax = torch.maximum(endpoints[...,0], endpoints[...,2])
    xmin = torch.minimum(endpoints[...,0], endpoints[...,2])
    return torch.logical_and(
        torch.logical_and(x <= xmax + 1e-6, x >= xmin - 1e-6),
        torch.logical_and(y <= ymax + 1e-6, y >= ymin - 1e-6)
    )

def on_line_segment(endpoints, x,y):
    dxe = endpoints[...,2] - endpoints[...,0]
    dx = endpoints[...,2] - x

    dye = endpoints[...,3] - endpoints[...,1]
    dy = endpoints[...,3] - y

    return torch.isclose(dy/(dye + 1e-6), dx/(dxe+1e-6)) & in_segment(endpoints,x,y)


def line_intersect(line1, line2):
    a1, b1, a1_general, b1_general, c1 = get_line_ab(line1)
    a2, b2, a2_general, b2_general, c2 = get_line_ab(line2)

    # Check for parallel lines (a1/b1 == a2/b2 and c1/b1 != c2/b2)
    parallel = torch.isclose(a1 * b2, a2 * b1)
    same_line = torch.isclose(a1_general * c2, a2_general * c1) & torch.isclose(b1_general * c2, b2_general * c1)

    # Calculate intersection point using the determinant method (Cramer's rule)
    determinant = a1_general * b2_general - b1_general * a2_general + 1e-9
    x = (c1 * b2_general - b1_general * c2) / determinant
    y = (a1_general * c2 - c1 * a2_general) / determinant

    print("XY intersect:", x[-1,...], y[-1,...], "of lines", line1[-1,...], line2[-1,...])

    # Validate if intersection points are within both segments
    in_bounds = in_segment(line1, x, y) & in_segment(line2, x, y)

    valid = (~parallel | ~same_line) & in_bounds
    return x, y, valid

def xywht2corner(rect):
    """
    Input: Nx5 (x,y,w,h,theta) tuple
    Output Nx(8) 4-corner tuples, tl-tr-br-bl order
    """
    # Define the center coordinates
    center = rect[:,:2].reshape(-1,1,2)
    
    # Half dimensions to simplify corner calculations
    half_width = rect[:,2] / 2
    half_height = rect[:,3] / 2
    
    # Corners of the rectangle in local coordinate space (unrotated)
    wid = torch.stack((-half_width, half_width, half_width, -half_width), dim=1)
    hei = torch.stack((-half_height, -half_height, half_height, half_height), dim=1)
    corners = torch.stack((wid,hei),dim=2)
    corners.reshape((rect.shape[0], -1, 2))
    
    # Rotation matrix
    cost = torch.cos(rect[:,4])
    sint = torch.sin(rect[:,4])
    rot_matrix = torch.stack(
        (torch.stack((cost, -sint), dim=1),
         torch.stack((sint, cost),dim=1)),
        dim=2
    )
    
    # Rotate corners
    rotated_corners = torch.bmm(corners, rot_matrix)
    # Translate corners to global coordinate space
    oriented_corners = rotated_corners + center
    return oriented_corners

def gen_line(corners):
    """
    Corners: Nx4x2
    Pass in corners, generate lines for this polygon, in Nx((x1,y1),(x2,y2)) format
    Output: Nx4x4 format: (Nx4x(x1,y1,x2,y2))
    """
    corners_shift = torch.zeros_like(corners)
    corners_shift[:,:-1,:] = corners[:,1:,:]
    corners_shift[:,-1,:] = corners[:,0,:]
    return torch.cat((corners, corners_shift),dim=2)
    
def polygon_area(poly):
    """
    poly: shape Nx(kx2), with (x1,y1, ...) in that order, clockwise.
    """
    # Deal with empty (no intersect) case
    if poly.numel() == 0:
        return 0
        
    poly_end = torch.zeros_like(poly) # Roll
    poly_end[...,:-1,:] = poly[..., 1:,:]  # Nxkx2
    poly_end[..., -1,:] = poly[..., 0,:]

    # Init area
    area = torch.zeros(size=(poly.shape[0],1))  # k

    # Get differences
    dx = (poly_end[...,0] + poly[...,0]) / 2  # Nxk
    dy = poly_end[...,1] - poly[...,1]

    area = torch.abs(torch.sum(dx * dy, dim=1))
    return area


# def points_in_polygon(points, polygon):
#     """
#     Check if multiple points are inside a polygon using a vectorized approach.
#     points: torch.tensor of shape (n, 2), where n is the number of points to test
#     polygon: torch.tensor of shape (k, 2), where k is the number of corners in the polygon
#     Returns a tensor of shape (n,) with boolean values where True indicates the point is inside the polygon.
#     """
#     n = points.shape[0]
#     k = polygon.shape[0]
    
#     # Extend points and polygon for vectorized operation
#     extended_points = points.unsqueeze(1).expand(-1, k, -1)  # Nxkx2
#     point_ylines = extended_points.repeat((1,1,2))
#     point_ylines[...,0] = torch.minimum(point_ylines[...,0]-1, torch.zeros(size=(point_ylines.shape[0],)))  # Set first x to be -1, never reached by boxes
    
#     extended_polygon = polygon.unsqueeze(0).expand(n, -1, -1) # Nxkx2
#     rolled_polygon = torch.roll(extended_polygon, shifts=-1, dims=1)
#     polygon_lines = torch.cat((extended_polygon, rolled_polygon), dim=2) # Nxkx4

#     # Check conditions for intersection
#     x,y,validity = line_intersect(point_ylines, polygon_lines)

#     print("Validity:", validity)
    
#     # Count intersections
#     intersection_counts = validity.sum(dim=1)

#     # Deal with on-segment lines specially
#     for i in range(points.shape[0]):
#         for j in range(polygon_lines.shape[1]):
#             if on_line_segment(polygon_lines[i,j,:], points[i,0], points[i,1]):
#                 print("Found point on line:", points[i,0], points[i,1], "in segment", polygon_lines[i,j,:])
#                 intersection_counts[i] = 1
#                 break

#     # Point is inside if the count of intersections is odd
#     return intersection_counts % 2 == 1

def points_in_polygon(points, polygon):
    n = points.shape[0]
    k = polygon.shape[0]

    # Get polygon segments
    extended_polygon = polygon.unsqueeze(0).expand(n, -1, -1) # Nxkx2
    rolled_polygon = torch.roll(extended_polygon, shifts=-1, dims=1)
    polygon_lines = rolled_polygon - extended_polygon # Nxkx2
    
    # Extend points and polygon for vectorized operation
    extended_points = points.unsqueeze(1).expand(-1, k, -1).to(torch.float64)  # Nxkx2
    pt2polygon_lines = (extended_points - extended_polygon).to(torch.float64)
    # Compute the cross product and get the sign
    print("POLYGON LINES:", polygon_lines.shape)
    print("PT LINES:", pt2polygon_lines.shape)
    cross_prod = torch.linalg.det(torch.stack((polygon_lines, pt2polygon_lines), dim=2)) # Nxk 

    print("CROSS PROD:", cross_prod.shape)

    return (cross_prod>=0).all(dim=1) | (cross_prod<=0).all(dim=1)
    
    




def sort_points(points):
    """
    Sort the points to order them clockwise around their centroid.
    points: torch.tensor of shape (N, 2)
    Returns a tensor of shape (N, 2) with sorted points.
    """
    centroid = torch.mean(points, dim=0)
    angles = torch.atan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    return points[torch.argsort(angles, descending=True)]

def oriented_box_intersect(rect1_corners, rect2_corners):
    """
    Find intersects of cornerss

    rect1_corners: Nx4x2
    """
    # Generate Lines
    lines1 = gen_line(rect1_corners)
    lines2 = gen_line(rect2_corners) # Nx(4x4)

    # Tile lines1, lines2
    lines1_tiled = torch.repeat_interleave(lines1, 4, dim=1)  # Nx(16x4)
    lines2_tiled = lines2.repeat(1,4,1)
    # Compute intersections
    x,y, valid_inter = line_intersect(lines1_tiled, lines2_tiled) # (Nx16) tensors

    # Init everything list
    all_poly = []

    # Each intersection may nave different number of endpoints, so have to calculate differently. 
    for i in range(lines1.shape[0]):

        # Get intersects
        intersection_points = torch.stack((x[i, valid_inter[i,...]], y[i,valid_inter[i,...]]),dim=1)  # kx2, where k is number of valid intersections
        print("INTSCT", intersection_points)

        # Add corners of rect1 that are inside rect2
        c1in2 = points_in_polygon(rect1_corners[i,...], rect2_corners[i,...])
        if intersection_points.numel() == 0:
            intersection_points = rect1_corners[i,c1in2,:]
        elif rect1_corners[i,c1in2,:].numel() > 0:
            intersection_points = torch.cat((intersection_points, rect1_corners[i,c1in2,:]), dim=0)
            print("Adding 1 in 2:", rect1_corners[i,c1in2,:])

        c2in1 = points_in_polygon(rect2_corners[i,...], rect1_corners[i,...])
        if intersection_points.numel() == 0:
            intersection_points = rect2_corners[i,c2in1,:]
        elif rect2_corners[i,c2in1,:].numel() > 0:
            intersection_points = torch.cat((intersection_points, rect2_corners[i,c2in1,:]), dim=0)
            print("Adding 2 in 1:", rect2_corners[i,c2in1,:])

        print("Found Polygon:", intersection_points)
    
        # Sort the points to form a proper polygon (optional, based on your specific need)
        sorted_points = sort_points(intersection_points)
        all_poly.append(sorted_points)
    
    return all_poly


def oriented_box_iou(rect1, rect2):
    """
    rect1, rect2: Nx(x,y,w,h,theta)
    """
    # Compute intersecting polygon
    rect1_corners = xywht2corner(rect1)
    rect2_corners = xywht2corner(rect2)
    polys = oriented_box_intersect(rect1_corners, rect2_corners)
    intersect_area = torch.zeros(size=(len(polys),))

    for i in range(len(polys)):
        intersect_area[i] = polygon_area(polys[i].reshape(1,-1,2))

    # Compute areas
    rect1_area = rect1[:,2] * rect1[:,3]
    rect2_area = rect2[:,2] * rect2[:,3]
    return (intersect_area/(rect1_area+rect2_area-intersect_area)), intersect_area, rect1_area, rect2_area

def quad_area(corners):
    """
    Compute areas using cross product (determinant)
    corners: Nx4x(x,y)
    """
    matr_1 = torch.stack((
                torch.stack((corners[:,3,:] - corners[:,0,:], corners[:,1,:] - corners[:,0,:]),dim=1),
                torch.stack((corners[:,3,:] - corners[:,2,:], corners[:,1,:] - corners[:,2,:]),dim=1)),
                dim=1).to(torch.float64)
    area = torch.sum(torch.abs(torch.linalg.det(matr_1)),dim=1) / 2
    return area

def oriented_quad_iou(rect1_corners, rect2_corners):
    """
    rect1, rect2: Nx4x(x,y)
    """
    # Compute intersecting polygon
    polys = oriented_box_intersect(rect1_corners, rect2_corners)
    intersect_area = torch.zeros(size=(len(polys),))

    for i in range(len(polys)):
        intersect_area[i] = polygon_area(polys[i].reshape(1,-1,2))

    # Compute areas using cross product (determinant)
    rect1_area = quad_area(rect1_corners)
    rect2_area = quad_area(rect2_corners)
    
    return (intersect_area/(rect1_area+rect2_area-intersect_area)), intersect_area, rect1_area, rect2_area



def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    Does Non Max Suppression given bboxes

    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU) 
        box_format (str): "midpoint" or "corners" used to specify bboxes

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def mean_average_precision(
    pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20
):
    """
    Calculates mean average precision 

    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones 
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes

    Returns:
        float: mAP value across all classes given a specific IoU threshold 
    """

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)
        
        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))
        print("length of the average_precisions is: ", len(average_precisions))


    return sum(average_precisions) / len(average_precisions)



def plot_image(image, boxes):
    """Plots predicted bounding boxes on the image"""
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle potch
    for box in boxes:
        #-----------------------------start_modification----------------------
        import json 

        with open("label_lookup_table.json") as f:
            data = json.load(f)
            label_lookup_table = data["labels"]
        label_index = int (box[0])
        confident_rate = box[1]
        label_name = label_lookup_table[label_index]

        #------------------------------end_modification-----------------------
        box = box[2:]
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)

        #-----------------------------start_modification----------------------
        # Print class name on the left top corner of the box
        ax.text(
            upper_left_x * width,
            upper_left_y * height,
            (label_name + "  con_rate:" + str(round(confident_rate,2))),
            fontsize=10,
            color="b",
            # bbox=dict(facecolor="white", alpha=0.5),
        )
        #------------------------------end_modification-----------------------

    plt.show()

def get_bboxes(
    loader,
    model,
    iou_threshold,
    threshold,
    pred_format="cells",
    box_format="midpoint",
    device="cuda",
):
    all_pred_boxes = []
    all_true_boxes = []

    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0

    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device)

        print(labels[0][6][5])

        exit(0)
        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels)
        bboxes = cellboxes_to_boxes(predictions)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )


            #if batch_idx == 0 and idx == 0:
            #    plot_image(x[idx].permute(1,2,0).to("cpu"), nms_boxes)
            #    print(nms_boxes)

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                # many will get converted to 0 pred
                print("the box is: ", box)
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes



def convert_cellboxes(predictions, S=7):
    """
    Converts bounding boxes output from Yolo with
    an image split size of S into entire image ratios
    rather than relative to cell ratios. Tried to do this
    vectorized, but this resulted in quite difficult to read
    code... Use as a black box? Or implement a more intuitive,
    using 2 for loops iterating range(S) and convert them one
    by one, resulting in a slower but more readable implementation.
    """

    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, 7, 7, 30)
    bboxes1 = predictions[..., 21:25]
    bboxes2 = predictions[..., 26:30]
    scores = torch.cat(
        (predictions[..., 20].unsqueeze(0), predictions[..., 25].unsqueeze(0)), dim=0
    )
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = 1 / S * best_boxes[..., 2:4]
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)
    predicted_class = predictions[..., :20].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., 20], predictions[..., 25]).unsqueeze(
        -1
    )
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )

    return converted_preds


def cellboxes_to_boxes(out, S=7):
    converted_pred = convert_cellboxes(out).reshape(out.shape[0], S * S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(S * S):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

import json
def generate_box_center_list(bounding_boxes,index):
    num = len(bounding_boxes)
    boxes = torch.tensor(bounding_boxes)
    boxes = boxes[:,2:]
    middle_point_matrix = torch.zeros(num,2)
    for index, box in enumerate(boxes):
        middle_x = round(float ((box[0] + box[2]) / 2), 2)
        middle_y = round(float ((box[1] + box[3]) / 2),2)
        print ("the middle point is: ", middle_x, middle_y)
        middle_point_matrix[index][0] = middle_x
        middle_point_matrix[index][1] = middle_y
        middle_point_list = middle_point_matrix.tolist()

    dictionary = {
        "index" : index,
        "boxes" : middle_point_list
    }

    # with open()


def plot_oriented_iou(rect1_corners, rect2_corners, id, val1, val2, val3, prefix=""):
    fig, (ax1, axc) = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'aspect': 'equal'})

    polys = oriented_box_intersect(rect1_corners[id,...].reshape(1,-1,2), rect2_corners[id,...].reshape(1,-1,2))

    rect1np = rect1_corners[id,...].numpy()
    rect2np = rect2_corners[id,...].numpy()
    polynp = polys[0].numpy()

    val1 = round(val1.item(),2)
    
    ax1.fill(rect1np[:,0],rect1np[:,1],facecolor='none', edgecolor='blue', label=f"box 1: {val2}", linewidth=4)
    ax1.fill(rect2np[:,0], rect2np[:,1],facecolor='none', edgecolor='magenta', label=f"box 2: {val3}", linewidth=3)
    ax1.legend()
    axc.fill(rect1np[:,0],rect1np[:,1],facecolor='none', edgecolor='blue', label=f"box 1: {val2}", linewidth=4)
    axc.fill(rect2np[:,0], rect2np[:,1],facecolor='none', edgecolor='magenta', label=f"box 2: {val3}", linewidth=3)
    axc.fill(polynp[:,0], polynp[:,1],facecolor='wheat', edgecolor='purple', label=f"intersection: {val1}", linewidth=2)
    axc.legend()
    plt.savefig(prefix+f"oriented_fig_{id}.png")

def test_iou():
    # Test iou corners
    print("TESTING IOU CORNERS")
    box_1 = torch.tensor([[0,0,2,2],[0,0,2,2],[0,0,2,2],[0,0,2,2]])
    box_2 = torch.tensor([[1,1,2,2],[2,2,5,3],[0,0,2,2],[0.5,0.5,1.5,4.5]])
    
    iou = iou_corners(box_1, box_2)
    iou_truth = [0.25, 0, 1, 1.5/6.5]
    for i in range(len(iou_truth)):
        print("Computed IOU:", iou[i], "expected", iou_truth[i])

    # Test iou xywh
    print("TESTING IOU XYWH")
    box_1_xywh = torch.tensor([[1,1,2,2],[1,1,2,2],[1,1,2,2],[1,1,2,2]])
    box_2_xywh = torch.tensor([[1.5,1.5,1,1],[3.5,2.5,3,1],[1,1,2,2],[1,2.5,1,4]])
    iou = iou_xywh(box_1_xywh, box_2_xywh)
    print(iou)
    for i in range(len(iou_truth)):
        print("Computed IOU:", iou[i], "expected", iou_truth[i])


def test_oriented_iou():
    # Test iou corners
    print("TESTING ORIENT IOU CORNERS")
    box_1 = torch.tensor([ [1,1,2,2,0]
                          ,[1,1,2,2,0]
                          ,[1,1,2,2,0]
                          ,[1,1,2,2,0]
                          ,[1,1,2,2,0]
                          ,[1,1,2,2,0]
                          ,[1,1,2,2,0]
                          ,[1,1,3,7,torch.pi*3/7]
                          ,[3,1,3,7,torch.pi*3/7]
                        ])
    box_2 = torch.tensor([ [1.5,1.5,1,1,torch.pi]
                          ,[3.5,2.5,3,1,torch.pi/2]
                          ,[1,1,2,2,-torch.pi]
                          ,[1,2.5,1,4,0]
                          ,[1,1,1,1,torch.pi/5]
                          ,[2,3,2,10,-torch.pi/3.6]
                          ,[1,1,2,5,-torch.pi/3]
                          ,[1,1,2,5,-torch.pi/3]
                          ,[1,3,2,5,-torch.pi/6]
                        ])

    # Test Box Lines    
    iou, inta, rect1a, rect2a = oriented_box_iou(box_1, box_2)
    iou_truth = [0.25, 0, 1, 1.5/6.5, 0.25, -1, -1, -1, -1]
    # print("Computed IOU:", iou)
    for i in range(len(iou_truth)):
        print("Computed IOU:", iou[i], "expected", iou_truth[i])
        rect1_corners = xywht2corner(box_1)
        rect2_corners = xywht2corner(box_2)
        plot_oriented_iou(rect1_corners, rect2_corners, i, val1=inta[i], val2=rect1a[i],val3=rect2a[i])

def test_quad_iou():
    # Test iou corners
    print("TESTING ORIENT IOU CORNERS")
    quad_1 = torch.tensor([ [1,1,4,2,5,7,0,4]
                        ]).reshape(1,-1,2)
    quad_2 = torch.tensor([ [1,3,3,2,8,7,2,6]
                        ]).reshape(1,-1,2)

    # Test Box Lines    
    quad_iou, inta, rect1a, rect2a = oriented_quad_iou(quad_1, quad_2)
    quad_iou_truth = [-1]
    for i in range(len(quad_iou_truth)):
        print("Computed IOU:", quad_iou[i], "expected", quad_iou_truth[i])
        plot_oriented_iou(quad_1, quad_2, i, val1=inta[i], val2=rect1a[i],val3=rect2a[i], prefix="quad_")

def test_oriented_box_utils():
    # Test iou corners
    print("TESTING ORIENT IOU UTILS")
    box_1 = torch.tensor([[1.0,1.0,2.0,2,0],[1.0,1.0,2,2,torch.pi/4]]).reshape(2,-1)
    box_2 = torch.tensor([[1.5,1.5,1,1,0],[1.5,1.5,1,1,torch.pi/4]]).reshape(2,-1)

    # Test boxes
    box_1_corner = xywht2corner(box_1)
    box_2_corner = xywht2corner(box_2)
    print("Box 1 Corners:", box_1_corner)
    print("Box 2 Corners:", box_2_corner)

    # Test Polygon Size
    box_1_area = polygon_area(box_1_corner)
    box_2_area = polygon_area(box_2_corner)
    print("Box 1 Area:", box_1_area, "expected both",4)
    print("Box 2 Area:", box_2_area, "expected both",1)

    # Test intersect
    print("Intersect test parallel:", points_in_polygon(torch.tensor([[-1.0,1.0,1.0,1.0]]).reshape(1,4), torch.tensor([[0.0,0.0,2.0,0.0]])))
    print("Intersect test parallel vertical:", points_in_polygon(torch.tensor([[1.0,-1.0,1.0,3.0]]).reshape(1,4), torch.tensor([[0.0,0.0,0.0,4.0]])))
    print("Intersect test overlap:", points_in_polygon(torch.tensor([[-1.0,1.0,1.0,1.0]]).reshape(1,4), torch.tensor([[1.0,1.0,2.0,1.0]])))
    print("Intersect test overlap vertical:", points_in_polygon(torch.tensor([[0.0,1.0,0.0,5.0]]).reshape(1,4), torch.tensor([[0.0,0.0,0.0,3.0]])))
    print("Intersect test actual:", points_in_polygon(torch.tensor([[-1.0,1.0,1.0,1.0]]).reshape(1,4), torch.tensor([[0.0,0.0,1.0,2.0]])))
    print("Intersect test no intersect:", points_in_polygon(torch.tensor([[-1.0,1.0,1.0,1.0]]).reshape(1,4), torch.tensor([[0.0,0.0,2.0,2.0]])))

    # Test point in box
    print("Inside test:", points_in_polygon(torch.tensor([[1.0,1.0]]).reshape(1,2), box_2_corner[0,...]))
    print("Edge test:", points_in_polygon(torch.tensor([[2.0,1.0]]).reshape(1,2), box_2_corner[0,...]))
    print("Corner test:", points_in_polygon(torch.tensor([[2.0,2.0]]).reshape(1,2), box_2_corner[0,...]))
    print("Outside test:", points_in_polygon(torch.tensor([[3.0,3.0]]).reshape(1,2), box_2_corner[0,...]))

if __name__ == "__main__":
    # test_iou()
    test_oriented_iou()
    test_quad_iou()
    # test_oriented_box_utils()
