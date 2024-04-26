import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter

# def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
#     """
#     Calculates intersection over union

#     Parameters:
#         boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
#         boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
#         box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

#     Returns:
#         tensor: Intersection over union for all examples
#     """

#     if box_format == "midpoint":
#         box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
#         box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
#         box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
#         box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
#         box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
#         box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
#         box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
#         box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

#     if box_format == "corners":
#         box1_x1 = boxes_preds[..., 0:1]
#         box1_y1 = boxes_preds[..., 1:2]
#         box1_x2 = boxes_preds[..., 2:3]
#         box1_y2 = boxes_preds[..., 3:4]  # (N, 1)
#         box2_x1 = boxes_labels[..., 0:1]
#         box2_y1 = boxes_labels[..., 1:2]
#         box2_x2 = boxes_labels[..., 2:3]
#         box2_y2 = boxes_labels[..., 3:4]

#     x1 = torch.max(box1_x1, box2_x1)
#     y1 = torch.max(box1_y1, box2_y1)
#     x2 = torch.min(box1_x2, box2_x2)
#     y2 = torch.min(box1_y2, box2_y2)

#     # .clamp(0) is for the case when they do not intersect
#     intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

#     box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
#     box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

#     return intersection / (box1_area + box2_area - intersection + 1e-6)



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

def area_corners(box):
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
    
    inter_area = area_corners(intersect_box)
    pred_area = area_corners(box_pred)
    gt_area = area_corners(box_gt)
    return inter_area / (pred_area + gt_area - inter_area + 1e-6)


def get_oriented_bbox_corners(x, y, h, w, theta):
    # Define the center coordinates
    center = torch.tensor([x, y], dtype=torch.float32)
    
    # Half dimensions to simplify corner calculations
    half_width = w / 2
    half_height = h / 2
    
    # Corners of the rectangle in local coordinate space (unrotated)
    corners = torch.tensor([
        [-half_width, -half_height],
        [half_width, -half_height],
        [half_width, half_height],
        [-half_width, half_height]
    ], dtype=torch.float32)  # shape [4, 2]
    
    # Rotation matrix
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    rotation_matrix = torch.tensor([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta]
    ], dtype=torch.float32)  # shape [2, 2]
    
    # Rotate corners
    rotated_corners = torch.matmul(corners, rotation_matrix.t())  # Transpose to align dimensions
    
    # Translate corners to global coordinate space
    oriented_corners = rotated_corners + center
    
    return oriented_corners


def get_polygon_area(rect1, rect2):
    """
    rect1, rect2 must be both arrays of shape (8,), with (x1,y1,x2,y2,x3,y3,x4,y4) in that order, clockwise.
    """
    pass



def oriented_iou(rect1, rect2):
    pass




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



if __name__ == "__main__":
    test_iou()

