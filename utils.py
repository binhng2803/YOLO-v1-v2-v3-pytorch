import torch
from collections import Counter


def intersection_over_union(boxes_preds, boxes_labels, box_format='midpoint'):
    '''
    Parameters:
        boxes_preds (tensor): predictions of bounding boxes (N, 4), N is number of bounding boxes
        boxes_preds (tensor): Correct labels of bounding boxes (N, 4)
        box_format (str): midpoint=(x, y, w, h), corners=(x1, y1, x2, y2)
    Returns:
        IoU (tensor): intersection over union for all boxes
    '''
    if box_format=='midpoint':
        # boxes_preds shape: (N, 4), each box contain: (x, y, w, h)
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3]/2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4]/2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3]/2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4]/2
        # boxes_labels shape: (N, 4), each box contain: (x, y, w, h)
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3]/2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4]/2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3]/2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4]/2
    
    elif box_format=='corners':
        # boxes_preds shape: (N, 4), each box contain: (x1, y1, x2, y2)
        box1_x1 = boxes_preds[..., 0:1] 
        box1_y1 = boxes_preds[..., 1:2] 
        box1_x2 = boxes_preds[..., 2:3] 
        box1_y2 = boxes_preds[..., 3:4] 
        # boxes_labels shape: (N, 4), each box contain: (x1, y1, x2, y2)
        box2_x1 = boxes_labels[..., 0:1] 
        box2_y1 = boxes_labels[..., 1:2] 
        box2_x2 = boxes_labels[..., 2:3] 
        box2_y2 = boxes_labels[..., 3:4] 
        
    # find the corners of intersection area
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)
    # intersection area
    intersection = (x2 - x1).clamp(0)*(y2 - y1).clamp(0) # use .clamp is for the case when boxes do not intersect
    
    # union area
    box1_area = abs((box1_x2 - box1_x1)*(box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1)*(box2_y2 - box2_y1))
    union = box1_area + box2_area - intersection
    
    IoU = intersection/(union + 1e-6)
    return IoU  



def nms(bboxes, iou_threshold, prob_threshold, box_format='midpoint'):
    '''
    Parameters:
        bboxes (list): list of bounding boxes, each box contains [class, prob, x, y, w h] or [class, prob, x1, y1, x2, y2]
        iou_threshold (float): 
        prob_threshold (float):
        box_format (str): midpoint=(x, y, w, h), corners=(x1, y1, x2, y2)
    Returns:
        bboxes_after_nms (list)
    '''
    # keep bounding boxes which have prob > prob_threshold
    bboxes = [box for box in bboxes if box[1]>prob_threshold]
    # sort the bboxes in descending order by prob
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []
    while bboxes:
        # choose the box with highest prob
        chosen_box = bboxes.pop(0)
        # just keep boxes which do not have same class with chosen_box or having IoU with chosen_box < iou_threshold
        bboxes = [box for box in bboxes 
                 if box[0] != chosen_box[0] 
                  or intersection_over_union(
                    torch.tensor(chosen_box[2:]),
                    torch.tensor(box[2:]),
                    box_format = box_format) < iou_threshold]
        bboxes_after_nms.append(chosen_box)
    return bboxes_after_nms    


def mAP(pred_boxes, true_boxes, iou_threshold=0.5, box_format='midpoint', num_classes=20):
    '''
    Parameters:
        pred_boxes (list): all bboxes with each box contained (training_img_idx, class, prob, x1, y1, x2, y2) or (train_idx, class, prob, x1, y1, x2, y2)
        true_boxes (list): all true boxes with each box contained (training_img_idx, class, prob, x1, y1, x2, y2) or (train_idx, class, prob, x1, y1, x2, y2)
        iou_threshold (float):
        box_format (str):
        num_class (int): number of classes in your dataset
    Returns:
        mAP (float): mAP value across all classes given a specific iou_theshold
    '''
    # list will store all AP for respective classes
    average_precisions = []
    
    for c in range(num_classes):
        detections = []
        ground_truths = []
        
        # get all the boxes in pred_boxes and true_boxes which have class=c
        for detection in pred_boxes:
            if detection[1]==c:
                detections.append(detection)
        for true_box in true_boxes:
            if true_box[1]==c:
                ground_truths.append(true_box)
        
        # count bboxes detect this class in each training image, for examples: image_idx=0 has 4 bboxes, image_idx=1 has 2 bboxes then we has {0:4, 1:2}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])
        
        # convert the key in amount_bboxes like from {0:4, 1:2} -> {0:tensor[0, 0, 0, 0], 1: tensor[0, 0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)
        
        # sort the detections in descending order by prob
        detections = sorted(detections, key=lambda x:x[2], reverse=True)
        
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)
        
        if total_true_bboxes==0:
            continue
        
        for detection_idx, detection in enumerate(detections):
            # only take out the ground truths in the same image which has the same training idx as detection
            ground_truth_img = [bbox for bbox in ground_truths if bbox[0]==detection[0]]
            num_gts = len(ground_truth_img)
            best_iou = 0
            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(torch.tensor(gt[3:]), torch.tensor(detection[3:]), box_format=box_format)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
            if best_iou > iou_threshold:
                # only detect the ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx]==0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else: 
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1
                
            TP_cumsum = torch.cumsum(TP, dim=0)
            FP_cumsum = torch.cumsum(FP, dim=0)
            recalls = TP_cumsum/(total_true_bboxes + 1e-6)
            precisions = TP_cumsum/(TP_cumsum + FP_cumsum + 1e-6)
            recalls = torch.cat((torch.tensor([0]), recalls))
            precisions = torch.cat((torch.tensor([1]), precisions))
            # torch.trapz is used to calculate the intergration
            ap = torch.trapz(precisions, recalls)
            average_precisions.append(ap)
    mean_average_precision = sum(average_precisions)/len(average_precisions)
    return mean_average_precision