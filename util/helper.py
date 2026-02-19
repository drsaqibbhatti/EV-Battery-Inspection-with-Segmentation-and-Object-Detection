import torch
import numpy as np
import torch.nn.functional as F
import math
import os
import torch.nn as nn
from torch import Tensor
from typing import Optional, List, Tuple
import cv2

from PIL import Image




def sorting_key(file_path):
    # Extract the base file name without extension
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    # Normalize the name by removing known suffixes like '_mask'
    normalized_name = base_name.replace('_mask', '')
    return normalized_name


# region Check Images and their pair masks for Segmentation Tasks

def extract_filename(full_path):
    return os.path.splitext(os.path.basename(full_path))[0]

def check_correspondence(images_list, masks_list):
    
    image_names = [extract_filename(image) for image in images_list]
    mask_names = [extract_filename(mask).replace('_mask', '') for mask in masks_list]

    missing_masks = [image for image in images_list if extract_filename(image) not in mask_names]
    missing_images = [mask for mask in masks_list if extract_filename(mask).replace('_mask', '') not in image_names]

    return missing_masks, missing_images



def check_missmatched_pairs(images_list, masks_list):
    # Assuming lists are sorted and corresponding files are at same indices
    mismatched_pairs = []
    for image, mask in zip(images_list, masks_list):
        image_name = extract_filename(image)
        mask_name = extract_filename(mask).replace('_mask', '')

        if image_name != mask_name:
            mismatched_pairs.append((image, mask))

    return mismatched_pairs

def IOU2(target, prediction, threshold=0.5):
    """
    Compute the Intersection over Union (IoU) between the ground truth and prediction masks.
    
    Args:
        target (np.array): Ground truth binary mask (0s and 1s, or values between 0 and 1).
        prediction (np.array): Predicted mask, values between 0 and 1.
        threshold (float): Threshold to binarize the predicted mask (default 0.5).
        
    Returns:
        iou_score (float): IOU score.
    """
    # Threshold the prediction to convert probabilities to binary values (0 or 1)
    prediction = np.where(prediction > threshold, 1, 0)
    
    # Compute intersection and union
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    
    # Summation of the union
    summation = np.sum(union)

    # If both prediction and target are empty, return IoU of 1 (perfect match)
    if summation == 0:
        return 1.0

    # IoU is the ratio of the intersection to the union
    iou_score = np.sum(intersection) / summation
    return iou_score



def IOU(target, prediction):
    prediction = np.where(prediction > 0.5, 1, 0)
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    summation = np.sum(union)

    if summation == 0:
        return 0

    iou_score = np.sum(intersection) / summation
    return iou_score

def IOU_pinhole_spatter(target, prediction):
    """
    Calculates the IoU for two classes: pinhole and spatter.
    
    Assumes `target` and `prediction` are one-hot encoded for two classes without a background:
    - Class 1: pinhole
    - Class 2: spatter
    """
    iou_scores = []
    
    # Loop over the batch
    for batch_index in range(target.shape[0]):
        # Binary threshold for prediction
        pred_batch = np.where(prediction[batch_index] > 0.5, 1, 0)
        target_batch = target[batch_index]  # Take a single batch

        # Loop over the two classes: 0 for pinhole, 1 for spatter
        for cls in range(2):
            # Create binary masks for the current class
            target_class = target_batch[cls]  # class 0 (pinhole) or class 1 (spatter)
            pred_class = pred_batch[cls]  # class 0 (pinhole) or class 1 (spatter)

            # Calculate intersection and union for this class
            intersection = np.logical_and(target_class, pred_class)
            union = np.logical_or(target_class, pred_class)

            summation = np.sum(union)

            if summation == 0:
                iou_score = 0  # Avoid division by zero if no pixels of this class exist
            else:
                iou_score = np.sum(intersection) / summation

            iou_scores.append(iou_score)

    # Return the mean IoU across both classes (pinhole and spatter)
    mean_iou = np.mean(iou_scores)
    return mean_iou




def IOU_pinholespatter_2(target, prediction, num_classes=3):
    iou_scores = []

    # Convert predictions to class indices using argmax
    prediction = np.argmax(prediction, axis=1)  # Shape: [batch_size, height, width]

    # Convert one-hot encoded target to class indices
    target = np.argmax(target, axis=1)  # Shape: [batch_size, height, width]

    # Loop over each class (0: background, 1: pinhole, 2: spatter)
    for cls in range(num_classes):
        # Create binary masks for the current class
        target_class = (target == cls).astype(np.uint8)  # Shape: [batch_size, height, width]
        pred_class = (prediction == cls).astype(np.uint8)  # Shape: [batch_size, height, width]

        # Calculate intersection and union for this class
        intersection = np.logical_and(target_class, pred_class)
        union = np.logical_or(target_class, pred_class)

        # Calculate IoU for this class
        summation = np.sum(union)

        if summation == 0:
            iou_score = np.nan  # Avoid division by zero if no pixels of this class exist
        else:
            iou_score = np.sum(intersection) / summation

        iou_scores.append(iou_score)

    # Return the mean IoU (mIoU)
    mean_iou = np.nanmean(iou_scores)
    return mean_iou


# # Function to convert segmentation masks to BGR for visualization (OpenCV uses BGR)
# def mask_to_bgr(mask, num_classes=2):
#     # Create an empty BGR image
#     h, w = mask.shape
#     bgr_image = np.zeros((h, w, 3), dtype=np.uint8)

#     # Define BGR colors for each class
#     colors = {
#         2: [0, 0, 0],       # Background: Black
#         0: [0, 0, 255],     # Pinhole: Red (in BGR)
#         1: [255, 0, 0],     # Spatter: Blue (in BGR)
#     }

#     # Map each class to its corresponding BGR color
#     for cls in range(num_classes):
#         bgr_image[mask == cls] = colors[cls]

#     return bgr_image

def mask_to_color(mask, color):
    """ Converts a binary mask to a color mask for visualization. """
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    colored_mask[mask > 0.5] = color  # Apply color to mask where pixel value > 0.5
    return colored_mask

def map_labels_to_colors(image):
    """
    Maps the multi-class labels to colors for visualization.
    - 0 -> background (black)
    - 1 -> pinhole (red)
    - 2 -> spatter (green)
    """
    color_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    # Map each class to a color
    #olor_image[image == 0] = [0, 0, 0]     # Black for background
    color_image[image == 0] = [0, 0, 255]   # Red for pinhole
    color_image[image == 1] = [0, 255, 0]   # Green for spatter

    return color_image



class DiceLoss(torch.nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        Dice_BCE = dice_loss

        return Dice_BCE
    

class FocalTverskyLoss(torch.nn.Module):
    def __init__(self, alpha=0.2, beta=0.8, gamma=2, smooth=1e-6):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.gamma = gamma


    def forward(self, inputs, targets):
        # comment out if your model contains a sigmoid or equivalent activation layer


        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        FocalTversky = (1 - Tversky) ** self.gamma

        return FocalTversky
    


class FocalTverskyLoss_bg(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=2, smooth=1e-6, weight_bg=0.1):
        """
        FocalTversky Loss with weighted adjustment for images without labels (background only).
        :param alpha: controls weight given to false positives
        :param beta: controls weight given to false negatives
        :param gamma: focal term to emphasize harder examples
        :param smooth: smoothing factor to avoid division by zero
        :param weight_bg: weight assigned to images without any positive labels
        """
        super(FocalTverskyLoss_bg, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.weight_bg = weight_bg  # Weight for background-only images (no labels)

    def forward(self, inputs, targets):
        # Flatten inputs and targets to compute per-image loss
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Calculate True Positives, False Positives, and False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        # Calculate Tversky Index
        Tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        FocalTversky = (1 - Tversky) ** self.gamma

        # Check if this image has no labels (i.e., it's all background)
        if targets.sum() == 0:
            # If it's background-only, apply the background weight
            return self.weight_bg * FocalTversky
        else:
            # Otherwise, use the normal loss
            return FocalTversky

    
    
def JaccardLoss(logits, true, eps=1e-7):
    """Computes the Jaccard loss, a.k.a the IoU loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the jaccard loss so we
    return the negated jaccard loss.
    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        jacc_loss: the Jaccard loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    union = cardinality - intersection
    jacc_loss = (intersection / (union + eps)).mean()
    return (1 - jacc_loss)

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE


def count_pixels(dataset):
    """
    Count positive and negative pixels in all masks of the dataset.
    
    Args:
        dataset (Dataset): A PyTorch Dataset containing images and masks.
    
    Returns:
        total_pos_pixels (int): Total number of positive pixels in the dataset.
        total_neg_pixels (int): Total number of negative pixels in the dataset.
    """
    total_pos_pixels = 0
    total_neg_pixels = 0

    for idx in range(len(dataset)):
        _, torch_label = dataset[idx]  # Get the mask for each item in the dataset
        mask = torch_label.squeeze().cpu().numpy()  # Convert to NumPy array if needed
        
        pos_pixels = np.sum(mask == 1)
        neg_pixels = np.sum(mask == 0)
        
        total_pos_pixels += pos_pixels
        total_neg_pixels += neg_pixels

    return total_pos_pixels, total_neg_pixels



def precision_score(y_true, y_pred):
    # Flatten arrays to handle multi-dimensional input
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    
    precision = TP / (TP + FP + 1e-7)  # Add small epsilon to avoid division by zero
    return precision

def recall_score(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    recall = TP / (TP + FN + 1e-7)  # Add small epsilon to avoid division by zero
    return recall

def f1_score(precision, recall):
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)  # Add small epsilon to avoid division by zero
    return f1

# Function to create an image from metrics (accuracy, loss, etc.)
def create_metrics_image(epoch, avg_cost, avg_acc, val_cost, val_acc, learning_rate):
    # Create a blank image for displaying text
    img = np.zeros((200, 600, 3), dtype=np.uint8)

    # Define font and scale
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (255, 255, 255)  # White text
    thickness = 1

    # Prepare text strings
    text1 = f'Epoch: {epoch}'
    text2 = f'Train Loss: {avg_cost:.4f}, Train Accuracy: {avg_acc:.4f}'
    text3 = f'Val Loss: {val_cost:.4f}, Val Accuracy: {val_acc:.4f}'
    text4 = f'Learning Rate: {learning_rate:.6f}'

    # Put text onto the image
    cv2.putText(img, text1, (20, 40), font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.putText(img, text2, (20, 80), font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.putText(img, text3, (20, 120), font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.putText(img, text4, (20, 160), font, font_scale, color, thickness, cv2.LINE_AA)

    return img


# Function to create a composite image with original image, binary mask, and overlay
def overlay_with_original(input_image, mask_tensor, overlay_image, vstack=True):
    # Convert the binary mask to a format suitable for visualization
    mask_np = mask_tensor.squeeze().cpu().numpy()  # Ground truth mask (HxW)
    mask_np = (mask_np * 255).astype(np.uint8)  # Convert binary mask to [0, 255]
    
    # Convert the binary mask to 3-channel for consistency with the original image
    mask_np = cv2.cvtColor(mask_np, cv2.COLOR_GRAY2BGR)  # Convert to 3 channels (binary mask visualization)

    # Stack the images vertically (original image, mask, overlay)
    composite_image = np.vstack((input_image, mask_np, overlay_image))
    if vstack==False:
        composite_image = np.hstack((input_image, mask_np, overlay_image))
    return composite_image



def save_checkpoint(state, model, checkpoint_dir, epoch):
    # Remove the previous checkpoint only if the epoch is greater than 0
    if epoch > 0:
        # Remove the previous checkpoint
        previous_checkpoint = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch - 1}.pth')
        if os.path.exists(previous_checkpoint):
            try:
                os.remove(previous_checkpoint)
                print(f"Previous checkpoint {previous_checkpoint} removed.")
            except Exception as e:
                print(f"Error removing previous checkpoint: {e}")
        else:
            print(f"No previous checkpoint found at {previous_checkpoint}")
        
        # Remove the previous scripted model
        previous_scripted_model = os.path.join(checkpoint_dir, f'scripted_model_epoch_{epoch - 1}.pt')
        if os.path.exists(previous_scripted_model):
            try:
                os.remove(previous_scripted_model)
                print(f"Previous scripted model {previous_scripted_model} removed.")
            except Exception as e:
                print(f"Error removing previous scripted model: {e}")
        else:
            print(f"No previous scripted model found at {previous_scripted_model}")
    
    # Save the new checkpoint
    filename = f'checkpoint_epoch_{epoch}.pth'
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save(state, checkpoint_path)

    # Script the model and save it as part of the checkpointing process
    # Check if the model is wrapped in DataParallel
    if isinstance(model, torch.nn.DataParallel):
        model = model.module  # Unwrap DataParallel to get the original model

    # Save the new scripted model
    scripted_model_path = os.path.join(checkpoint_dir, f'scripted_model_epoch_{epoch}.pt')
    scripted_model = torch.jit.script(model)
    torch.jit.save(scripted_model, scripted_model_path)

    print(f"Checkpoint saved at {checkpoint_path}, scripted model saved at {scripted_model_path}")



def load_checkpoint(checkpoint_path, model, optimizer):
    # Load the optimizer state and other metadata
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Resuming training from epoch {epoch}")
    return epoch


def apply_motion_blur(image, kernel_size=2):
    """Applies horizontal motion blur to the input image."""
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[:, int((kernel_size - 1)/2)] = np.ones(kernel_size)
    kernel /= kernel_size

    # Apply the motion blur
    motion_blur = cv2.filter2D(image, -1, kernel)
    return motion_blur


def to_one_hot(tensor,nClasses,device):
    n,h,w = tensor.size()
    one_hot = torch.zeros(n,nClasses,h,w).to(device).scatter_(1,tensor.view(n,1,h,w),1)
    return one_hot

class mIoULoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True, n_classes=1, device='cuda'):
        super(mIoULoss, self).__init__()
        self.classes = n_classes
        self.device = device

    def forward(self, inputs, target):
    	# inputs => N x Classes x H x W
    	# target_oneHot => N x Classes x H x W
        inputs = inputs.to(self.device)
        target = target.to(self.device)
        
        SMOOTH = 1e-6
        N = inputs.size()[0]

        #target_oneHot = to_one_hot(target, self.classes,self.device)
        # Numerator Product
        inter = inputs * target
        ## Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N,self.classes,-1).sum(2) + SMOOTH

        #Denominator 
        union= inputs + target - (inputs*target)
        ## Sum over all pixels N x C x H x W => N x C
        union = union.view(N,self.classes,-1).sum(2) + SMOOTH

        loss = inter/union

        ## Return average loss over classes and batch
        return 1-loss.mean()
    
    

import time
import torchvision
def wh2xy(x):
    y = x.clone()
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def non_max_suppression(prediction, conf_threshold=0.25, iou_threshold=0.45):
    nc = prediction.shape[1] - 4  # number of classes
    xc = prediction[:, 4:4 + nc].amax(1) > conf_threshold  # candidates

    # Settings
    max_wh = 512  # (pixels) maximum box width and height
    max_det = 30 # the maximum number of boxes to keep after NMS
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()

    start = time.time()
    outputs = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for index, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        x = x.transpose(0, -1)[xc[index]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (box, conf, cls)
        box, cls = x.split((4, nc), 1)
        # center_x, center_y, width, height) to (x1, y1, x2, y2)
        box = wh2xy(box)
        if nc > 1:
            i, j = (cls > conf_threshold).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_threshold]
        # Check shape
        if not x.shape[0]:  # no boxes
            continue
        # sort by confidence and remove excess boxes
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        c = x[:, 5:6] * max_wh  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_threshold)  # NMS
        i = i[:max_det]  # limit detections
        outputs[index] = x[i]
        if (time.time() - start) > 0.5 + 0.05 * prediction.shape[0]:
            print(f'WARNING ⚠️ NMS time limit {0.5 + 0.05 * prediction.shape[0]:.3f}s exceeded')
            break  # time limit exceeded

    return outputs
