import cv2
import torch
import numpy as np
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from model.CustomSegmentationV4 import CustomSegmentationV4
from util.PinholeDataset import PinholeDataset
from util.helper import overlay_with_original
from tqdm import tqdm
import pandas as pd
import os

def evaluate_model(model_path, data_path, save_dir, generate_overlays=False, overlay_dir=None):
    # Check if CUDA is available
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    print("The device is:", device)

    # Load the model
    model = torch.load(model_path, map_location=device)
    model.eval()

    # Transformations
    imageWidth = 640
    imageHeight = 640

    transformNormalCollection = []
    transformNormalCollection.append(transforms.Resize((imageHeight, imageWidth)))
    transformNormalCollection.append(transforms.ToTensor())
    transNormalProcess = transforms.Compose(transformNormalCollection)

    # Dataset and DataLoader
    testDataset = PinholeDataset(path=data_path,
                                      transform=transNormalProcess,
                                      category="test",
                                      useVFlip=False,
                                      useHFlip=False)

    testDatasetLoader = DataLoader(testDataset, batch_size=1, shuffle=False)

    # Create directory to save overlay masks if needed
    if generate_overlays and overlay_dir:
        os.makedirs(overlay_dir, exist_ok=True)

    # Metrics variables
    inference_times = []
    image_accuracies = []
    image_precisions = []
    image_recalls = []
    image_f1scores = []
    conf_matrix = np.zeros((2, 2))  # Initialize confusion matrix for binary classification

    # Use tqdm to create a progress bar
    with torch.no_grad():
        for i, (X_test, Y_test) in enumerate(tqdm(testDatasetLoader, desc="Evaluating", total=len(testDatasetLoader))):
            X_test = X_test.to(device)
            Y_test = Y_test.to(device)

            # Inference time calculation
            start_time = time.time()
            output = model(X_test)
            end_time = time.time()
            inference_times.append(end_time - start_time)

            # Convert predictions and labels to binary format
            preds = output.cpu().numpy().round().astype(int)
            labels = Y_test.cpu().numpy().astype(int)

            preds_flat = preds.flatten()   
            labels_flat = labels.flatten()

            # Calculate metrics for the current image
            current_accuracy = accuracy_score(labels_flat, preds_flat) * 100
            current_precision = precision_score(labels_flat, preds_flat, average='binary', zero_division=0) * 100
            current_recall = recall_score(labels_flat, preds_flat, average='binary', zero_division=0) * 100
            current_f1score = f1_score(labels_flat, preds_flat, average='binary', zero_division=0) * 100
            
            # Update confusion matrix
            conf_matrix += confusion_matrix(labels_flat, preds_flat, labels=[0, 1])
        
            # Store metrics
            image_accuracies.append(current_accuracy)
            image_precisions.append(current_precision)
            image_recalls.append(current_recall)
            image_f1scores.append(current_f1score)
            
            # Generate and save overlay masks if required
            if generate_overlays and overlay_dir:
                input_image = X_test[0].cpu().numpy().transpose(1, 2, 0)  # Convert tensor to numpy and adjust channel order
                input_image = cv2.normalize(input_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)  # Normalize to [0, 255] for visualization
                
                # Create a mask for the segmented area
                predicted_mask = preds[0][0] * 255  # Convert mask to [0, 255]
                predicted_mask = predicted_mask.astype(np.uint8)

                # Find contours of the mask
                contours, _ = cv2.findContours(predicted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Draw the contours on the original image
                overlay = input_image.copy()
                cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)  # Draw red contours

                # Write accuracy text on the overlay image
                accuracy_text = f"Accuracy: {current_accuracy:.9f}%"
                precision_text = f"Precision: {current_precision:.9f}%"
                recall_text = f"Recall: {current_recall:.9f}%"
                f1_score_text = f"F1 Score: {current_f1score:.9f}%"
                
                cv2.putText(overlay, accuracy_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                cv2.putText(overlay, precision_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(overlay, recall_text, (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (95, 113, 70), 2)
                cv2.putText(overlay, f1_score_text, (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (94, 66, 109), 2)
                
                combined_images=overlay_with_original(input_image, Y_test[0], overlay, vstack=False)
                # Save overlay image
                overlay_filename = os.path.join(overlay_dir, f"precision_{current_precision:.9f}_recall_{current_recall:.9f}_accuracy_{current_accuracy:.9f}.png")
                cv2.imwrite(overlay_filename, combined_images)

    # Calculate mean metrics across all images
    mean_accuracy = np.mean(image_accuracies)
    mean_precision = np.mean(image_precisions)
    mean_recall = np.mean(image_recalls)
    mean_f1score = np.mean(image_f1scores)
    avg_inference_time = np.mean(inference_times)

    # Normalize the confusion matrix to percentage
    conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    # Output results
    print(f"Mean Image-wise Accuracy: {mean_accuracy:.2f}%")
    print(f"Mean Image-wise Precision: {mean_precision:.2f}%")
    print(f"Mean Image-wise Recall: {mean_recall:.2f}%")
    print(f"Mean Image-wise F1 Score: {mean_f1score:.2f}%")
    print(f"Average Inference Time per Image: {avg_inference_time * 1000:.4f} milliseconds")

    # Plot confusion matrix
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix_normalized * 100, annot=True, fmt=".2f", cmap="Blues", xticklabels=['Background', 'Pinhole'], yticklabels=['Background', 'Pinhole'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (in %)')

    # Save confusion matrix
    cm_filename = os.path.join(save_dir, "confusion_matrix_percentage.png")
    plt.savefig(cm_filename)
    plt.close()
    print(f"Confusion matrix saved to {cm_filename}")

    # Save metrics to CSV
    metrics = {
        'Accuracy': [mean_accuracy],
        'Precision': [mean_precision],
        'Recall': [mean_recall],
        'F1 Score': [mean_f1score],
        'Average Inference Time (ms)': [avg_inference_time * 1000]
    }
    evaluation_metrics_df = pd.DataFrame(metrics)
    evaluation_metrics_csv_path = os.path.join(save_dir, "evaluation_metrics.csv")
    evaluation_metrics_df.to_csv(evaluation_metrics_csv_path, index=False)
    print(f"Evaluation metrics saved to {evaluation_metrics_csv_path}")

if __name__ == "__main__":
    model_path = "/home/saqib/Projects/sebang_welding/sebang_welding_git/sebangweldingproject/trainedModel/Phase_2/clamp/pinhole/2024-10-23/run_clamp_1_alpha_0.7_beta_0.3_gamma_2/clamp_InspectionSegmentationV1_2024-10-23_accuracy_train_0.488358467_loss_0.066526644_epoch_72_torch_segmentation_V6_lr_0.0017399999999999972_time_70091.912093M_best.pth" 
    data_path = "/home/saqib/Projects/sebang_welding/dataset_phase_2/Clamp_inspection/dataset"
    save_dir = "home/saqib/Projects/sebang_welding/sebang_welding_git/sebangweldingproject/trainedModel/Phase_2/clamp/pinhole/2024-10-23/run_clamp_1_alpha_0.7_beta_0.3_gamma_2"  # Update with your desired folder to save outputs
    overlay_dir = "/home/saqib/Projects/sebang_welding/sebang_welding_evaluation/trainedModel/Phase_2/clamp/pinhole/2024-10-23/run_clamp_1_alpha_0.7_beta_0.3_gamma_2"  # Directory to save overlays if enabled
    os.makedirs(save_dir, exist_ok=True)  # Create the folder if it doesn't exist

    # Set generate_overlays to True or False depending on whether you want to generate overlays
    generate_overlays = True
    if generate_overlays:
        os.makedirs(overlay_dir, exist_ok=True)  # Create overlay folder only if generating overlays

    evaluate_model(model_path, data_path, save_dir, generate_overlays=generate_overlays, overlay_dir=overlay_dir)
