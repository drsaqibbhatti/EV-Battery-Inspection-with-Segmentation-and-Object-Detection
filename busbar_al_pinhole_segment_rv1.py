import cv2 
import torch
import random
import numpy as np
import time
import datetime
from datetime import date
from util.PinholeDataset import PinholeDataset
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from model.CustomSegmentationV4 import CustomSegmentationV4
# from torchsummary import summary
from util.helper import IOU
from util.helper import FocalTverskyLoss, DiceLoss, count_pixels
from util.helper import JaccardLoss
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import accuracy_score

def train(load_pre_trained_model=False):

    USE_CUDA = torch.cuda.is_available() # Returns True if the GPU is available, False otherwise
    device = torch.device("cuda" if USE_CUDA else "cpu") # Use GPU if available, otherwise use CPU
    print("The device is:", device)

    # for reproducibility
    random.seed(777)
    torch.manual_seed(777)
    if device == 'cuda':
        torch.cuda.manual_seed_all(777)



    #hyper parameter
    imageWidth=640
    imageHeight=128
    batchSize=32
    #learningRate = 0.003 * batchSize / 32 
    learningRate = 0.003
    epochs = 500
    targetAccuracy = 0.991000
    #hyper parameter
    alphaa=0.7
    betaa=0.3
    gammaa=2

    transformAugCollection = []
    transformAugCollection.append(transforms.Resize((imageHeight,imageWidth), interpolation=Image.NEAREST))
    # transformAugCollection.append(transforms.RandomAdjustSharpness(p=0.2, sharpness_factor=2))
    # transformAugCollection.append(transforms.RandomAutocontrast(p=0.2))
    transformAugCollection.append(transforms.ToTensor())
    transAugProcess = transforms.Compose(transformAugCollection)

    transformNormalCollection = []
    transformNormalCollection.append(transforms.Resize((imageHeight,imageWidth))) 
    transformNormalCollection.append(transforms.ToTensor()) 
    transNormalProcess = transforms.Compose(transformNormalCollection)


 
    
    trainDataset = PinholeDataset(path="D:\\Projects\\Projects_Work\\SebangWelding\\trainedModel\\Phase_2025\\busbar_al\\inspection",
                                      transform=transAugProcess,
                                      category="train", num_images=1)

    trainDatasetLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True, drop_last=False)

    validDataset = PinholeDataset(path="D:\\Projects\\Projects_Work\\SebangWelding\\trainedModel\\Phase_2025\\busbar_al\\inspection",
                                       transform=transNormalProcess,
                                       category="train",
                                       useVFlip=False,
                                       useHFlip=False, num_images=1)

    validDatasetLoader = DataLoader(validDataset, batch_size=1, shuffle=True, drop_last=False)

    # ####################### Pretrained model path######################################################################################
    #pretrained_model_path = "/home/saqib/Projects/sebang_welding/trainedModel/Phase_2/clamp/width/run_clamp_2/clamp_width_CustomSegmentationV4_2024-10-29_accuracy_train_0.97235_loss_0.01404_epoch_99_lr_0.003_time_1153.65M_alpha_0.7_beta_0.3_gamma_2_best.pth"
    # ###################################################################################################################################



    # summary
    totalTrainBatch = len(trainDatasetLoader)
    totalValidBatch = len(validDatasetLoader)
    CustomSegmentation = CustomSegmentationV4().to(device)
    # CustomSegmentation = SegNextV2(inputWidth=imageWidth,
    #                             inputHeight=imageHeight,
    #                             num_class=1,
    #                             embed_dims=[32, 64, 128, 128],
    #                             ffn_ratios=[2, 2, 2, 2],
    #                             depths=[3,3,3,2]).to(device)
    print('==== model info ====')
    # summary(CustomSegmentation, (3, imageHeight, imageWidth))
    print('====================')

    print('total_batch_size = ', totalTrainBatch)
    print('val_batch_size = ', totalValidBatch)
    print('learning rate = ', learningRate)
    # summaru

    # loss function setting
    

    
    # total_pos_pixels, total_neg_pixels = count_pixels(trainDataset)
    # print(f"Total positive pixels in the training dataset: {total_pos_pixels}")
    # print(f"Total negative pixels in the training dataset: {total_neg_pixels}")
    # pos_weightage=total_neg_pixels/total_pos_pixels
    # print(f"Weightage given to Pos loss: {pos_weightage}")
    # pos_weight = torch.tensor([pos_weightage*2], dtype=torch.float32).to(device)    
    # loss_fn=DiceBCELoss(weight=pos_weight).to(device) # Combined Dice BCE function
    loss_fn = FocalTverskyLoss(alpha=alphaa, beta=betaa, gamma=gammaa).to(device)
    loss_dice_fn = DiceLoss()
    dice_loss_weightage=1
        
    optimizer = torch.optim.RAdam(CustomSegmentation.parameters(), lr=learningRate)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.05, threshold=1e-4, threshold_mode='rel', patience=4, verbose=True, min_lr=0.000001)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)  # Reduce LR after 100 epochs
    #scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0009, max_lr=0.003, step_size_up=totalTrainBatch*2, mode='triangular')
    # Initialize variables to track the best model
    best_train_acc = 0.0  # Or use np.inf if tracking the lowest loss
    best_model_path = None
    Base_dir='D:\\Projects\\Projects_Work\\SebangWelding\\trainedModel\\Phase_2025\\busbar_al\\inspection'
    model_arch = "CustomSegmentationV4"
    dataset_name = "al_pinhole"
    build_date = str(date.today())
    model_name = f"{dataset_name}_{model_arch}_{build_date}"
    date_dir = os.path.join(Base_dir, 'trainedModel', 'Phase_3', 'busbar_al', 'pinhole')
    if not os.path.exists(date_dir):
        os.makedirs(date_dir)
    
    # Determine the next run folder (run_1, run_2, etc.)
    existing_runs = [d for d in os.listdir(date_dir) if os.path.isdir(os.path.join(date_dir, d))]
    run_number = len(existing_runs) + 1
    run_dir = os.path.join(date_dir, f"run_{run_number}_dice_{dice_loss_weightage}_alpha_{alphaa}_beta_{betaa}_gamma_{gammaa}")
    os.makedirs(run_dir)
    metrics=[]
    Total_training_time=0
    

    
    for epoch in range(epochs):
        start = datetime.datetime.now()
        batch_costs = []
        batch_accuracies = []
        batch_accuracies_pixel = []
        
        with tqdm(total=len(trainDatasetLoader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
            for X, Y in trainDatasetLoader:
                gpu_X = X.to(device)
                gpu_Y = Y.to(device)
                ##cost calculation
                
                CustomSegmentation.train()
                optimizer.zero_grad()
                
        
                hypothesis = CustomSegmentation(gpu_X)
         
                cost = (loss_fn(hypothesis, gpu_Y)*(1-dice_loss_weightage)) + (loss_dice_fn(hypothesis, gpu_Y)*dice_loss_weightage)
                
                # print('GPU_Y_unique',np.unique(gpu_Y.cpu().numpy()))
                # print('Hypothesis_unique',np.unique(hypothesis.detach().cpu().numpy()))
                # cost = (dice_loss_weightage * bce_loss_fn) + ((1-dice_loss_weightage) * dice_loss_fn) # For combined loss BCE and DIce


                cost.backward()
                
                optimizer.step()
                
                
                # Accumulate metrics for the current batch
                batch_costs.append(cost.cpu().detach().numpy())
                accuracy = IOU(gpu_Y.cpu().detach().numpy(), hypothesis.cpu().detach().numpy())
                pixel_accuracy = accuracy_score(
                    gpu_Y.cpu().detach().numpy().flatten(),  # Ground truth labels, flattened
                    (hypothesis.cpu().detach().numpy().flatten() > 0.5).astype(int)
                )
                
                
                batch_accuracies.append(accuracy)
                batch_accuracies_pixel.append(pixel_accuracy)  
                
                # Update the tqdm progress bar
                pbar.update(1)
                pbar.set_postfix(cost=cost.item(), accuracy=accuracy, pixel_accuracy=pixel_accuracy)      
                          
                CustomSegmentation.eval()
                accuracy = IOU(gpu_Y.cpu().detach().numpy(), hypothesis.cpu().detach().numpy())
                batch_accuracies.append(accuracy)
                # Update the tqdm progress bar
                pbar.update(1)
                pbar.set_postfix(cost=cost.item(), accuracy=accuracy)
                input_image = gpu_X[0].detach().permute(1, 2, 0).squeeze(0).cpu().numpy().astype(np.float32).copy()
                input_image =  cv2.normalize(input_image, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)

                label_image = gpu_Y[0][0].detach().cpu().numpy ().astype(np.float32).copy()
                label_image =  cv2.normalize(label_image, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
                label_image = cv2.cvtColor(label_image, cv2.COLOR_GRAY2BGR)

                predict_image = hypothesis[0][0].detach().cpu().numpy ().astype(np.float32).copy()
                predict_image =  cv2.normalize(predict_image, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
                predict_image = cv2.cvtColor(predict_image, cv2.COLOR_GRAY2BGR)
                
                # Create overlay with red contours on original image
                predict_mask = (predict_image[:, :, 0] > 127).astype(np.uint8) * 255
                contours, _ = cv2.findContours(predict_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                overlay_image = input_image.copy()
                if len(input_image.shape) == 2 or input_image.shape[2] == 1:
                    overlay_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
                cv2.drawContours(overlay_image, contours, -1, (0, 0, 255), 2)  # Red contours, thickness 2

                cv2.imshow("result", input_image)
                cv2.imshow("label", label_image)
                cv2.imshow("predict", predict_image)
                cv2.imshow("overlay", overlay_image)
                cv2.waitKey(10)
                
                
        # Calculate mean metrics for the epoch
        avg_cost = np.mean(batch_costs)
        avg_acc = np.mean(batch_accuracies)
        avg_acc_pixel = np.mean(batch_accuracies_pixel)
            
        end = datetime.datetime.now()
        time_per_epoch=end-start
        Total_training_time += time_per_epoch.total_seconds()
        print(f"Total training time so far: {Total_training_time:.2f} seconds")
            #input_image = gpu_X[0].detach().permute(1, 2, 0).squeeze(0).cpu().numpy().astype(np.float32).copy()
            #input_image =  cv2.normalize(input_image, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
            #cv2.imshow("input", input_image)
            #cv2.waitKey(10)


        # Validation Loop
        val_costs = []
        val_accuracies = []
        CustomSegmentation.eval()  # Ensure model is in evaluation mode
        with torch.no_grad():  # Disable gradient calculation
            for X_val, Y_val in validDatasetLoader:
                gpu_X_val = X_val.to(device)
                gpu_Y_val = Y_val.to(device)
                
                val_output = CustomSegmentation(gpu_X_val)
                val_loss = loss_fn(val_output, gpu_Y_val)

                val_accuracy = IOU(gpu_Y_val.cpu().detach().numpy(), val_output.cpu().detach().numpy())

                val_costs.append(val_loss.cpu().detach().numpy())
                val_accuracies.append(val_accuracy)

                # input_image = gpu_X_val[0].detach().permute(1, 2, 0).squeeze(0).cpu().numpy().astype(np.float32).copy()
                # input_image =  cv2.normalize(input_image, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)

                # label_image = gpu_Y_val[0][0].detach().cpu().numpy ().astype(np.float32).copy()
                # label_image =  cv2.normalize(label_image, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
                # label_image = cv2.cvtColor(label_image, cv2.COLOR_GRAY2BGR)

                # predict_image = val_output[0][0].detach().cpu().numpy ().astype(np.float32).copy()
                # predict_image =  cv2.normalize(predict_image, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
                # predict_image = cv2.cvtColor(predict_image, cv2.COLOR_GRAY2BGR)

                # cv2.imshow("result", input_image)
                # cv2.imshow("label", label_image)
                # cv2.imshow("predict", predict_image)
                # cv2.waitKey(10)
                
        # Calculate mean validation metrics
        val_cost = np.mean(val_costs)
        val_acc = np.mean(val_accuracies)        
   

        
        # # Enforce minimum learning rate
        # for param_group in optimizer.param_groups:
        #     if param_group['lr'] < min_lr:
        #         param_group['lr'] = min_lr

        # # Print the current learning rate to track it
        # current_lr = optimizer.param_groups[0]['lr']
        

        # input_image = gpu_X_val[0].detach().permute(1, 2, 0).squeeze(0).cpu().numpy().astype(np.float32).copy()
        # input_image =  cv2.normalize(input_image, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)

        # label_image = gpu_Y_val[0][0].detach().cpu().numpy ().astype(np.float32).copy()
        # label_image =  cv2.normalize(label_image, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        # label_image = cv2.cvtColor(label_image, cv2.COLOR_GRAY2BGR)

        # predict_image = val_output[0][0].detach().cpu().numpy ().astype(np.float32).copy()
        # predict_image =  cv2.normalize(predict_image, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        # predict_image = cv2.cvtColor(predict_image, cv2.COLOR_GRAY2BGR)

        # cv2.imshow("result", input_image)
        # cv2.imshow("label", label_image)
        # cv2.imshow("predict", predict_image)
        # cv2.waitKey(10)



        print('Epoch: %01d, Train cost: %.9f, Train accuracy: %.9f, Val cost: %.9f, Val accuracy: %.9f, learning rate: %.5f' % (epoch + 1, avg_cost, avg_acc, val_cost, val_acc, learningRate))

        # Save the best model
        if avg_acc > best_train_acc:
            if best_model_path is not None:
                os.remove(best_model_path)
            best_train_acc = avg_acc
            # best_model_path = os.path.join(run_dir,f"{model_name}_trAcc_{best_train_acc:.5f}_trloss_{avg_cost:.5f}_epoch_{epoch+1}_lr_{learningRate:,.4f}_time_{Total_training_time / 60:.2f}M_best.pth")
            # torch.save(CustomSegmentation.state_dict(), best_model_path)
            print(f"Saved best model at epoch {epoch+1} with training accuracy: {best_train_acc:.9f}")
            
        # Early stopping based on validation accuracy
        if avg_acc > targetAccuracy:  # Use train accuracy for early stopping
            final_accuracy = avg_acc
            final_cost = avg_cost
            print(f'Target accuracy achieved at epoch number {epoch + 1}', ', final train accuracy=', final_accuracy, ', final cost=', final_cost)
            break
        # Append metrics to the list for CSV
        metrics.append({
            'Epoch': epoch + 1,
            'Train Accuracy': avg_acc,
            'Validation Accuracy': val_acc,
            'Train Cost': avg_cost,
            'Validation Cost': val_cost,
            'Learning Rate': learningRate
        })

        if epoch%5==0:
            metrics_df = pd.DataFrame(metrics)
            csv_path = os.path.join(run_dir, "training_metrics.csv")
            metrics_df.to_csv(csv_path, index=False)
            print(f"Saved training metrics till epoch:{epoch} to {csv_path}")
            plt.figure(figsize=(10, 6))
            plt.plot(metrics_df['Epoch'], metrics_df['Train Accuracy'], label='Training Accuracy')
            plt.plot(metrics_df['Epoch'], metrics_df['Validation Accuracy'], label='Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Training and Validation Accuracy')
            # plt.ylim(0.001, 0.99)
            plt.legend()
            plt.grid(True)
            # Save the figure without displaying it
            accuracy_fig_path = os.path.join(run_dir, 'accuracy_plot.png')
            plt.savefig(accuracy_fig_path)
            plt.close()  # Close the figure to free up memory
            print(f'Saved accuracy plot to {accuracy_fig_path}')
            
            # Plot Training and Validation Loss
            plt.figure(figsize=(10, 6))
            plt.plot(metrics_df['Epoch'], metrics_df['Train Cost'], label='Training Loss')
            plt.plot(metrics_df['Epoch'], metrics_df['Validation Cost'], label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.grid(True)

            # Save the figure in the same directory
            loss_fig_path = os.path.join(run_dir, 'loss_plot.png')
            plt.savefig(loss_fig_path)
            plt.close()  # Close the figure to free up memory
            print(f'Saved loss plot to {loss_fig_path}')
                    
    print(f"Total training time: {Total_training_time / 60:.2f} minutes")
    # model save
    # CustomSegmentation.eval()
    # compiled_model = torch.jit.script(CustomSegmentation)
    # torch.jit.save(compiled_model, "E://busbar_al_welding_prediction.pt")


    last_model_path = os.path.join(run_dir,f"{model_name}_trAcc_{avg_acc:.5f}_trloss_{avg_cost:.5f}_epoch_{epoch+1}_lr_{learningRate:.4f}_time_{Total_training_time / 60:.2f}M_last.pth")
    torch.save(CustomSegmentation.state_dict(), last_model_path)
    print(f"Saved last model after epoch {epoch+1}")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics)
    csv_path = os.path.join(run_dir, "training_metrics.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"Saved training metrics to {csv_path}")
    
    # Plot Training and Validation Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_df['Epoch'], metrics_df['Train Accuracy'], label='Training Accuracy')
    plt.plot(metrics_df['Epoch'], metrics_df['Validation Accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.ylim(0.1, 0.99)
    plt.legend()
    plt.grid(True)

    # Save the figure in the same directory
    accuracy_fig_path = os.path.join(run_dir, 'accuracy_plot.png')
    plt.savefig(accuracy_fig_path)
    print(f'Saved accuracy plot to {accuracy_fig_path}')

    # Plot Training and Validation Loss
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_df['Epoch'], metrics_df['Train Cost'], label='Training Loss')
    plt.plot(metrics_df['Epoch'], metrics_df['Validation Cost'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # Save the figure in the same directory
    loss_fig_path = os.path.join(run_dir, 'loss_plot.png')
    plt.savefig(loss_fig_path)
    print(f'Saved loss plot to {loss_fig_path}')

    # Plot Learning Rate
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_df['Epoch'], metrics_df['Learning Rate'], label='Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True)

    # Save the figure in the same directory
    lr_fig_path = os.path.join(run_dir, 'learning_rate_plot.png')
    plt.savefig(lr_fig_path)
    print(f'Saved learning rate plot to {lr_fig_path}')

    plt.show()
    
if __name__ == '__main__':
    train()