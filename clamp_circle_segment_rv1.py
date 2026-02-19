import cv2
import torch
import random
import numpy as np
import time
import datetime
from datetime import date
from util.CircleDataset import CircleDataset
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from model.SegNextV2 import SegNextV2
from torchsummary import summary
from util.helper import IOU
from util.helper import DiceLoss
from util.helper import FocalTverskyLoss, save_checkpoint, load_checkpoint, apply_motion_blur, mIoULoss
import matplotlib.pyplot as plt
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score

def train(load_pre_trained_model=False):
    #USING GPU # 1
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
    imageHeight=640
    batchSize=32
    #learningRate = 0.003 * batchSize / 32 
    learningRate = 0.003
    epochs = 100
    targetAccuracy = 0.991
    #hyper parameter
    blur_probability = 0.5 

    alphaa=0.7
    betaa=0.3
    gammaa=2
    
    transformAugCollection = []
    transformAugCollection.append(transforms.Resize((imageHeight,imageWidth), interpolation=Image.NEAREST)) 
    # transformAugCollection.append(transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)) 
    # #transformAugCollection.append(transforms.RandomGrayscale(p=0.2))
    # #transformAugCollection.append(transforms.GaussianBlur(kernel_size=3))
    # transformAugCollection.append(transforms.RandomAdjustSharpness(p=0.2, sharpness_factor=2))
    # transformAugCollection.append(transforms.RandomAutocontrast(p=0.2))
    transformAugCollection.append(transforms.ToTensor()) 
    transAugProcess = transforms.Compose(transformAugCollection)

    transformNormalCollection = []
    transformNormalCollection.append(transforms.Resize((imageHeight,imageWidth), interpolation=Image.NEAREST)) 
    transformNormalCollection.append(transforms.ToTensor()) 
    transNormalProcess = transforms.Compose(transformNormalCollection)


 
    
    trainDataset = CircleDataset(path="/home/saqib/Projects/sebang_welding/Sebang_Latest_Mar25/clamp/circle/dataset",
                                      transform=transAugProcess,
                                      category="train",num_images=None)

    trainDatasetLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True, drop_last=True)

    validDataset = CircleDataset(path="/home/saqib/Projects/sebang_welding/Sebang_Latest_Mar25/clamp/circle/dataset",
                                       transform=transNormalProcess,
                                       category="test",
                                       useVFlip=False,
                                       useHFlip=False,num_images=None)

    validDatasetLoader = DataLoader(validDataset, batch_size=1, shuffle=True, drop_last=False)


    # summary
    totalTrainBatch = len(trainDatasetLoader)
    totalValidBatch = len(validDatasetLoader)
    # CustomSegmentation = CustomSegmentationV4().to(device)
    CustomSegmentation = SegNextV2(inputWidth=imageWidth,
                                    inputHeight=imageHeight,
                                    num_class=1,
                                    embed_dims=[32, 64, 128, 128],
                                    ffn_ratios=[2, 2, 2, 2],
                                    depths=[3,3,3,2]).to(device)
    
    # CustomSegmentation = CustomSegmentationV4().to(device)
    
    print('==== model info ====')
    summary(CustomSegmentation, (3, imageHeight, imageWidth))
    print('====================')

    print('total_batch_size = ', totalTrainBatch)
    print('val_batch_size = ', totalValidBatch)
    print('learning rate = ', learningRate)
    # summaru

    # loss function setting
    
    loss_fn = FocalTverskyLoss(alpha=alphaa, beta=betaa, gamma=gammaa).to(device)
    loss_dice_fn = DiceLoss()
    dice_loss_weightage=1

    optimizer = torch.optim.RAdam(CustomSegmentation.parameters(), lr=learningRate)

    #agc_optimizer= AGC(CustomSegmentation.parameters(), optimizer, clipping=0.5)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, threshold=5e-3, threshold_mode='rel', patience=4, verbose=True, min_lr=0.00001)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0009, max_lr=learningRate, step_size_up=totalTrainBatch*5, mode='triangular')

    # Initialize variables to track the best model
    best_train_acc = 0.0  # Or use np.inf if tracking the lowest loss
    best_model_path = None
    Base_dir='/home/saqib/Projects/sebang_welding/'
    model_arch = "SegNextV2"
    dataset_name = "clamp_circle"
    build_date = str(date.today())
    model_name = f"{dataset_name}_{model_arch}_{build_date}_accuracy"
    date_dir = os.path.join(Base_dir, 'trainedModel/Phase_2025/clamp/circle')
    if not os.path.exists(date_dir):
        os.makedirs(date_dir)
    

    existing_runs = [d for d in os.listdir(date_dir) if os.path.isdir(os.path.join(date_dir, d))]
    run_number = len(existing_runs) + 1
    run_dir = os.path.join(date_dir, f"run_clamp_{run_number}")
    os.makedirs(run_dir)
    
    # checkpoint_dir = os.path.join(run_dir, 'Checkpoints')
    # if not os.path.exists(checkpoint_dir):
    #     os.makedirs(checkpoint_dir)

    # #################### Load pretrained model ###########################
    # if load_pre_trained_model:
    #     ####################### Pretrained model path######################################################################################
    #     pretrained_model_path = "/mnt/d/hvs/Hyvsion_Projects/Welding_Project/trainedModel/Phase_2/clamp/circle/run_clamp_3/clamp_circle_SegNextV2_2024-11-03_accuracy_train_0.92044_loss_0.04157_epoch_1_lr_0.003_time_41.63M_alpha_0.7_beta_0.3_gamma_2_best.pth"
    #     # # ###################################################################################################################################
    
    #     if os.path.isfile(pretrained_model_path):
    #         print(f"Loading pre-trained model from {pretrained_model_path}")
    #         CustomSegmentation = torch.load(pretrained_model_path, map_location=device)  # Load the state_dict
    #     else:
    #         print("No pre-trained model found at the specified path. Training from scratch.")
    # ##########################################################################                    
    
    metrics=[]
    Total_training_time=0
    for epoch in range(epochs):
        start = datetime.datetime.now()
        batch_costs = []
        batch_accuracies = []
        batch_accuracies_pixel = []
        #current_lr = scheduler.get_last_lr()[0]
        with tqdm(total=len(trainDatasetLoader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar: 
            for X, Y in trainDatasetLoader:
                gpu_X = X.to(device)
                gpu_Y = Y.to(device)
                
                # ############################## RGB BGR Test#########################
                # print(f"Epoch {epoch + 1}, Batch Channel Means (R/G/B or B/G/R): {gpu_X.mean(dim=(0, 2, 3))}")
                # ################################################################
                
                # ####################### Motion blur#################
                # for img_idx in range(gpu_X.size(0)):
                #     if random.random()<blur_probability:
                #         motion_random_kernel= random.randint(5, 10)
                #         image_np = gpu_X[img_idx].cpu().detach().numpy().transpose(1, 2, 0)  # Convert tensor to numpy (H, W, C)
                #         #image_np = cv2.normalize(image_np, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
                #         blurred_image = apply_motion_blur(image_np, kernel_size=motion_random_kernel) 
                #         blurred_image = blurred_image.astype(np.float32)
                #         # cv2.imshow("input", blurred_image)
                #         # cv2.waitKey(10)
                        
                #         gpu_X[img_idx] = torch.from_numpy(blurred_image.transpose(2, 0, 1)).to(device)  
                # ######################################################
                ##cost calculation
                CustomSegmentation.train()
                optimizer.zero_grad()
                
        
                hypothesis = CustomSegmentation(gpu_X)
                #cost = (loss_fn(hypothesis, gpu_Y) + loss_iou_fn(hypothesis, gpu_Y))/2
                cost = (loss_fn(hypothesis, gpu_Y)*(1-dice_loss_weightage)) + (loss_dice_fn(hypothesis, gpu_Y)*dice_loss_weightage)
                
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
                
                pbar.update(1)
                pbar.set_postfix(cost=cost.item(), accuracy=accuracy, pixel_accuracy=pixel_accuracy)
                
                #scheduler.step()

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
                val_loss = (loss_fn(val_output, gpu_Y_val)*(1-dice_loss_weightage)) + (loss_dice_fn(val_output, gpu_Y_val)*dice_loss_weightage)

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
        val_cost = np.mean(val_costs)
        val_acc = np.mean(val_accuracies)         
        #scheduler.step(avg_cost)
        
        
        # # Save checkpoint after every epoch
        # checkpoint_state = {
        #     'epoch': epoch,
        #     'model_state_dict': CustomSegmentation.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict()
        # }
        # save_checkpoint(checkpoint_state, CustomSegmentation, checkpoint_dir, epoch)        

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



        print('Epoch: %01d, Train cost: %.9f, Train accuracy: %.9f, Pixel accuracy: %.9f, Val cost: %.9f, Val accuracy: %.9f, learning rate: %.9f' % (epoch + 1, avg_cost, avg_acc, avg_acc_pixel, val_cost, val_acc, learningRate))

        # Save the best model
        if avg_acc > best_train_acc:
            if best_model_path is not None:
                os.remove(best_model_path)
            best_train_acc = avg_acc
            best_model_path = os.path.join(run_dir,f"{model_name}_acc_{best_train_acc:.5f}_loss_{avg_cost:.5f}_E_{epoch+1}_time_{Total_training_time / 60:.2f}min_W{imageWidth}_H{imageHeight}_best.pth")
            best_model_path_state_dict = os.path.join(run_dir,f"{model_name}_acc_{best_train_acc:.5f}_loss_{avg_cost:.5f}_E_{epoch+1}_time_{Total_training_time / 60:.2f}min_W{imageWidth}_H{imageHeight}_best_State_Dict.pth")

            torch.save(CustomSegmentation, best_model_path)
            torch.save(CustomSegmentation.state_dict(), best_model_path_state_dict)
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


    last_model_path = os.path.join(run_dir,f"{model_name}_acc_{avg_acc:.5f}_loss_{avg_cost:.5f}_E_{epoch+1}_time_{Total_training_time / 60:.2f}min_W{imageWidth}_H{imageHeight}_last.pth")
    torch.save(CustomSegmentation, last_model_path)
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
    plt.ylim(0.4, 0.99)
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