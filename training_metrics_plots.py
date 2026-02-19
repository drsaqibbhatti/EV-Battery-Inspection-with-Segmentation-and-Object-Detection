import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the metrics CSV file
csv_path = "/mnt/d/hvs/Hyvsion_Projects/Welding_Project/sebang_welding_git/trainedModel/clamp/2024-09-05/run_clamp_1/training_metrics.csv"  # Replace with your actual CSV path
metrics_df = pd.read_csv(csv_path)
save_dir = os.path.dirname(csv_path)
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
accuracy_fig_path = os.path.join(save_dir, 'accuracy_plot.png')
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
loss_fig_path = os.path.join(save_dir, 'loss_plot.png')
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
lr_fig_path = os.path.join(save_dir, 'learning_rate_plot.png')
plt.savefig(lr_fig_path)
print(f'Saved learning rate plot to {lr_fig_path}')

plt.show()
