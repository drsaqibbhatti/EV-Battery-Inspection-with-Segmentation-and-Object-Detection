import onnxruntime as ort
import numpy as np
import time

# Path to your ONNX model
onnx_model_path = "/mnt/d/hvs/Hyvsion_Projects/Welding_Project/trainedModel/Phase_2/clamp/width/run_clamp_8/model/model_clamp_weld_v2.onnx"

# Load the ONNX model
session = ort.InferenceSession(onnx_model_path)

# Get model input details
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
input_type = session.get_inputs()[0].type

# Prepare a dummy input based on the expected input shape and type
# Replace `np.float32` with the appropriate data type if needed
dummy_input = np.random.rand(*[dim if dim else 1 for dim in input_shape]).astype(np.float32)

# Warm up the model (optional, to eliminate cold-start overhead)
for _ in range(5):
    session.run(None, {input_name: dummy_input})

# Measure inference time
num_runs = 100  # Number of runs to calculate average inference time
total_time = 0

for _ in range(num_runs):
    start_time = time.time()
    outputs = session.run(None, {input_name: dummy_input})
    total_time += time.time() - start_time

# Calculate average inference time in milliseconds
average_inference_time = (total_time / num_runs) * 1000
print(f"Average Inference Time: {average_inference_time:.2f} ms")
