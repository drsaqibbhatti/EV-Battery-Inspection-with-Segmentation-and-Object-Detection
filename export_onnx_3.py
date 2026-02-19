from model.SegNextV2 import SegNextV2
import torch
import io
import base64
import json

# Model configuration
imageWidth=640
imageHeight=128

model_path = "/mnt/d/hvs/Hyvsion_Projects/Welding_Project/trainedModel/Phase_2025/busbar_al/weld/run_busbar_al_2_W640_H128/busbar_al_width_SegNextV2_2025-03-27_Acc_0.986856470_loss_0.006624890_E125_time_577.9M_W640_H128_best.pth"
onnx_model_path = "/mnt/d/hvs/Hyvsion_Projects/Welding_Project/trainedModel/Phase_2025/busbar_al/weld/run_busbar_al_2_W640_H128/busbar_al_weld_SegN.onnx"
dsm_model_path = "/mnt/d/hvs/Hyvsion_Projects/Welding_Project/trainedModel/Phase_2025/busbar_al/weld/run_busbar_al_2_W640_H128/busbar_al_weld_SegN.dsm"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the full model
model = torch.load(model_path, map_location=device)
model.to(device)
model.eval()

# Dummy input for ONNX export
trace_input = torch.randn(1, 3, imageHeight, imageWidth).to(device)
 
# Export to ONNX
buffer = io.BytesIO()
#compiled_model = torch.jit.script(model)
torch.onnx.export(
    model,
    trace_input,
    buffer,
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output']
)
#keep_initializers_as_inputs= True,    
# dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
# Convert buffer to base64 for DSM format
buffer.seek(0)
bytes_data = buffer.read()
base64Model = base64.b64encode(bytes_data).decode('ascii')

# Create custom DSM JSON structure
hvs_model_json = {
    "inputHeight": imageHeight,
    "inputWidth": imageWidth,
    # "Channel": 3,
    # "Freeze": True,
    "module": base64Model,
    # "ModelType": "ONNX",
    # "MaxLabelCount": 1,
    "classNames": ["weld"]
}

# Save DSM format
jsonFormatString = json.dumps(hvs_model_json)
with open(dsm_model_path, "w") as text_file:
    text_file.write(jsonFormatString)

# Save the ONNX model to disk
with open(onnx_model_path, "wb") as f:
    f.write(bytes_data)

print(f"ONNX model exported to {onnx_model_path}")
print(f"DSM format saved to {dsm_model_path}")
