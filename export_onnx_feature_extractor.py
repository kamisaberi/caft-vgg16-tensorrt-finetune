# export_onnx_feature_extractor.py
import torch
import torch.nn as nn
import torchvision.models as models

# --- Configuration ---
ONNX_OUTPUT_PATH = "vgg16_bn_feature_extractor.onnx"

print(f"Exporting VGG16-BN feature extractor to ONNX format at {ONNX_OUTPUT_PATH}...")

# 1. Load the pretrained VGG16 model
vgg16_bn = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT)

# 2. Create a sequential module for the frozen part
class FeatureExtractorWrapper(nn.Module):
    def __init__(self, features_module, avgpool_module):
        super().__init__()
        self.features = features_module
        self.avgpool = avgpool_module

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1) # Flatten the output
        return x

feature_extractor_py = FeatureExtractorWrapper(vgg16_bn.features, vgg16_bn.avgpool)
feature_extractor_py.eval()

# 3. Create a dummy input tensor
# We will define the batch size as dynamic for flexibility
dummy_input = torch.randn(1, 3, 224, 224, requires_grad=False)

# 4. Export the model to ONNX
torch.onnx.export(
    feature_extractor_py,
    dummy_input,
    ONNX_OUTPUT_PATH,
    export_params=True,
    opset_version=11, # A commonly used opset version
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},  # Make the batch dimension dynamic
        'output': {0: 'batch_size'}
    }
)

print(f"ONNX model saved successfully to {ONNX_OUTPUT_PATH}")